"""Training script for the CUB-200-2011 bird species classifier.

This module is responsible for ONE thing: training BirdClassifier
(model.py) on data provided by dataset.py, and saving the best checkpoint
to disk. It contains no dataset-parsing logic and no model-architecture
logic — only the training loop and its supporting utilities.

Usage:
    python -m src.classification.train --data-dir data/CUB_200_2011 \
        --output-dir models --epochs 30
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.classification.dataset import IMAGE_SIZE, get_dataloaders
from src.classification.model import create_model


@dataclass
class TrainConfig:
    """Hyperparameters and paths controlling a single training run."""

    data_dir: str
    output_dir: str
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    num_workers: int = 4
    image_size: int = IMAGE_SIZE
    early_stopping_patience: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> tuple[float, float]:
    """Run a single training epoch with mixed precision.

    Args:
        model: The classifier being trained.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for automatic mixed precision.
        device: Device string ("cuda" or "cpu").

    Returns:
        Tuple of (average training loss, training accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate the model on a validation/test DataLoader.

    Args:
        model: The classifier being evaluated.
        loader: Validation or test DataLoader.
        criterion: Loss function.
        device: Device string ("cuda" or "cpu").

    Returns:
        Tuple of (average loss, accuracy) over the full loader.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    config: TrainConfig,
    epoch: int,
    best_accuracy: float,
    output_path: Path,
) -> None:
    """Persist model weights and metadata needed for inference.

    The checkpoint format is consumed unmodified by infer.py, so any
    change here must be mirrored there.

    Args:
        model: Trained classifier.
        class_names: Ordered list of class names matching label indices.
        config: Training configuration used to produce this checkpoint.
        epoch: Epoch at which this checkpoint was saved.
        best_accuracy: Best validation accuracy achieved so far.
        output_path: File path to write the checkpoint to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": config.image_size,
        "epoch": epoch,
        "best_accuracy": best_accuracy,
    }
    torch.save(checkpoint, output_path)


def run_training(config: TrainConfig) -> None:
    """Execute the full training loop with early stopping.

    Args:
        config: Fully populated training configuration.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "bird_classifier_best.pt"

    train_loader, test_loader, class_names = get_dataloaders(
        root_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
    )

    model = create_model(num_classes=len(class_names), pretrained=True)
    model.to(config.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=(config.device == "cuda"))

    best_accuracy = 0.0
    epochs_without_improvement = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config.device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, config.device)
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{config.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| lr={scheduler.get_last_lr()[0]:.6f} "
            f"| time={elapsed:.1f}s"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(model, class_names, config, epoch, best_accuracy, checkpoint_path)
            print(f"  -> New best model saved (val_acc={best_accuracy:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {config.early_stopping_patience} epochs)."
            )
            break

    model.load_state_dict(best_state_dict)
    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint saved at: {checkpoint_path}")


def parse_args() -> TrainConfig:
    """Parse command-line arguments into a TrainConfig.

    Returns:
        Populated TrainConfig instance.
    """
    parser = argparse.ArgumentParser(description="Train the bird species classifier.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the extracted CUB_200_2011 directory.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Directory to save model checkpoints."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    run_training(parse_args())
