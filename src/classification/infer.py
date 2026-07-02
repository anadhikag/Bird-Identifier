"""Inference utilities for the bird species classifier.

This module is responsible for ONE thing: loading a trained checkpoint
(produced by train.py) and running predictions on new images, returning
top-1 and top-5 species predictions with confidence scores.

It intentionally does NOT implement Grad-CAM yet. The `BirdInference`
class exposes the underlying `self.model` (a BirdClassifier from
model.py) so a future Grad-CAM module can attach hooks to
`self.model.get_feature_extractor()` without any changes to this file's
public interface. Downstream consumers (RAG, Gemini, Streamlit, and
later the FastAPI backend) should depend only on `BirdInference.predict`
and the `Prediction` / `InferenceResult` dataclasses below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image

from src.classification.dataset import get_eval_transforms
from src.classification.model import BirdClassifier, create_model


@dataclass
class Prediction:
    """A single species prediction with its confidence score."""

    class_name: str
    class_index: int
    confidence: float


@dataclass
class InferenceResult:
    """Full result of a classification call, ready for downstream RAG/Gemini use."""

    top1: Prediction
    top5: list[Prediction]


class BirdInference:
    """Loads a trained checkpoint and serves species predictions.

    This class is the single entry point Stage 1's Streamlit app (and
    later Stage 2's FastAPI backend) should use to run classification.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        """Load model weights and metadata from a training checkpoint.

        Args:
            checkpoint_path: Path to the .pt file saved by train.py.
            device: Device to run inference on. Defaults to "cuda" if
                available, otherwise "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.class_names: list[str] = checkpoint["class_names"]
        self.image_size: int = checkpoint["image_size"]

        self.model: BirdClassifier = create_model(
            num_classes=checkpoint["num_classes"], pretrained=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_eval_transforms(image_size=self.image_size)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert a PIL image into a normalized, batched tensor.

        Args:
            image: Input PIL image, assumed to already contain a cropped
                bird (per the current pipeline stage — YOLOv8 detection
                and cropping happen upstream in a later iteration).

        Returns:
            Tensor of shape (1, 3, H, W) on `self.device`.
        """
        tensor = self.transform(image.convert("RGB"))
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> InferenceResult:
        """Classify a single bird image.

        Args:
            image: Input PIL image containing a cropped bird.
            top_k: Number of top predictions to return.

        Returns:
            InferenceResult containing the top-1 prediction and the
            top-k ranked predictions with confidence scores.
        """
        input_tensor = self._preprocess(image)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        predictions = [
            Prediction(
                class_name=self.class_names[int(idx)],
                class_index=int(idx),
                confidence=float(prob),
            )
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
        ]

        return InferenceResult(top1=predictions[0], top5=predictions)

    def predict_from_path(self, image_path: Union[str, Path], top_k: int = 5) -> InferenceResult:
        """Convenience wrapper to classify an image given a file path.

        Args:
            image_path: Path to an image file on disk.
            top_k: Number of top predictions to return.

        Returns:
            InferenceResult for the loaded image.
        """
        image = Image.open(image_path)
        return self.predict(image, top_k=top_k)
