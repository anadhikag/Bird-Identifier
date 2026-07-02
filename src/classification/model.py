"""EfficientNetV2-S classifier definition for bird species classification.

This module is responsible for ONE thing: defining and constructing the
neural network architecture. It has no knowledge of datasets, training
loops, or inference/decoding logic.

The model exposes its convolutional feature extractor via
`get_feature_extractor()` so that a future Grad-CAM implementation can
attach forward/backward hooks to a specific layer WITHOUT modifying this
file or any file that depends on it.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class BirdClassifier(nn.Module):
    """EfficientNetV2-S backbone with a custom classification head.

    Attributes:
        backbone: The torchvision EfficientNetV2-S module (features +
            pooling + classifier), kept as a single attribute so future
            Grad-CAM code can target `backbone.features` directly.
        num_classes: Number of output bird species classes.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True) -> None:
        """Initialize the classifier.

        Args:
            num_classes: Number of bird species to classify (200 for
                CUB-200-2011).
            pretrained: If True, initialize the backbone with ImageNet
                pretrained weights. Set to False when loading a
                fine-tuned checkpoint, since the checkpoint already
                contains trained weights.
        """
        super().__init__()
        self.num_classes = num_classes

        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Batch of input images, shape (B, 3, H, W), already
                normalized with ImageNet statistics.

        Returns:
            Raw, unnormalized class logits of shape (B, num_classes).
        """
        return self.backbone(x)

    def get_feature_extractor(self) -> nn.Module:
        """Expose the convolutional feature layers for future Grad-CAM use.

        Returns:
            The `features` submodule of the EfficientNetV2-S backbone.
            The last block of this submodule is the conventional
            Grad-CAM target layer for EfficientNet-style architectures.
        """
        return self.backbone.features


def create_model(num_classes: int = 200, pretrained: bool = True) -> BirdClassifier:
    """Factory function to construct a BirdClassifier.

    Kept separate from the class definition so training and inference
    code construct the model identically, without duplicating
    architecture-configuration logic in multiple places.

    Args:
        num_classes: Number of bird species to classify.
        pretrained: If True, initialize the backbone with ImageNet
            pretrained weights.

    Returns:
        An initialized BirdClassifier instance.
    """
    return BirdClassifier(num_classes=num_classes, pretrained=pretrained)
