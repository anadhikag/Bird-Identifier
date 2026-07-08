"""Grad-CAM visualization for the trained bird species classifier.

This module is responsible for ONE thing: producing Grad-CAM heatmaps
and overlays that explain which regions of an input image drove the
classifier's prediction.

It reuses the existing project components rather than duplicating them:
- Model architecture and checkpoint loading come from `model.create_model`
  via `infer.BirdInference`, so there is exactly one place that knows how
  to construct a `BirdClassifier` from a checkpoint.
- Image preprocessing comes from `infer.BirdInference.transform`, which
  is itself built from `dataset.get_eval_transforms`, so there is exactly
  one place that defines how an image becomes a model input tensor.
- The Grad-CAM target layer comes from `model.BirdClassifier.get_feature_extractor`,
  so no layer name is hardcoded here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
# pytorch_grad_cam imports are deferred into __init__ and generate()
# to prevent cv2 / libxcb from loading at module import time.
# See inline comments at each deferred import site.

from src.classification.infer import BirdInference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradCAMResult:
    """Result of running Grad-CAM on a single image.

    Attributes:
        predicted_class: Human-readable name of the predicted species.
        confidence: Softmax probability of the predicted class, in
            [0.0, 1.0].
        heatmap: Grad-CAM activation map as a 2D numpy array of shape
            (H, W), with values in [0.0, 1.0], at the model's input
            resolution.
        overlay: The Grad-CAM heatmap blended on top of the (resized)
            input image, as a PIL Image.
    """

    predicted_class: str
    confidence: float
    heatmap: np.ndarray
    overlay: Image.Image


class GradCAMGenerator:
    """Generates Grad-CAM explanations for the trained bird classifier.

    Loads a trained checkpoint once and reuses it across all subsequent
    `generate` calls, mirroring the lifecycle of `BirdInference`.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        """Load the checkpoint, class names, model, and Grad-CAM hooks.

        Args:
            checkpoint_path: Path to the .pt checkpoint file saved by
                train.py (e.g. "models/bird_classifier_best.pt").
            device: Device to run on. Defaults to "cuda" if available,
                otherwise "cpu".

        Raises:
            FileNotFoundError: If checkpoint_path does not exist.
        """
        # Reuses the existing checkpoint/model/preprocessing loading
        # logic instead of duplicating it. `create_model` is invoked
        # exactly once, inside BirdInference.
        self._inference = BirdInference(checkpoint_path=checkpoint_path, device=device)
        self._model = self._inference.model
        self._model.eval()

        # Deferred: importing pytorch_grad_cam at module scope loads cv2
        # transitively (via the package __init__ re-exporting all submodules,
        # each of which imports pytorch_grad_cam.utils.image, which imports cv2).
        # cv2 links against libxcb.so.1, an X11/GUI library absent on Azure
        # App Service headless Linux, causing an ImportError at startup.
        # Deferring to here means cv2 only loads on the first /predict call.
        from pytorch_grad_cam import GradCAM
        target_layer = self._model.get_feature_extractor()[-1]
        self._cam = GradCAM(model=self._model, target_layers=[target_layer])

        self._last_overlay: Optional[Image.Image] = None

        logger.info(
            "GradCAMGenerator ready on device=%s with target layer=%s",
            self._inference.device,
            type(target_layer).__name__,
        )

    def generate(self, image: Image.Image) -> GradCAMResult:
        """Run Grad-CAM on a single image.

        Args:
            image: Input PIL image containing a cropped bird.

        Returns:
            GradCAMResult containing the predicted class, confidence,
            raw heatmap array, and a blended overlay image.
        """
        rgb_image = image.convert("RGB")
        input_tensor = self._inference.transform(rgb_image).unsqueeze(0).to(self._inference.device)

        with torch.no_grad():
            logits = self._model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())
        predicted_class = self._inference.class_names[predicted_index]

        # Deferred: same reason as the GradCAM import above.
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(predicted_index)]
        grayscale_cam = self._cam(input_tensor=input_tensor, targets=targets)[0]

        input_height, input_width = input_tensor.shape[-2], input_tensor.shape[-1]
        resized_image = rgb_image.resize((input_width, input_height))
        normalized_rgb = np.asarray(resized_image, dtype=np.float32) / 255.0

        # Deferred import: pytorch_grad_cam.utils.image imports cv2 at module
        # scope, which requires libxcb.so.1 (an X11/GUI library absent on
        # Azure App Service headless Linux). Importing here instead of at
        # module level defers the cv2 load until the first /predict request,
        # by which point it is irrelevant whether libxcb exists — cv2 is
        # only used by show_cam_on_image for colormap application, which
        # works fine without a display. Python caches the import after the
        # first call so there is no repeated import overhead.
        from pytorch_grad_cam.utils.image import show_cam_on_image
        overlay_array = show_cam_on_image(normalized_rgb, grayscale_cam, use_rgb=True)
        overlay_image = Image.fromarray(overlay_array)

        self._last_overlay = overlay_image

        logger.info(
            "Grad-CAM generated for predicted class '%s' (confidence=%.4f)",
            predicted_class,
            confidence,
        )

        return GradCAMResult(
            predicted_class=predicted_class,
            confidence=confidence,
            heatmap=grayscale_cam,
            overlay=overlay_image,
        )

    def save_overlay(self, path: Union[str, Path]) -> None:
        """Save the most recently generated Grad-CAM overlay to disk.

        Args:
            path: Destination file path (parent directories are created
                automatically).

        Raises:
            RuntimeError: If `generate` has not been called yet.
        """
        if self._last_overlay is None:
            raise RuntimeError("No overlay available. Call generate() before save_overlay().")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_overlay.save(output_path)
        logger.info("Saved Grad-CAM overlay to: %s", output_path)

    def close(self) -> None:
        """Release the forward/backward hooks registered on the model.

        Safe to call multiple times. After calling this, `generate`
        should not be called again on this instance.
        """
        self._cam.activations_and_grads.release()

    def __enter__(self) -> "GradCAMGenerator":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
