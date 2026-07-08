"""Grad-CAM visualization for the trained bird species classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import torch
from PIL import Image

from src.classification.infer import BirdInference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradCAMResult:
    predicted_class: str
    confidence: float
    heatmap: np.ndarray
    overlay: Image.Image


class GradCAMGenerator:
    """Generates Grad-CAM explanations for the trained bird classifier."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device_arg = device
        self._inference: Optional[BirdInference] = None
        self._model: Optional[Any] = None
        self._cam: Optional[Any] = None
        self._last_overlay: Optional[Image.Image] = None

    def _ensure_model_loaded((self) -> None:
        """Lazy load the inference model safely."""
        if self._inference is None:
            self._inference = BirdInference(checkpoint_path=self._checkpoint_path, device=self._device_arg)
            self._model = self._inference.model
            self._model.eval()

    def _get_cam_engine(self) -> Any:
        """Lazy load pytorch_grad_cam ONLY when generating a heatmap."""
        self._ensure_model_loaded()
        if self._cam is None:
            # Deferred import prevents cv2 / libxcb loading during app startup
            from pytorch_grad_cam import GradCAM
            target_layer = self._model.get_feature_extractor()[-1]
            self._cam = GradCAM(model=self._model, target_layers=[target_layer])
            logger.info("GradCAM engine initialized.")
        return self._cam

    def generate(self, image: Image.Image) -> GradCAMResult:
        """Run Grad-CAM on a single image."""
        self._ensure_model_loaded()
        cam_engine = self._get_cam_engine()

        rgb_image = image.convert("RGB")
        input_tensor = self._inference.transform(rgb_image).unsqueeze(0).to(self._inference.device)

        with torch.no_grad():
            logits = self._model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())
        predicted_class = self._inference.class_names[predicted_index]

        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(predicted_index)]
        grayscale_cam = cam_engine(input_tensor=input_tensor, targets=targets)[0]

        input_height, input_width = input_tensor.shape[-2], input_tensor.shape[-1]
        resized_image = rgb_image.resize((input_width, input_height))
        normalized_rgb = np.asarray(resized_image, dtype=np.float32) / 255.0

        from pytorch_grad_cam.utils.image import show_cam_on_image
        overlay_array = show_cam_on_image(normalized_rgb, grayscale_cam, use_rgb=True)
        overlay_image = Image.fromarray(overlay_array)

        self._last_overlay = overlay_image

        return GradCAMResult(
            predicted_class=predicted_class,
            confidence=confidence,
            heatmap=grayscale_cam,
            overlay=overlay_image,
        )

    def save_overlay(self, path: Union[str, Path]) -> None:
        if self._last_overlay is None:
            raise RuntimeError("No overlay available. Call generate() before save_overlay().")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_overlay.save(output_path)

    def close(self) -> None:
        if self._cam is not None:
            self._cam.activations_and_grads.release()

    def __enter__(self) -> "GradCAMGenerator":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()