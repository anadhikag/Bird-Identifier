"""Classifier service: application-layer facade over the existing ML classes.

This module is responsible for ONE thing: adapting `BirdInference` and
`GradCAMGenerator` (both complete and unmodified) into a single
API-friendly call. It contains NO machine learning logic of its own —
no model construction, no preprocessing, no Grad-CAM math. It only
composes the two existing classes and reshapes their outputs.

Intended to be constructed exactly once per process (singleton) and
reused across all requests; see backend/app.py's lifespan handler.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from src.classification.gradcam import GradCAMGenerator
    from src.classification.infer import BirdInference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesPrediction:
    """A single ranked species prediction, decoupled from the API schema layer."""

    species: str
    class_index: int
    confidence: float


@dataclass(frozen=True)
class ClassifierPredictionResult:
    """Full result of a classification + Grad-CAM call.

    Attributes:
        predicted_species: Top-1 predicted species common name.
        predicted_class_index: Top-1 predicted species class index.
        confidence: Top-1 confidence score in [0, 1].
        top5: Top-5 predictions, ordered by descending confidence.
        gradcam_overlay_base64: Grad-CAM overlay image, PNG-encoded and
            base64-serialized, ready to embed directly in a JSON
            response.
    """

    predicted_species: str
    predicted_class_index: int
    confidence: float
    top5: list[SpeciesPrediction]
    gradcam_overlay_base64: str


def _encode_image_as_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base64-serialized PNG string.

    Args:
        image: Image to encode.

    Returns:
        Base64-encoded PNG bytes, decoded to a UTF-8 string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class ClassifierService:
    """Application-layer facade over BirdInference and GradCAMGenerator.

    Loads both underlying components once at construction time. Both
    components load the same checkpoint independently (BirdInference
    for top-k classification, GradCAMGenerator for the explanation
    heatmap) since neither exposes the other's functionality — this
    service does not reimplement or duplicate either one.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        """Store configuration for lazy loading the classification and Grad-CAM components.

        Args:
            checkpoint_path: Path to the .pt checkpoint saved by
                train.py (e.g. "models/bird_classifier_best.pt").
            device: Device to run inference on. Defaults to "cuda" if
                available, otherwise "cpu".
        """
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._inference: Optional[BirdInference] = None
        self._gradcam_generator: Optional[GradCAMGenerator] = None
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Initialize BirdInference and GradCAMGenerator thread-safely."""
        if self._inference is not None and self._gradcam_generator is not None:
            return

        with self._lock:
            if self._inference is not None and self._gradcam_generator is not None:
                return

            logger.info("Initializing ClassifierService...")
            # Delayed imports to avoid loading torch and models on startup
            from src.classification.gradcam import GradCAMGenerator as _GradCAMGenerator
            from src.classification.infer import BirdInference as _BirdInference

            self._inference = _BirdInference(checkpoint_path=self._checkpoint_path, device=self._device)
            self._gradcam_generator = _GradCAMGenerator(checkpoint_path=self._checkpoint_path, device=self._device)
            logger.info("ClassifierService ready.")

    def predict(self, image: Image.Image, top_k: int = 5) -> ClassifierPredictionResult:
        """Classify an image and generate its Grad-CAM explanation.

        Args:
            image: Input PIL image containing a cropped bird.
            top_k: Number of top predictions to include in the result.

        Returns:
            ClassifierPredictionResult with the top-1 prediction, the
            full top-k ranking, and a base64-encoded PNG Grad-CAM
            overlay.
        """
        self._ensure_initialized()
        assert self._inference is not None
        assert self._gradcam_generator is not None

        inference_result = self._inference.predict(image, top_k=top_k)
        gradcam_result = self._gradcam_generator.generate(image)

        top_predictions = [
            SpeciesPrediction(
                species=item.class_name,
                class_index=item.class_index,
                confidence=item.confidence,
            )
            for item in inference_result.top5
        ]

        return ClassifierPredictionResult(
            predicted_species=inference_result.top1.class_name,
            predicted_class_index=inference_result.top1.class_index,
            confidence=inference_result.top1.confidence,
            top5=top_predictions,
            gradcam_overlay_base64=_encode_image_as_base64_png(gradcam_result.overlay),
        )

    def close(self) -> None:
        """Release Grad-CAM hooks held by the underlying generator.

        Should be called once at application shutdown.
        """
        with self._lock:
            if self._gradcam_generator is not None:
                self._gradcam_generator.close()

