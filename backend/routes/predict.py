"""Prediction endpoint: species classification with a Grad-CAM explanation."""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.dependencies import get_classifier_service, get_knowledge_service
from backend.schemas import PredictionItem, PredictResponse
from backend.services.classifier_service import ClassifierService
from backend.services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="Classify a bird image")
async def predict(
    file: UploadFile = File(..., description="Image file containing a cropped bird."),
    classifier_service: ClassifierService = Depends(get_classifier_service),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> PredictResponse:
    """Classify an uploaded bird image and return its species details and explanation.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.).
        classifier_service: Injected singleton classifier service.
        knowledge_service: Injected singleton knowledge guide service.

    Returns:
        PredictResponse containing top-1 species_id/common_name, confidence,
        top-5 rankings, and base64-encoded Grad-CAM image.

    Raises:
        HTTPException: 400 if the uploaded file is not a valid image.
    """
    raw_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        ) from error

    logger.info("Received prediction request: filename=%s, size=%d bytes", file.filename, len(raw_bytes))

    result = classifier_service.predict(image)

    # Map the predicted top class index to its canonical species ID
    top_species_id = knowledge_service.get_species_id_by_index(result.predicted_class_index)
    if not top_species_id:
        top_species_id = f"{result.predicted_class_index + 1:03d}.Unknown"
        
    top_common_name = result.predicted_species
    top_data = knowledge_service.get_species(top_species_id)
    if top_data:
        top_common_name = top_data["common_name"]

    # Map the top-5 alternative lists
    predictions_list = []
    for item in result.top5:
        item_species_id = knowledge_service.get_species_id_by_index(item.class_index)
        if not item_species_id:
            item_species_id = f"{item.class_index + 1:03d}.Unknown"
            
        item_common_name = item.species
        item_data = knowledge_service.get_species(item_species_id)
        if item_data:
            item_common_name = item_data["common_name"]
            
        predictions_list.append(
            PredictionItem(
                species_id=item_species_id,
                common_name=item_common_name,
                confidence=item.confidence,
            )
        )

    return PredictResponse(
        species_id=top_species_id,
        common_name=top_common_name,
        confidence=result.confidence,
        top_predictions=predictions_list,
        gradcam_image=result.gradcam_overlay_base64,
    )

