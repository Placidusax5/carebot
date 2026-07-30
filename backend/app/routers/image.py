"""Image-Based Disease Detection endpoint (Module 3)."""
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app import models
from app.core.config import settings
from app.core.security import get_current_user
from app.database import get_db
from app.schemas.schemas import ImageDetectResponse
from app.services import image_service

router = APIRouter(tags=["Image Detection"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


@router.post("/image-detect", response_model=ImageDetectResponse)
async def image_detect(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a JPEG, PNG, or WEBP image.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB).")

    try:
        result = image_service.classify_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process this image file.")

    # Save the uploaded image + result
    ext = (file.filename or "upload.jpg").split(".")[-1]
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    with open(settings.UPLOADS_DIR / saved_name, "wb") as f:
        f.write(image_bytes)

    db.add(models.UploadedImage(
        user_id=current_user.id,
        filename=saved_name,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
    ))
    db.commit()

    is_placeholder = result.get("is_placeholder_model", True)
    note = (
        "Image classification is running on a transparent feature-based heuristic in this "
        "build — not a medically validated model. It highlights visual features (redness, "
        "texture, pigmentation) but cannot diagnose. Always see a dermatologist to confirm."
        if is_placeholder
        else "Prediction from a trained CNN model. Still not a substitute for a professional "
        "diagnosis — see a dermatologist to confirm."
    )

    return ImageDetectResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        predictions=result.get("predictions", []),
        recommendation=result["recommendation"],
        quality_warning=result.get("quality_warning"),
        is_placeholder_model=is_placeholder,
        note=note,
    )
