"""Disease prediction endpoint (Module 1)."""
import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import models
from app.core.security import get_current_user
from app.database import get_db
from app.schemas.schemas import PredictRequest, PredictResponse
from app.services import ml_service

router = APIRouter(tags=["Disease Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    result = ml_service.predict_disease(payload.symptoms)

    # Save to prediction history for the dashboard/history/reports modules
    record = models.PredictionHistory(
        user_id=current_user.id,
        symptoms=json.dumps(payload.symptoms),
        top_prediction=result["top_prediction"],
        confidence=result["predictions"][0]["confidence"] if result["predictions"] else 0.0,
        all_predictions=json.dumps(result["predictions"]),
        risk_level=result["risk_level"],
    )
    db.add(record)
    db.commit()

    return PredictResponse(
        predictions=result["predictions"],
        top_prediction=result["top_prediction"],
        risk_level=result["risk_level"],
        description=result["description"],
        precautions=result["precautions"],
        home_remedies=result.get("home_remedies", []),
        recommended_specialty=result.get("specialty", "General Medicine"),
    )


@router.get("/symptoms")
def list_symptoms():
    """Helper endpoint for the frontend to populate the symptom picker."""
    return {"symptoms": ml_service.get_known_symptoms()}
