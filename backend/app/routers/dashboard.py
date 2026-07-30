"""User dashboard analytics endpoint (Module 5)."""
import json
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import models
from app.core.security import get_current_user
from app.database import get_db
from app.schemas.schemas import DashboardResponse, PredictionHistoryOut

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    records = (
        db.query(models.PredictionHistory)
        .filter(models.PredictionHistory.user_id == current_user.id)
        .order_by(models.PredictionHistory.created_at.desc())
        .all()
    )

    disease_frequency = Counter(r.top_prediction for r in records)
    risk_breakdown = Counter(r.risk_level for r in records)
    confidence_trend = [
        {"date": r.created_at.isoformat(), "confidence": r.confidence, "disease": r.top_prediction}
        for r in reversed(records[:30])
    ]

    recent = []
    for r in records[:10]:
        recent.append(PredictionHistoryOut(
            id=r.id,
            symptoms=json.loads(r.symptoms),
            top_prediction=r.top_prediction,
            confidence=r.confidence,
            all_predictions=json.loads(r.all_predictions),
            risk_level=r.risk_level,
            created_at=r.created_at,
        ))

    return DashboardResponse(
        total_predictions=len(records),
        disease_frequency=dict(disease_frequency),
        risk_breakdown=dict(risk_breakdown),
        confidence_trend=confidence_trend,
        recent_predictions=recent,
    )
