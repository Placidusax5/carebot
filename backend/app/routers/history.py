"""Prediction history endpoint (Module 5)."""
import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import models
from app.core.security import get_current_user
from app.database import get_db
from app.schemas.schemas import PredictionHistoryOut

router = APIRouter(tags=["History"])


@router.get("/history", response_model=list[PredictionHistoryOut])
def get_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    limit: int = 50,
):
    records = (
        db.query(models.PredictionHistory)
        .filter(models.PredictionHistory.user_id == current_user.id)
        .order_by(models.PredictionHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    out = []
    for r in records:
        out.append(PredictionHistoryOut(
            id=r.id,
            symptoms=json.loads(r.symptoms),
            top_prediction=r.top_prediction,
            confidence=r.confidence,
            all_predictions=json.loads(r.all_predictions),
            risk_level=r.risk_level,
            created_at=r.created_at,
        ))
    return out
