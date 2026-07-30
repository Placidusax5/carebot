"""AI Healthcare Chatbot endpoint (Module 2)."""
import json
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import models
from app.core.security import get_current_user
from app.database import get_db
from app.schemas.schemas import ChatRequest, ChatResponse, PredictResponse
from app.services import chatbot_service

router = APIRouter(tags=["AI Chatbot"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    session_id = payload.session_id or str(uuid.uuid4())

    # Reconstruct prior extracted symptoms from this session's user messages
    prior_messages = (
        db.query(models.ChatHistory)
        .filter(
            models.ChatHistory.user_id == current_user.id,
            models.ChatHistory.session_id == session_id,
            models.ChatHistory.role == "user",
        )
        .order_by(models.ChatHistory.created_at)
        .all()
    )
    prior_symptoms = set()
    for m in prior_messages:
        prior_symptoms |= set(chatbot_service.extract_symptoms_from_text(m.message))

    reply, all_symptoms, followup, prediction = chatbot_service.handle_message(
        payload.message, sorted(prior_symptoms)
    )

    # Persist the exchange
    db.add(models.ChatHistory(user_id=current_user.id, session_id=session_id, role="user", message=payload.message))
    db.add(models.ChatHistory(user_id=current_user.id, session_id=session_id, role="assistant", message=reply))

    prediction_response = None
    if prediction:
        prediction_response = PredictResponse(
            predictions=prediction["predictions"],
            top_prediction=prediction["top_prediction"],
            risk_level=prediction["risk_level"],
            description=prediction["description"],
            precautions=prediction["precautions"],
            home_remedies=prediction.get("home_remedies", []),
            recommended_specialty=prediction.get("specialty", "General Medicine"),
        )
        db.add(models.PredictionHistory(
            user_id=current_user.id,
            symptoms=json.dumps(all_symptoms),
            top_prediction=prediction["top_prediction"],
            confidence=prediction["predictions"][0]["confidence"] if prediction["predictions"] else 0.0,
            all_predictions=json.dumps(prediction["predictions"]),
            risk_level=prediction["risk_level"],
        ))

    db.commit()

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        extracted_symptoms=all_symptoms,
        follow_up_question=followup or None,
        prediction=prediction_response,
    )
