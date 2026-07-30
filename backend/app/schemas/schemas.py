"""Pydantic request/response schemas for all API endpoints."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth / Users
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------------------------------------------------------------------------
# Disease Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., min_length=1, description="List of symptom identifiers")


class DiseasePrediction(BaseModel):
    disease: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: List[DiseasePrediction]
    top_prediction: str
    risk_level: str
    description: str
    precautions: List[str]
    home_remedies: List[str] = []
    recommended_specialty: str = "General Medicine"
    disclaimer: str = (
        "This is an AI-generated preliminary assessment, not a medical diagnosis. "
        "Please consult a licensed healthcare professional for accurate diagnosis and treatment."
    )


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    extracted_symptoms: List[str] = []
    follow_up_question: Optional[str] = None
    prediction: Optional[PredictResponse] = None


# ---------------------------------------------------------------------------
# Image Detection
# ---------------------------------------------------------------------------
class ImagePrediction(BaseModel):
    label: str
    confidence: float


class ImageDetectResponse(BaseModel):
    predicted_class: str
    confidence: float
    predictions: List[ImagePrediction] = []
    recommendation: str
    quality_warning: Optional[str] = None
    is_placeholder_model: bool = True
    note: str = (
        "Image classification is running on a transparent feature-based heuristic in this "
        "build — not a medically validated model. Drop a trained CNN into models/image_model/ "
        "(see scripts/train_image_model.py) for real predictions."
    )


# ---------------------------------------------------------------------------
# History / Dashboard / Reports
# ---------------------------------------------------------------------------
class PredictionHistoryOut(BaseModel):
    id: int
    symptoms: List[str]
    top_prediction: str
    confidence: float
    all_predictions: List[DiseasePrediction]
    risk_level: str
    created_at: datetime

    class Config:
        from_attributes = True


class DashboardResponse(BaseModel):
    total_predictions: int
    disease_frequency: dict
    risk_breakdown: dict
    confidence_trend: List[dict]
    recent_predictions: List[PredictionHistoryOut]


class ReportOut(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class HospitalOut(BaseModel):
    name: str
    address: str
    phone: str
    distance_km: float
    maps_link: str
    department: Optional[str] = None
