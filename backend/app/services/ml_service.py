"""
Disease Prediction Module (Module 1)

Loads the trained RandomForest model + label encoder + symptom feature list,
and exposes a predict_disease() function returning the top-3 most probable
diseases with confidence scores, plus a risk-level assessment.
"""
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings

_model = None
_label_encoder = None
_symptoms: List[str] = []
_disease_info: dict = {}


def _load():
    global _model, _label_encoder, _symptoms, _disease_info
    if _model is not None:
        return

    model_path = settings.MODELS_DIR / "disease_model.joblib"
    encoder_path = settings.MODELS_DIR / "label_encoder.joblib"
    symptoms_path = settings.MODELS_DIR / "symptoms_list.json"
    info_path = settings.DATASETS_DIR / "disease_info.json"

    if not model_path.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python scripts/train_model.py` "
            "from the project root before starting the API."
        )

    _model = joblib.load(model_path)
    _label_encoder = joblib.load(encoder_path)
    with open(symptoms_path) as f:
        _symptoms = json.load(f)
    with open(info_path) as f:
        _disease_info = json.load(f)


def get_known_symptoms() -> List[str]:
    _load()
    return _symptoms


def get_disease_info(disease: str) -> dict:
    _load()
    return _disease_info.get(disease, {
        "description": "No additional information available for this condition.",
        "precautions": ["Consult a healthcare professional for guidance."],
        "home_remedies": ["Rest, stay hydrated, and consult a healthcare professional for guidance."],
        "specialty": "General Medicine",
        "severity": "Medium",
    })


def _risk_from_confidence_and_severity(confidence: float, severity: str) -> str:
    """Combine model confidence with the disease's baseline severity to
    produce a single Low/Medium/High risk level shown to the user."""
    severity_rank = {"Low": 0, "Medium": 1, "High": 2}.get(severity, 1)
    if severity_rank == 2 and confidence >= 0.35:
        return "High"
    if severity_rank == 2 or (severity_rank == 1 and confidence >= 0.55):
        return "Medium" if severity_rank != 2 else "High"
    if severity_rank == 1:
        return "Medium"
    return "Low" if confidence < 0.6 else "Medium"


def predict_disease(symptoms: List[str], top_k: int = 3) -> dict:
    """Predict the top-k diseases for a given list of symptom identifiers."""
    _load()

    input_vector = np.zeros((1, len(_symptoms)), dtype=int)
    matched = 0
    for s in symptoms:
        s_norm = s.strip().lower().replace(" ", "_")
        if s_norm in _symptoms:
            idx = _symptoms.index(s_norm)
            input_vector[0, idx] = 1
            matched += 1

    if matched == 0:
        return {
            "predictions": [],
            "top_prediction": "Unknown",
            "risk_level": "Low",
            "description": "No recognized symptoms were provided. Please select from the known symptom list.",
            "precautions": ["Try describing your symptoms differently or select from the suggested list."],
            "home_remedies": [],
            "specialty": "General Medicine",
        }

    input_df = pd.DataFrame(input_vector, columns=_symptoms)
    proba = _model.predict_proba(input_df)[0]
    top_idx = np.argsort(proba)[-top_k:][::-1]

    predictions = [
        {"disease": _label_encoder.classes_[i], "confidence": round(float(proba[i]) * 100, 2)}
        for i in top_idx
        if proba[i] > 0
    ]
    if not predictions:
        predictions = [{"disease": _label_encoder.classes_[top_idx[0]], "confidence": round(float(proba[top_idx[0]]) * 100, 2)}]

    top = predictions[0]
    info = get_disease_info(top["disease"])
    risk = _risk_from_confidence_and_severity(top["confidence"] / 100, info["severity"])

    return {
        "predictions": predictions,
        "top_prediction": top["disease"],
        "risk_level": risk,
        "description": info["description"],
        "precautions": info["precautions"],
        "home_remedies": info.get("home_remedies", []),
        "specialty": info.get("specialty", "General Medicine"),
    }
