"""Hospital Locator endpoint (Module 6)."""
from fastapi import APIRouter, Depends

from app import models
from app.core.security import get_current_user
from app.schemas.schemas import HospitalOut
from app.services import hospital_service, ml_service

router = APIRouter(tags=["Hospital Locator"])


@router.get("/hospitals", response_model=list[HospitalOut])
def get_hospitals(
    city: str = "your area",
    current_user: models.User = Depends(get_current_user),
):
    return hospital_service.get_hospitals(city)


@router.get("/hospitals/for-disease", response_model=list[HospitalOut])
def get_hospitals_for_disease(
    disease: str,
    city: str = "your area",
    current_user: models.User = Depends(get_current_user),
):
    """Return hospitals matched to the medical specialty a predicted disease
    calls for (e.g. Migraine -> Neurology, Appendicitis -> Emergency)."""
    info = ml_service.get_disease_info(disease)
    specialty = info.get("specialty", "General Medicine")
    severity = info.get("severity", "Medium")
    emergency = severity == "High" or "Emergency" in specialty
    return hospital_service.get_hospitals_for_specialty(
        specialty, city, emergency=emergency
    )
