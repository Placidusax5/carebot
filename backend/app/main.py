"""CareBot AI - FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database import Base, engine
from app import models  # noqa: F401 - ensures models are registered before create_all
from app.routers import auth, chat, dashboard, history, hospitals, image, predict, reports

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "CareBot AI: an intelligent healthcare assistant combining Machine Learning, "
        "Deep Learning, and NLP for preliminary health assessment, education, and guidance. "
        "This tool does not replace professional medical advice."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(image.router)
app.include_router(dashboard.router)
app.include_router(history.router)
app.include_router(reports.router)
app.include_router(hospitals.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "CareBot AI API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
