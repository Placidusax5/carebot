"""Application configuration, loaded from environment variables / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
PROJECT_ROOT = BASE_DIR.parent  # CareBot/

load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = "CareBot AI"
    API_V1_PREFIX: str = ""

    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'database' / 'carebot.db'}"
    )

    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
    UPLOADS_DIR: Path = PROJECT_ROOT / "database" / "uploaded_images"

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # If the image CNN puts at least this much probability on a malignant or
    # pre-malignant class, the response escalates even when the top prediction
    # is something benign. Tune it with scripts/tune_escalation.py - the
    # default is a starting point, not an evidence-based choice.
    IMAGE_ESCALATION_THRESHOLD: float = float(
        os.getenv("IMAGE_ESCALATION_THRESHOLD", "0.15")
    )


settings = Settings()
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "database").mkdir(parents=True, exist_ok=True)
