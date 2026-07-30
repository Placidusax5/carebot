"""SQLAlchemy ORM models: Users, Prediction History, Chat History, Reports, Uploaded Images."""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utcnow)

    predictions = relationship("PredictionHistory", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")
    images = relationship("UploadedImage", back_populates="user", cascade="all, delete-orphan")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symptoms = Column(Text, nullable=False)  # JSON-encoded list of symptom strings
    top_prediction = Column(String(120), nullable=False)
    confidence = Column(Float, nullable=False)
    all_predictions = Column(Text, nullable=False)  # JSON-encoded top-3 list
    risk_level = Column(String(20), nullable=False)  # Low / Medium / High
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="predictions")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(16), nullable=False)  # 'user' or 'assistant'
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="chats")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("prediction_history.id"), nullable=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)  # plain-text report body
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="reports")


class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    predicted_class = Column(String(120), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="images")
