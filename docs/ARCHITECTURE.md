# CareBot AI — Architecture

## Overview

```
                        CareBot AI
                             |
     +-----------------------+-------------------------+
     |                       |                          |
 Frontend                Backend                   AI Engine
 (React + Vite)          (FastAPI)             (ML + rule-based NLP)
     |                       |                          |
     +---------------- Database (SQLite / PostgreSQL) --+
```

## Request flow

1. The React SPA (`frontend/`) calls the FastAPI backend (`backend/`) over
   REST, authenticating with a JWT stored in `localStorage` and sent as a
   `Bearer` token on every request (see `frontend/src/services/api.js`).
2. FastAPI routers (`backend/app/routers/`) validate the request with
   Pydantic schemas, call into the services layer
   (`backend/app/services/`) for the actual ML/NLP/image logic, then
   persist results via SQLAlchemy models (`backend/app/models.py`).
3. Disease prediction (`app/services/ml_service.py`) prefers a Keras MLP at
   `models/disease_nn/` (trained by `scripts/train_disease_nn.py`) and falls
   back to the scikit-learn `RandomForestClassifier` at
   `models/disease_model.joblib` (`scripts/train_model.py`) when the net is
   absent, TensorFlow is missing, or the saved feature order has drifted from
   `models/symptoms_list.json`. Either model is loaded once per process and
   reused. The net was adopted for calibration rather than accuracy - see the
   comparison table in README.md.
4. The chatbot (`app/services/chatbot_service.py`) is a rule-based NLP
   engine: it extracts symptoms from free text via keyword/synonym
   matching (no external LLM call required), tracks the running set of
   symptoms per chat session, and calls the same prediction service once
   it has enough signal.
5. Image detection (`app/services/image_service.py`) is a placeholder
   heuristic classifier — it implements the full API contract so the rest
   of the app is wired correctly, but is not a trained CNN. See the
   module's docstring for how to swap in a real trained model.

## Why these design choices

- **SQLite by default, PostgreSQL for production** — zero-config local
  dev, a documented path to swap `DATABASE_URL` for real deployments.
- **Rule-based chatbot instead of a hosted LLM** — works fully offline
  with no API key required. `chatbot_service.py` is intentionally
  isolated so it can be swapped for a real LLM call (OpenAI/Anthropic)
  without touching the router or response schema.
- **JWT auth** — standard stateless auth suitable for a SPA + API split
  deployment (frontend on Vercel, backend on Render/Railway per the
  original spec).
- **Services layer** — ML/NLP/image/hospital logic lives outside the
  routers so it can be unit-tested or swapped independently of the HTTP
  layer.

## Known placeholders / upgrade paths

| Module | Current state | To make production-real |
|---|---|---|
| Image detection | Heuristic color-based stand-in **until a CNN is trained** | Run `scripts/prepare_skin_dataset.py` then `scripts/train_image_model.py` (MobileNetV2 transfer learning, 12 merged classes). The API auto-loads `models/image_model/` on restart; no code change needed. |
| Chatbot | Rule-based keyword/synonym NLP | Wire `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` in `.env` and swap `extract_symptoms_from_text`/`generate_reply` for an LLM call |
| Hospital locator | Mock data with real Google Maps search links | Add `GOOGLE_MAPS_API_KEY` and call the Places API in `hospital_service.py` |
| Reports | Plain-text `.txt` download | Swap `_build_report_text` output for a PDF (e.g. via `reportlab` or `weasyprint`) |
