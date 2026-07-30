# 🏥 CareBot AI

An intelligent healthcare assistant combining Machine Learning, Deep Learning, and NLP
for preliminary health assessment, education, and guidance.

**CareBot AI does not replace doctors.** It provides preliminary health assessments,
educational information, and guidance to help users make informed decisions about
seeking medical care.

## What's included

| Module | Status |
|---|---|
| Disease prediction (ML) | ✅ Fully working — RandomForest trained on 40 conditions / 134 symptoms |
| AI chatbot | ✅ Fully working — rule-based NLP symptom extraction, no API key required |
| Image-based detection | ⚠️ Placeholder — full API/UI wired up, backed by a heuristic (not a trained CNN). See `docs/ARCHITECTURE.md`. |
| Recommendation engine | ✅ Fully working — descriptions, precautions per condition |
| Risk assessment | ✅ Fully working — Low / Medium / High |
| Hospital locator | ⚠️ Mock data with real Google Maps links — swap in a Places API key for live results |
| User dashboard | ✅ Fully working — charts via Recharts |
| Reports | ✅ Fully working — generate + download as `.txt` |
| Auth (JWT) | ✅ Fully working |

## Project structure

```
CareBot/
├── frontend/        React + Vite + Tailwind SPA
├── backend/          FastAPI app (routers, services, models, auth)
├── datasets/         Generated symptom-disease training data + disease info
├── models/            Trained ML model artifacts (joblib)
├── docs/              Architecture & API docs
├── database/          SQLite file + uploaded images live here at runtime
├── scripts/           Dataset generation + model training scripts
├── docker-compose.yml
└── README.md
```

## Quick start (local dev)

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the ML model (only needed once, or after editing datasets/)
cd ..
python3 scripts/generate_dataset.py   # regenerate training data (optional, already included)
python3 scripts/train_model.py        # trains + saves models/disease_model.joblib

# Run the API
cd backend
cp .env.example .env            # edit SECRET_KEY etc. for production
uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env            # points VITE_API_URL at the backend
npm run dev
```

The app is now live at `http://localhost:5173`.

### 3. Docker (both services)

```bash
docker-compose up --build
```

## Retraining / extending the ML model

- Edit `scripts/generate_dataset.py` to add diseases, symptoms, descriptions,
  or precautions, then re-run it to regenerate `datasets/*.csv` / `*.json`.
- Run `python3 scripts/train_model.py` to retrain and re-save the model.
- The model is a `RandomForestClassifier` (scikit-learn) achieving ~99.5%
  accuracy on the held-out synthetic test split — see
  `models/model_metrics.json` after training.

**Note on the dataset:** the symptom-disease associations are commonly-known,
general medical knowledge (the same structure used by widely published open
symptom-checker datasets), generated programmatically for this project. They
are for educational/demo purposes and are not a substitute for a clinically
validated dataset in a real deployment.

## Deployment (per the original spec)

- **Frontend** → Vercel (`npm run build`, deploy `frontend/dist`)
- **Backend** → Render or Railway (`Dockerfile` included, or run
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`)
- **Database** → SQLite for dev; set `DATABASE_URL` to a PostgreSQL
  connection string for production (SQLAlchemy handles both transparently)

## Tech stack

**Frontend:** React, Vite, Tailwind CSS, React Router, Axios, Recharts, Framer Motion, Lucide Icons
**Backend:** FastAPI, SQLAlchemy, Pydantic, Uvicorn, JWT (python-jose), Passlib/bcrypt
**ML:** scikit-learn (RandomForest), Pandas, NumPy, Joblib
**Database:** SQLite (dev) / PostgreSQL (prod)

## Disclaimer

CareBot AI is an educational/demo project. All predictions, risk levels, and
recommendations are AI-generated and are **not** a medical diagnosis. Always
consult a licensed healthcare professional for medical concerns. In an
emergency, contact your local emergency services immediately.
