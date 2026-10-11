# 🏥 CareBot AI

An intelligent healthcare assistant combining Machine Learning, Deep Learning, and NLP
for preliminary health assessment, education, and guidance.

**CareBot AI does not replace doctors.** It provides preliminary health assessments,
educational information, and guidance to help users make informed decisions about
seeking medical care.

## What's included

| Module | Status |
|---|---|
| Disease prediction (ML + DL) | ✅ Fully working — MLP (Keras) over 40 conditions / 134 symptoms, with the original RandomForest as automatic fallback |
| AI chatbot | ✅ Fully working — rule-based NLP symptom extraction, no API key required |
| Image-based detection | ⚠️ Heuristic by default — a real CNN (MobileNetV2, 12 classes) can be trained via `scripts/`; the API picks it up automatically. See "Training the image CNN" below. |
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

## The disease model: neural net + RandomForest fallback

`models/disease_nn/` holds a small MLP trained by `scripts/train_disease_nn.py`.
The API loads it automatically and falls back to `models/disease_model.joblib`
(RandomForest) if it is absent, if TensorFlow is not installed, or if the saved
feature order no longer matches `models/symptoms_list.json`.

```bash
python scripts/train_disease_nn.py        # seconds on CPU
```

Measured on a deduplicated, stratified test split (both models, same rows):

| | Neural net | RandomForest |
|---|---|---|
| top-1 | **0.9940** | 0.9910 |
| top-3 | 1.0000 | 1.0000 |
| macro F1 | **0.9934** | 0.9899 |
| calibration error (ECE) | **0.049** | 0.149 |
| claims exactly 100% | **0%** | 10.4% |
| 3rd prediction is exactly 0 | **0%** | 19.4% |

Architecture `256 -> 128`, dropout 0.4, label smoothing 0.05, chosen by
`--tune` (grid search on the validation split; the test split is never used for
selection).

The accuracy gain is marginal and was never the point - the dataset is
generated, so only 5 of 2229 unique symptom patterns are ambiguous and the
ceiling is ~99.8%. The win is calibration. RandomForest probabilities are
tree-vote fractions, so it claimed total certainty on a tenth of inputs and
assigned the third-ranked disease a hard zero 19% of the time, which made the
API silently return fewer than the three predictions the UI promises.

**A trap worth knowing about if you re-tune:** do not select on ECE alone.
ECE measures |confidence - accuracy|, so on a dataset this easy a model that is
right 99.4% of the time and claims 100% on every input scores a near-perfect
ECE of 0.003 - while reintroducing exactly the overconfidence this model exists
to avoid. A first sweep selected purely on ECE picked `label_smoothing=0.0` and
came back claiming total certainty on 88.7% of inputs with a hard-zero third
prediction 15.8% of the time. `sweep()` now caps how often a candidate may
claim certainty (5%) before it is eligible. In the full grid, every
`label_smoothing=0` config saturated at 60-90%; every config with smoothing
above zero sat at 0%.

**On deduplication:** the generator samples from each disease's fixed symptom
list, so the 6000 rows collapse to 2229 distinct patterns. `train_model.py`
splits the raw rows, which puts 75% of its test set verbatim into training and
inflates its reported score. `train_disease_nn.py` deduplicates first, so its
numbers are directly comparable and honest.

## Training the image CNN

By default image detection runs a transparent feature heuristic. To replace it
with a real convolutional network:

```bash
# 1. Install TensorFlow into the backend venv (~600MB)
cd backend && pip install -r requirements-ml.txt && cd ..

# 2. Download both datasets from Kaggle (free account required), unzip them:
#    https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
#    https://www.kaggle.com/datasets/shubhamgoel27/dermnet

# 3. Build the training manifest
python scripts/prepare_skin_dataset.py --ham path/to/ham10000 --dermnet path/to/dermnet

# 4. Train (several hours on CPU; see notebooks/train_on_colab.ipynb for a
#    ~25-minute GPU run on Colab instead)
python scripts/train_image_model.py

# 5. Restart the API - it auto-loads models/image_model/
```

This produces a **12-class** model merging two label spaces:

- **Pigmented / neoplastic** (HAM10000, dermoscopy): Melanoma, Melanocytic
  Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis,
  Dermatofibroma, Vascular Lesion
- **Common inflammatory** (DermNet, clinical photos): Acne, Eczema, Psoriasis,
  Fungal Skin Infection

### Training on Colab instead

TensorFlow has no GPU support on native Windows, so a local run is CPU-only and
takes hours. `notebooks/train_on_colab.ipynb` does the same job on a free T4 in
roughly 25 minutes: upload it to <https://colab.research.google.com>, set
`Runtime > Change runtime type > T4 GPU`, and run the cells top to bottom. It
pulls both datasets straight from Kaggle, so nothing large touches your machine.
You end up with an `image_model.zip` to unpack into `models/image_model/`.

### Reading the results

Check `models/image_model/model_metrics.json` and judge the model on
**macro recall**, not accuracy. The dataset is heavily imbalanced (Melanocytic
Nevus alone is roughly two-thirds of HAM10000), so a model that always answers
with the majority class scores ~67% accuracy while being useless. Training
applies inverse-frequency class weights, and the split is grouped by
`lesion_id` because HAM10000 contains several photographs of the same lesion -
splitting randomly leaks them across train/validation and inflates the score.

### Note on the "healthy skin" class

Neither public dataset contains healthy-skin images, so by default the model
has no way to say "this looks normal" and will assign every photo some
condition. The API compensates with a confidence-abstention threshold, but the
better fix is to supply your own images:
`prepare_skin_dataset.py --normal path/to/healthy_skin_photos`.

### Safety

A trained model here screens for skin cancers, which is a materially
higher-stakes output than the heuristic's "redness detected". It is trained on
public research datasets and is **not** clinically validated. A reassuring
result is not evidence a lesion is harmless.

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
