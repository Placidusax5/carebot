# CareBot AI — API Reference

Base URL (dev): `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger) or `/redoc`

All endpoints except `/`, `/health`, `/auth/register`, `/auth/login` require
an `Authorization: Bearer <token>` header.

## Auth

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create an account. Body: `{full_name, email, password}`. Returns a JWT + user. |
| POST | `/auth/login` | Log in. Body: `{email, password}`. Returns a JWT + user. |
| GET | `/auth/me` | Return the current authenticated user. |

## Disease Prediction

| Method | Path | Description |
|---|---|---|
| GET | `/symptoms` | List all known symptom identifiers (for populating the picker). |
| POST | `/predict` | Body: `{symptoms: ["fever", "cough", ...]}`. Returns top-3 predictions, risk level, description, precautions. Saves to prediction history. |

## AI Chatbot

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Body: `{message, session_id?}`. Returns the assistant's reply, extracted symptoms so far, an optional follow-up question, and a prediction once enough symptoms are gathered. |

## Image Detection

| Method | Path | Description |
|---|---|---|
| POST | `/image-detect` | Multipart form upload, field `file` (JPEG/PNG/WEBP, max 10MB). Returns a predicted class + confidence + recommendation. Currently backed by a placeholder heuristic — see `docs/ARCHITECTURE.md`. |

## Dashboard / History / Reports

| Method | Path | Description |
|---|---|---|
| GET | `/dashboard` | Aggregated stats: total predictions, disease frequency, risk breakdown, confidence trend, recent predictions. |
| GET | `/history?limit=50` | Full prediction history for the current user. |
| GET | `/reports` | List generated reports. |
| POST | `/reports/generate/{prediction_id}` | Generate a plain-text report from a past prediction. |
| GET | `/reports/{report_id}/download` | Download a report as a `.txt` file. |

## Hospital Locator

| Method | Path | Description |
|---|---|---|
| GET | `/hospitals?city=Springfield` | Mock hospital list scoped to the given city, each with a working Google Maps directions link. |
