"""
Image-Based Disease Detection Module (Module 3).

This module runs a transparent, deterministic computer-vision analysis over
the uploaded photo — extracting dermatology-relevant features (redness /
erythema, texture / scaliness, pigmentation, spottiness, colour heterogeneity)
and scoring them across the candidate skin conditions, returning the top-3 with
softmax confidences plus image-quality warnings so it doesn't give confident
answers on unusable photos.

It is still a heuristic, NOT a medically validated classifier. But it is
"CNN-ready": if a trained Keras model is placed in `models/image_model/`
(with a `labels.json`), `classify_image()` will automatically use it instead
of the heuristic. Train one with `scripts/train_image_model.py`.
"""
import io
import json
from typing import List, Optional

import numpy as np
from PIL import Image

from app.core.config import settings

# Candidate classes for the heuristic path. A trained CNN can define its own
# labels via models/image_model/labels.json.
CLASSES = [
    "Eczema",
    "Psoriasis",
    "Fungal Skin Infection",
    "Acne",
    "Likely Normal / Healthy Skin",
]

RECOMMENDATIONS = {
    "Eczema": "Consider moisturizing regularly and avoiding known irritants. See a dermatologist if it persists or worsens.",
    "Psoriasis": "Keep the area moisturized and avoid known triggers. A dermatologist can confirm and recommend treatment.",
    "Fungal Skin Infection": "Keep the area clean and dry. Over-the-counter antifungal treatment may help; see a doctor if it spreads.",
    "Acne": "Gentle cleansing and non-comedogenic products may help. See a dermatologist for persistent or severe acne.",
    "Likely Normal / Healthy Skin": "No obvious concerning features detected. Continue routine skin care and monitor for changes.",
}

_WORK_SIZE = 160  # analysis resolution

# --- optional trained-CNN hook -------------------------------------------------
_cnn = None
_cnn_labels: Optional[List[str]] = None
_cnn_checked = False


def _try_load_cnn():
    """Load a trained Keras model from models/image_model/ if present and
    TensorFlow is installed. Returns the model, or None to use the heuristic."""
    global _cnn, _cnn_labels, _cnn_checked
    if _cnn_checked:
        return _cnn
    _cnn_checked = True

    model_dir = settings.MODELS_DIR / "image_model"
    model_file = model_dir / "model.keras"
    if not model_file.exists():
        return None
    try:
        import tensorflow as tf  # noqa: F401  (only needed if a model exists)

        _cnn = tf.keras.models.load_model(str(model_file))
        labels_path = model_dir / "labels.json"
        _cnn_labels = json.loads(labels_path.read_text()) if labels_path.exists() else CLASSES
    except Exception:
        _cnn = None  # any failure -> fall back to the heuristic silently
    return _cnn


# --- feature extraction --------------------------------------------------------
def _extract_features(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((_WORK_SIZE, _WORK_SIZE))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    mean_v = float(v.mean())
    mean_s = float(s.mean())

    # Erythema / redness: how much red dominates green+blue (inflammation proxy).
    redness = float(np.clip((r - (g + b) / 2).mean() * 2.0, 0.0, 1.0))

    darkness = float(1.0 - mean_v)
    heterogeneity = float(np.clip(gray.std() * 2.0, 0.0, 1.0))

    # Texture / scaliness via gradient magnitude.
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    edge_energy = float(np.clip(grad.mean() * 4.0, 0.0, 1.0))
    # Spottiness: fraction of clearly-stronger-gradient pixels (small bumps / spots).
    strong = grad > (grad.mean() + 1.5 * grad.std())
    spottiness = float(np.clip(strong.mean() * 2.0, 0.0, 1.0))

    # Brown/orange, moderately-saturated, darker regions (fungal patch proxy).
    hue = h * 255.0  # PIL hue is 0-255
    brownish = float(np.clip(((hue > 10) & (hue < 40) & (s > 0.25) & (v < 0.7)).mean() * 2.5, 0.0, 1.0))

    # Skin-likeliness: skin typically has R >= G >= B and mid brightness.
    skin_mask = (r >= g) & (g >= b) & (r > 0.2) & (v > 0.15) & (v < 0.98)
    skin_fraction = float(skin_mask.mean())

    glare = float((v > 0.96).mean())

    return {
        "mean_v": mean_v,
        "mean_s": mean_s,
        "redness": redness,
        "darkness": darkness,
        "heterogeneity": heterogeneity,
        "edge_energy": edge_energy,
        "spottiness": spottiness,
        "brownish": brownish,
        "skin_fraction": skin_fraction,
        "glare": glare,
    }


def _heuristic_scores(f: dict) -> np.ndarray:
    """Transparent, hand-tuned affinity of each class to the extracted
    features. Order matches CLASSES."""
    redness, tex, spot = f["redness"], f["edge_energy"], f["spottiness"]
    dark, brown, het, v = f["darkness"], f["brownish"], f["heterogeneity"], f["mean_v"]

    eczema = 1.7 * redness + 0.3 * tex - 0.3 * spot           # red & inflamed, not very spotty
    psoriasis = 1.0 * redness + 1.8 * tex + 0.6 * het         # red, scaly, textured plaques
    fungal = 2.0 * brown + 0.9 * dark + 0.3 * tex             # brown/darker ring-like patches
    acne = 0.7 * redness + 1.8 * spot                         # discrete spots / bumps
    normal = (1.2 * (1 - redness) + 0.9 * (1 - tex) + 0.5 * (1 - het)
              - 1.2 * redness - 1.0 * brown - 0.6 * spot)   # inflammation/spots push away from "healthy"
    return np.array([eczema, psoriasis, fungal, acne, normal], dtype=np.float64)


def _softmax(x: np.ndarray, temp: float = 0.6) -> np.ndarray:
    z = (x - x.max()) / temp
    e = np.exp(z)
    return e / e.sum()


def _quality_warning(f: dict) -> Optional[str]:
    if f["skin_fraction"] < 0.25:
        return ("This doesn't look like a clear close-up photo of skin, so the assessment "
                "may be unreliable. Try a well-lit, centered close-up of the affected area.")
    if f["mean_v"] < 0.12:
        return "The image looks very dark. Retake it in better lighting for a more reliable result."
    if f["glare"] > 0.35 or f["mean_v"] > 0.96:
        return "The image looks overexposed. Reduce glare/brightness and retake for a more reliable result."
    if f["edge_energy"] < 0.03:
        return "The image looks blurry or low-detail. A sharper, closer photo gives a more reliable result."
    return None


def _top3(probs: np.ndarray, labels: List[str]) -> List[dict]:
    order = np.argsort(probs)[::-1][:3]
    return [{"label": labels[i], "confidence": round(float(probs[i]) * 100, 2)} for i in order]


def _classify_heuristic(image_bytes: bytes) -> dict:
    f = _extract_features(image_bytes)
    probs = _softmax(_heuristic_scores(f))
    warning = _quality_warning(f)

    # On a poor-quality image, pull confidence toward uniform so we don't
    # report a confident answer we can't stand behind.
    if warning:
        uniform = np.ones_like(probs) / len(probs)
        probs = 0.5 * probs + 0.5 * uniform
        probs /= probs.sum()

    predictions = _top3(probs, CLASSES)
    top = predictions[0]
    return {
        "predicted_class": top["label"],
        "confidence": top["confidence"],
        "predictions": predictions,
        "recommendation": RECOMMENDATIONS.get(top["label"], "See a dermatologist for evaluation."),
        "quality_warning": warning,
        "is_placeholder_model": True,
    }


def _classify_with_cnn(cnn, image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    # Raw 0-255 pixels: the trained model (scripts/train_image_model.py) bakes
    # MobileNetV2 preprocessing into its own graph, so we must NOT rescale here.
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    preds = np.asarray(cnn.predict(arr, verbose=0))[0]
    labels = _cnn_labels or CLASSES
    predictions = _top3(preds, labels)
    top = predictions[0]
    return {
        "predicted_class": top["label"],
        "confidence": top["confidence"],
        "predictions": predictions,
        "recommendation": RECOMMENDATIONS.get(top["label"], "See a dermatologist for professional evaluation."),
        "quality_warning": None,
        "is_placeholder_model": False,
    }


def classify_image(image_bytes: bytes) -> dict:
    """Classify a skin image. Uses a trained CNN if one is installed at
    models/image_model/, otherwise the transparent feature heuristic."""
    cnn = _try_load_cnn()
    if cnn is not None:
        try:
            return _classify_with_cnn(cnn, image_bytes)
        except Exception:
            pass  # fall back to heuristic if inference fails
    return _classify_heuristic(image_bytes)
