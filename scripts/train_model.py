"""
CareBot AI - Disease Prediction Model Training

Trains a RandomForestClassifier on the symptom-disease dataset and saves:
  - models/disease_model.joblib      (trained classifier)
  - models/label_encoder.joblib      (disease label encoder)
  - models/symptoms_list.json        (ordered feature/symptom list, copied)
  - models/model_metrics.json        (accuracy + report, for transparency)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.join(os.path.dirname(__file__), "..")
DATASETS = os.path.join(BASE, "datasets")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(os.path.join(DATASETS, "symptom_disease_dataset.csv"))
with open(os.path.join(DATASETS, "symptoms_list.json")) as f:
    symptoms = json.load(f)

X = df[symptoms]
y_raw = df["prognosis"]

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training RandomForest on {len(X_train)} samples, {len(symptoms)} features, {len(le.classes_)} classes...")
clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",   # keep every condition well-represented
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# Evaluate with both top-1 and top-3 accuracy. Top-3 is the metric that
# matters here because the app surfaces the three most probable conditions.
proba_test = clf.predict_proba(X_test)
y_pred = proba_test.argmax(axis=1)
acc = accuracy_score(y_test, y_pred)
top3_idx = np.argsort(proba_test, axis=1)[:, -3:]
top3_acc = float(np.mean([y_test[i] in top3_idx[i] for i in range(len(y_test))]))
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)

# 5-fold cross-validation for an honest, less optimistic accuracy estimate
cv_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

print(f"Test accuracy (top-1): {acc:.4f}")
print(f"Test accuracy (top-3): {top3_acc:.4f}")
print(f"5-fold CV accuracy:     {cv_mean:.4f} +/- {cv_std:.4f}")

joblib.dump(clf, os.path.join(MODELS, "disease_model.joblib"), compress=3)
joblib.dump(le, os.path.join(MODELS, "label_encoder.joblib"), compress=3)
with open(os.path.join(MODELS, "symptoms_list.json"), "w") as f:
    json.dump(symptoms, f, indent=2)
with open(os.path.join(MODELS, "model_metrics.json"), "w") as f:
    json.dump({
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "n_classes": len(le.classes_),
        "n_features": len(symptoms),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }, f, indent=2)

print("Saved model artifacts to models/")

# quick sanity check: top-3 prediction demo
sample = X_test.iloc[[0]]
proba = clf.predict_proba(sample)[0]
demo_idx = proba.argsort()[-3:][::-1]
print("\nSample prediction (top-3):")
for i in demo_idx:
    print(f"  {le.classes_[i]}: {proba[i]*100:.1f}%")
print(f"  Actual: {le.classes_[y_test[0]]}")
