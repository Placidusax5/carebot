"""
CareBot AI - choose IMAGE_ESCALATION_THRESHOLD from evidence instead of guessing.

The API escalates when the CNN puts at least THRESHOLD probability on a
malignant / pre-malignant class, even if a benign class won the argmax. That
rule exists because argmax alone misses a large share of melanomas. But the
threshold itself is a trade: lower catches more cancers and raises more false
alarms, and alarm fatigue has a real cost too.

This sweeps the threshold over the SAME held-out test split train_image_model.py
used (same manifest, same seed, same grouped split) and prints what each value
actually buys you.

USAGE
    python scripts/tune_escalation.py
    python scripts/tune_escalation.py --manifest datasets/skin_manifest.csv

Then put the value you pick in backend/.env:
    IMAGE_ESCALATION_THRESHOLD=0.12
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

# Reuse the exact split logic so the test set matches what training evaluated.
from train_image_model import load_manifest, grouped_split, make_ds  # noqa: E402

# Keep in sync with URGENT_CLASSES in backend/app/services/image_service.py
URGENT = {"Melanoma", "Basal Cell Carcinoma", "Actinic Keratosis"}
# The classes where a miss is an actual missed cancer.
MALIGNANT = {"Melanoma", "Basal Cell Carcinoma"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "datasets", "skin_manifest.csv"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "image_model"))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit("No manifest at %s - run prepare_skin_dataset.py first." % args.manifest)

    labels_path = os.path.join(args.model, "labels.json")
    if not os.path.exists(labels_path):
        raise SystemExit("No trained model at %s" % args.model)
    labels = json.load(open(labels_path))

    rows = load_manifest(args.manifest)
    splits = grouped_split(rows)
    test = splits["test"]
    print("test images: %d" % len(test))

    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(os.path.join(args.model, "model.keras"))
    ds = make_ds(tf, test, labels, False, args.batch)

    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)

    urgent_idx = [i for i, l in enumerate(labels) if l in URGENT]
    if not urgent_idx:
        raise SystemExit("None of the URGENT classes are in this model's labels.")
    urgent_score = probs[:, urgent_idx].max(axis=1)

    name_of = {i: l for i, l in enumerate(labels)}
    true_names = np.array([name_of[int(i)] for i in y_true])

    is_malignant = np.isin(true_names, list(MALIGNANT))
    is_mel = true_names == "Melanoma"
    is_benign = ~np.isin(true_names, list(URGENT))

    print("\n  true malignant (Mel+BCC): %d" % is_malignant.sum())
    print("  true melanoma:            %d" % is_mel.sum())
    print("  true non-urgent:          %d" % is_benign.sum())

    print("\n%-10s %12s %12s %14s %12s" % (
        "thresh", "mel recall", "malig rec", "false alarm", "escalated"))
    print("-" * 64)

    baseline = None
    for t in [1.01, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10,
              0.08, 0.06, 0.04, 0.02]:
        if t > 1.0:
            # argmax-only behaviour, for reference: escalates only when an
            # urgent class actually wins.
            fired = np.isin([name_of[int(i)] for i in probs.argmax(axis=1)], list(URGENT))
            tag = "argmax"
        else:
            fired = urgent_score >= t
            tag = "%.2f" % t

        mel_rec = fired[is_mel].mean() if is_mel.sum() else 0.0
        mal_rec = fired[is_malignant].mean() if is_malignant.sum() else 0.0
        fa = fired[is_benign].mean() if is_benign.sum() else 0.0
        esc = fired.mean()

        if tag == "argmax":
            baseline = (mel_rec, mal_rec, fa)
        print("%-10s %11.1f%% %11.1f%% %13.1f%% %11.1f%%" % (
            tag, mel_rec * 100, mal_rec * 100, fa * 100, esc * 100))

    if baseline:
        print("\n'argmax' is the behaviour WITHOUT the escalation rule: melanoma recall "
              "%.1f%%, false alarms %.1f%%." % (baseline[0] * 100, baseline[2] * 100))
    print("\nPick the loosest threshold whose false-alarm rate you can live with,\n"
          "then set IMAGE_ESCALATION_THRESHOLD in backend/.env.")


if __name__ == "__main__":
    main()
