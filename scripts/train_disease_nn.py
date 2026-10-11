"""
CareBot AI - Disease Prediction, neural-network version (Module 1).

Trains a small MLP over the 134 binary symptom features and saves:
    models/disease_nn/model.keras          (Keras 3 native format)
    models/disease_nn/labels.json          (ordered class names)
    models/disease_nn/symptoms_list.json   (feature order - MUST match inference)
    models/disease_nn/model_metrics.json   (NN vs RandomForest, same split)

The API loads this automatically if present and falls back to the RandomForest
otherwise (see backend/app/services/ml_service.py).

WHY A NET AT ALL
    Not for accuracy. The RandomForest already gets ~98.9% top-1 / 100% top-3
    on a clean split, and only 5 of 2229 unique symptom patterns are genuinely
    ambiguous, so there is roughly one point of headroom in total.

    It is for CALIBRATION. RandomForest probabilities are tree-vote fractions:
    it says exactly 100% on a quarter of all inputs, and the third-ranked class
    gets a hard zero 38% of the time - which makes the API drop it and render
    fewer than the three predictions the UI promises. A softmax head with label
    smoothing produces a smooth distribution over all 40 classes instead.

WHY DEDUPLICATE
    The generator samples from each disease's fixed symptom list, so 6000 rows
    collapse to 2229 distinct patterns. Splitting the raw rows puts identical
    patterns on both sides - 75% of test rows appear verbatim in training - and
    inflates the score. Every number here is computed on deduplicated data.

USAGE
    python scripts/train_disease_nn.py
    python scripts/train_disease_nn.py --epochs 300 --no-compare
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SEED = 42


def expected_calibration_error(probs, y_true, n_bins=10):
    """Mean gap between stated confidence and observed accuracy.

    0 is perfect. A model that says 90% should be right 90% of the time.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(err)


def summarise(probs, y_true, labels, name):
    top1 = float((probs.argmax(axis=1) == y_true).mean())
    order = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y_true[i] in order[i] for i in range(len(y_true))]))

    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(y_true, probs.argmax(axis=1), average="macro",
                              labels=list(range(len(labels))), zero_division=0))

    conf = probs.max(axis=1)
    third = np.sort(probs, axis=1)[:, -3]
    return {
        "name": name,
        "top1": top1,
        "top3": top3,
        "macro_f1": macro_f1,
        "ece": expected_calibration_error(probs, y_true),
        "pct_saying_100": float((conf >= 0.999).mean()),
        "median_confidence": float(np.median(conf)),
        "pct_third_is_zero": float((third <= 1e-9).mean()),
    }


def print_row(s):
    print("  %-16s %7.4f %7.4f %8.4f %7.4f %9.1f%% %9.1f%% %10.1f%%" % (
        s["name"], s["top1"], s["top3"], s["macro_f1"], s["ece"],
        s["pct_saying_100"] * 100, s["median_confidence"] * 100,
        s["pct_third_is_zero"] * 100))


def build_and_fit(tf, cfg, X_train, y_train_oh, X_val, y_val_oh, n_features,
                  n_classes, epochs, batch, verbose=0):
    """Build + train one configuration. Returns the fitted model."""
    from tensorflow.keras import layers

    seq = [tf.keras.Input(shape=(n_features,))]
    for i, width in enumerate(cfg["hidden"]):
        seq.append(layers.Dense(width, activation="relu"))
        if i == 0:
            seq.append(layers.BatchNormalization())
        seq.append(layers.Dropout(cfg["dropout"]))
    seq.append(layers.Dense(n_classes, activation="softmax"))
    model = tf.keras.Sequential(seq)

    # Label smoothing is the main calibration lever: it stops the net driving
    # the winning logit to 1.0, which is exactly the overconfidence being
    # fixed. It needs one-hot targets, hence CategoricalCrossentropy.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg["label_smoothing"]),
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train_oh, validation_data=(X_val, y_val_oh),
        epochs=epochs, batch_size=batch, verbose=verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=10, min_lr=1e-6),
        ],
    )
    return model


def sweep(tf, X_train, y_train_oh, X_val, y_val, y_val_oh, n_features, n_classes,
          epochs, batch):
    """Grid-search on the VALIDATION split only.

    Selection never touches the test set - picking a config by test score would
    make the reported number meaningless.

    ECE ALONE IS THE WRONG OBJECTIVE HERE, and selecting on it is a trap worth
    spelling out. ECE measures |confidence - accuracy|, so on a dataset this
    easy a model that is right 99.4% of the time and shouts 100% on every input
    scores a near-perfect ECE of 0.003. That is precisely the behaviour this
    model was built to eliminate: a run selected purely on ECE picked
    label_smoothing=0.0 and came back claiming exactly 100% on 88.7% of inputs,
    with a hard-zero third prediction 15.8% of the time - i.e. it reintroduced
    both UI defects while looking better on paper.

    So the rule is: within a tolerance of the best validation accuracy, and
    subject to a cap on how often the model claims total certainty, take the
    lowest ECE.
    """
    grid = []
    for hidden in [(256, 128), (512, 256), (128, 64), (256,)]:
        for dropout in (0.2, 0.3, 0.4):
            for ls in (0.0, 0.05, 0.10):
                grid.append({"hidden": hidden, "dropout": dropout,
                             "label_smoothing": ls})

    # How often a config may claim total certainty before we refuse it.
    MAX_OVERCONFIDENT = 0.05

    print("\nsweeping %d configurations (validation only)..." % len(grid), flush=True)
    print("  %-14s %8s %6s %9s %8s %9s"
          % ("hidden", "dropout", "ls", "val top1", "val ECE", "says100%"), flush=True)
    print("  " + "-" * 60, flush=True)

    rows = []
    for cfg in grid:
        model = build_and_fit(tf, cfg, X_train, y_train_oh, X_val, y_val_oh,
                              n_features, n_classes, epochs, batch)
        p = model.predict(X_val, verbose=0)
        acc = float((p.argmax(axis=1) == y_val).mean())
        ece = expected_calibration_error(p, y_val)
        over = float((p.max(axis=1) >= 0.999).mean())
        rows.append((cfg, acc, ece, over))
        print("  %-14s %8.1f %6.2f %9.4f %8.4f %8.1f%%"
              % (str(cfg["hidden"]), cfg["dropout"], cfg["label_smoothing"],
                 acc, ece, over * 100), flush=True)

    best_acc = max(r[1] for r in rows)
    eligible = [r for r in rows if r[1] >= best_acc - 0.005]
    usable = [r for r in eligible if r[3] <= MAX_OVERCONFIDENT]
    if not usable:
        print("\n  ! no config stayed under %.0f%% total-certainty; falling back to"
              " lowest ECE overall" % (MAX_OVERCONFIDENT * 100))
        usable = eligible
    winner = min(usable, key=lambda r: r[2])

    print("\n  best val accuracy %.4f; %d within 0.005 of it; %d of those under the"
          " %.0f%% certainty cap" % (best_acc, len(eligible), len(usable),
                                     MAX_OVERCONFIDENT * 100))
    print("  chosen: %s dropout=%.1f ls=%.2f -> val top1 %.4f, val ECE %.4f,"
          " says100%% %.1f%%"
          % (winner[0]["hidden"], winner[0]["dropout"], winner[0]["label_smoothing"],
             winner[1], winner[2], winner[3] * 100))
    return winner[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", default="256,128",
                    help="Comma-separated hidden layer widths")
    ap.add_argument("--tune", action="store_true",
                    help="Grid-search architecture/dropout/label-smoothing on the "
                         "validation split, then train the winner")
    ap.add_argument("--no-compare", action="store_true",
                    help="Skip training the RandomForest baseline")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "disease_nn"))
    args = ap.parse_args()

    # Feature order comes from models/, which is what inference loads. Using the
    # datasets/ copy instead would silently scramble the inputs if they differ.
    sym_path = os.path.join(ROOT, "models", "symptoms_list.json")
    if not os.path.exists(sym_path):
        sym_path = os.path.join(ROOT, "datasets", "symptoms_list.json")
    symptoms = json.load(open(sym_path))

    df = pd.read_csv(os.path.join(ROOT, "datasets", "symptom_disease_dataset.csv"))
    print("raw rows: %d" % len(df))

    # --- deduplicate on the feature pattern -------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=symptoms).reset_index(drop=True)
    print("unique symptom patterns: %d  (dropped %d duplicate rows)"
          % (len(df), before - len(df)))

    # Classes need at least 3 unique patterns to appear in all three splits.
    counts = df["prognosis"].value_counts()
    too_few = counts[counts < 3]
    if len(too_few):
        print("! dropping %d class(es) with <3 unique patterns: %s"
              % (len(too_few), list(too_few.index)))
        df = df[~df["prognosis"].isin(too_few.index)].reset_index(drop=True)

    labels = sorted(df["prognosis"].unique())
    idx_of = {l: i for i, l in enumerate(labels)}
    X = df[symptoms].values.astype("float32")
    y = df["prognosis"].map(idx_of).values.astype("int64")
    print("classes: %d | features: %d" % (len(labels), X.shape[1]))

    from sklearn.model_selection import train_test_split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, random_state=SEED, stratify=y_tmp)
    print("train %d | val %d | test %d" % (len(X_train), len(X_val), len(X_test)))

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    y_train_oh = tf.keras.utils.to_categorical(y_train, len(labels))
    y_val_oh = tf.keras.utils.to_categorical(y_val, len(labels))

    # Deliberately small by default: ~1.5k training rows cannot support anything
    # much wider without memorising. No convolutions or recurrence - the features
    # are an unordered binary set, so there is no locality or sequence to exploit.
    if args.tune:
        cfg = sweep(tf, X_train, y_train_oh, X_val, y_val, y_val_oh,
                    len(symptoms), len(labels), args.epochs, args.batch)
    else:
        cfg = {
            "hidden": tuple(int(w) for w in args.hidden.split(",") if w.strip()),
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
        }
    print("\ntraining final model: %s" % cfg)

    model = build_and_fit(tf, cfg, X_train, y_train_oh, X_val, y_val_oh,
                          len(symptoms), len(labels), args.epochs, args.batch,
                          verbose=2 if not args.tune else 0)
    model.summary()

    nn_probs = model.predict(X_test, verbose=0)
    results = [summarise(nn_probs, y_test, labels, "Neural net")]

    if not args.no_compare:
        # Same split, same rows - the only variable is the model family.
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
        rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        rf_probs = rf.predict_proba(X_test)
        results.append(summarise(rf_probs, y_test, labels, "RandomForest"))

    print("\n" + "=" * 86)
    print("  %-16s %7s %7s %8s %7s %10s %10s %11s" % (
        "model", "top1", "top3", "macroF1", "ECE", "says 100%", "median c", "3rd == 0"))
    print("-" * 86)
    for r in results:
        print_row(r)
    print("=" * 86)
    print("  ECE: lower is better-calibrated. 'says 100%' and '3rd == 0' are the")
    print("  UI problems - the latter makes the API render fewer than 3 results.")

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, "model.keras"))
    json.dump(labels, open(os.path.join(out, "labels.json"), "w"), indent=2)
    json.dump(symptoms, open(os.path.join(out, "symptoms_list.json"), "w"), indent=2)
    json.dump({
        "models": results,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_classes": len(labels), "n_features": len(symptoms),
        "split": "deduplicated on symptom pattern, stratified 70/15/15",
        "config": {"hidden": list(cfg["hidden"]), "dropout": cfg["dropout"],
                   "label_smoothing": cfg["label_smoothing"]},
        "tuned": bool(args.tune),
    }, open(os.path.join(out, "model_metrics.json"), "w"), indent=2)

    print("\nSaved -> %s" % out)
    print("Restart the API; it will use the neural net and keep the forest as fallback.")


if __name__ == "__main__":
    main()
