"""
CareBot AI - Skin Image CNN training (MobileNetV2 transfer learning).

Trains the classifier that backend/app/services/image_service.py loads
automatically from models/image_model/. Until this has been run, the API falls
back to the transparent feature heuristic.

REQUIREMENTS
    pip install -r backend/requirements-ml.txt      # tensorflow-cpu

DATA
    Build the manifest first:
        python scripts/prepare_skin_dataset.py --ham <dir> --dermnet <dir>

    Legacy folder layout (one subfolder per class) still works via --data.

WHAT THIS DOES THAT A NAIVE SCRIPT DOES NOT
  * Splits on `group_id`, not on rows. HAM10000 has several photos of the same
    lesion; a random split leaks them across train/val and inflates accuracy.
  * Applies class weights. The merged set is heavily imbalanced (Melanocytic
    Nevus alone is a majority of HAM10000), and an unweighted model learns to
    answer with the majority class and still reports high accuracy.
  * Fine-tunes. Phase 1 trains the head on a frozen backbone; phase 2 unfreezes
    the top layers at a low LR, which is where most of the real accuracy comes
    from on dermoscopy images (ImageNet features alone transfer poorly).
  * Holds out a real test split and writes per-class precision/recall/F1 plus a
    confusion matrix to models/image_model/model_metrics.json. Overall accuracy
    on this dataset is close to meaningless - read macro recall.

USAGE
    python scripts/train_image_model.py                      # full run
    python scripts/train_image_model.py --limit 400 --epochs 1 --finetune-epochs 1
    python scripts/train_image_model.py --data datasets/skin_images   # legacy

OUTPUT
    models/image_model/model.keras
    models/image_model/labels.json
    models/image_model/model_metrics.json
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

IMG_SIZE = 224
BATCH = 32
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_MANIFEST = os.path.join(ROOT, "datasets", "skin_manifest.csv")
DEFAULT_OUT = os.path.join(ROOT, "models", "image_model")

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_manifest(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["filepath"], r["label"], r["group_id"]))
    if not rows:
        raise SystemExit("Manifest %s is empty" % path)
    return rows


def load_folders(root):
    """Legacy layout: datasets/skin_images/<Class>/*.jpg (each image its own group)."""
    rows = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith(IMG_EXT):
                rows.append((os.path.join(d, name), cls, os.path.splitext(name)[0]))
    if not rows:
        raise SystemExit("No images found under " + root)
    return rows


def grouped_split(rows, val_frac=0.15, test_frac=0.15):
    """Split by group_id so no lesion appears in more than one split.

    Groups are bucketed per class first, so each split keeps roughly the same
    class mix even though the unit of splitting is the group.
    """
    by_class = defaultdict(set)
    for _, label, gid in rows:
        by_class[label].add(gid)

    train_g, val_g, test_g = set(), set(), set()
    rng = random.Random(SEED)
    for label, gids in by_class.items():
        gids = sorted(gids)
        rng.shuffle(gids)
        n = len(gids)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
        test_g.update(gids[:n_test])
        val_g.update(gids[n_test:n_test + n_val])
        train_g.update(gids[n_test + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for path, label, gid in rows:
        if gid in test_g:
            splits["test"].append((path, label))
        elif gid in val_g:
            splits["val"].append((path, label))
        else:
            splits["train"].append((path, label))
    return splits


def make_ds(tf, items, labels, training, batch, limit=0):
    """tf.data pipeline over (filepath, label) pairs."""
    if limit and training:
        items = items[:limit]
    idx = {l: i for i, l in enumerate(labels)}
    paths = [p for p, _ in items]
    ys = [idx[l] for _, l in items]

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True)

    def _load(p, y):
        img = tf.io.read_file(p)
        # expand_animations=False keeps the static shape known for resize.
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_model(tf, n_classes):
    from tensorflow.keras import layers, models

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.1, value_range=(0, 255)),
    ], name="augment")

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = aug(inputs)
    # preprocess_input is baked into the graph, so the API feeds raw 0-255
    # pixels at inference time (see image_service._classify_with_cnn).
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs), base


def unfreeze(tf, base, n_layers):
    """Unfreeze the top n_layers, but keep BatchNorm frozen.

    Un-freezing BN layers on small batches destabilises transfer learning -
    the running statistics get overwritten by noisy batch estimates.
    """
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--data", help="Legacy: folder with one subfolder per class")
    ap.add_argument("--epochs", type=int, default=10, help="phase 1 (frozen backbone)")
    ap.add_argument("--finetune-epochs", type=int, default=8, help="phase 2 (unfrozen top)")
    ap.add_argument("--finetune-layers", type=int, default=40)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap training images for a quick pipeline smoke test")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.data:
        rows = load_folders(args.data)
    else:
        if not os.path.exists(args.manifest):
            raise SystemExit(
                "No manifest at %s.\nRun scripts/prepare_skin_dataset.py first, "
                "or pass --data <folder>." % args.manifest)
        rows = load_manifest(args.manifest)

    labels = sorted(set(l for _, l, _ in rows))
    splits = grouped_split(rows)
    print("Classes (%d): %s" % (len(labels), labels))
    for name in ("train", "val", "test"):
        print("  %-5s %6d images" % (name, len(splits[name])))

    import numpy as np
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    train_ds = make_ds(tf, splits["train"], labels, True, args.batch, args.limit)
    val_ds = make_ds(tf, splits["val"], labels, False, args.batch)
    test_ds = make_ds(tf, splits["test"], labels, False, args.batch)

    # Class weights, inverse-frequency, normalised to mean 1.
    train_counts = Counter(l for _, l in splits["train"])
    total = sum(train_counts.values())
    raw = {i: total / (len(labels) * max(train_counts[l], 1))
           for i, l in enumerate(labels)}
    mean_w = sum(raw.values()) / len(raw)
    class_weight = {i: w / mean_w for i, w in raw.items()}
    print("\nclass weights:")
    for i, l in enumerate(labels):
        print("  %-32s n=%-6d w=%.2f" % (l, train_counts[l], class_weight[i]))

    model, base = build_model(tf, len(labels))

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=2, min_lr=1e-7),
    ]

    print("\n=== Phase 1: frozen backbone, training the head ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weight, callbacks=cbs)

    if args.finetune_epochs > 0:
        print("\n=== Phase 2: fine-tuning the top %d layers ===" % args.finetune_layers)
        unfreeze(tf, base, args.finetune_layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                  class_weight=class_weight, callbacks=cbs)

    # --- evaluation on the held-out test split ---------------------------
    print("\n=== Test-set evaluation ===")
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    report = classification_report(y_true, y_pred, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()

    print(classification_report(y_true, y_pred, labels=list(range(len(labels))),
                                target_names=labels, zero_division=0))
    macro_recall = report["macro avg"]["recall"]
    print("accuracy      %.4f" % report["accuracy"])
    print("macro recall  %.4f   <- the number that matters on an imbalanced set"
          % macro_recall)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, "model.keras"))
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    with open(os.path.join(out, "model_metrics.json"), "w") as f:
        json.dump({
            "accuracy": report["accuracy"],
            "macro_recall": macro_recall,
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {l: report[l] for l in labels},
            "confusion_matrix": cm,
            "labels": labels,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "split": "grouped by lesion/group_id",
        }, f, indent=2)

    print("\nSaved -> %s" % out)
    print("Restart the API; it will use the CNN instead of the heuristic.")


if __name__ == "__main__":
    main()
