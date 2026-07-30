"""
CareBot AI - Skin Image Model Training (transfer learning) - OPTIONAL.

This trains a real CNN classifier that the backend will use automatically
(app/services/image_service.py loads models/image_model/ if it exists).
Until you run this, the app falls back to the transparent feature heuristic.

REQUIREMENTS (not installed by default to keep the base image light):
    pip install "tensorflow>=2.15"       # or tensorflow-cpu

DATA LAYOUT (one folder per class, e.g. from HAM10000 or your own set):
    datasets/skin_images/
        Eczema/                *.jpg
        Psoriasis/             *.jpg
        Fungal Skin Infection/ *.jpg
        Acne/                  *.jpg
        Likely Normal / Healthy Skin/ *.jpg

USAGE:
    python scripts/train_image_model.py --data datasets/skin_images --epochs 12

OUTPUT:
    models/image_model/            (Keras SavedModel — auto-loaded by the API)
    models/image_model/labels.json (ordered class labels)
"""
import argparse
import json
import os

IMG_SIZE = 224
BATCH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to image folder (one subfolder per class)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0,
                        help="Quick test: cap training to ~this many images (0 = use the full set)")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "image_model"))
    args = parser.parse_args()

    # Imported here so the rest of the project doesn't require TensorFlow.
    import tensorflow as tf
    from tensorflow.keras import layers, models

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="training", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="validation", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
    )
    class_names = list(train_ds.class_names)
    print(f"Classes: {class_names}")

    # Quick-test mode: use only a small number of batches so a run finishes in
    # seconds/minutes. NOT for a real model — just to verify the pipeline.
    if args.limit:
        n_batches = max(1, args.limit // BATCH)
        train_ds = train_ds.take(n_batches)
        val_ds = val_ds.take(max(1, n_batches // 4))
        print(f"[--limit] Quick test: using ~{args.limit} images ({n_batches} train batches).")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    # Transfer learning on MobileNetV2 (ImageNet features), frozen backbone.
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    # NOTE: preprocess_input is baked into this graph, so the API feeds raw
    # 0-255 pixels at inference (see image_service._classify_with_cnn).
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, "model.keras"))  # Keras 3 native format
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"\nSaved trained model -> {out}")
    print("Restart the API; it will now use the CNN instead of the heuristic.")


if __name__ == "__main__":
    main()
