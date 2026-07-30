"""
Organize the HAM10000 dermatology dataset into the per-class folder layout
that scripts/train_image_model.py expects.

HAM10000 ships as loose images + a metadata CSV (diagnosis in the `dx` column),
not as class folders. This script reads the CSV and copies each image into
datasets/skin_images/<Readable Class Name>/.

DOWNLOAD (free, requires a Kaggle account):
    https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
Unzip it somewhere, then point --src at that folder. It should contain:
    HAM10000_metadata.csv
    HAM10000_images_part_1/  (and _part_2/)   *.jpg

USAGE:
    python scripts/prepare_ham10000.py --src path/to/ham10000
    python scripts/train_image_model.py --data datasets/skin_images --epochs 12
"""
import argparse
import csv
import glob
import os
import shutil

# HAM10000 `dx` codes -> human-readable class names used as folder/label names.
DX_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}


def find_image(src: str, image_id: str) -> str | None:
    for part in ("", "HAM10000_images_part_1", "HAM10000_images_part_2", "images", "ham10000_images"):
        hit = glob.glob(os.path.join(src, part, f"{image_id}.jpg"))
        if hit:
            return hit[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing HAM10000_metadata.csv + image folders")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "datasets", "skin_images"))
    args = ap.parse_args()

    meta = os.path.join(args.src, "HAM10000_metadata.csv")
    if not os.path.exists(meta):
        raise SystemExit(f"HAM10000_metadata.csv not found in {args.src}")

    out = os.path.abspath(args.out)
    copied, missing = 0, 0
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            cls = DX_NAMES.get(row["dx"], row["dx"])
            src_img = find_image(args.src, row["image_id"])
            if not src_img:
                missing += 1
                continue
            dst_dir = os.path.join(out, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src_img, os.path.join(dst_dir, f"{row['image_id']}.jpg"))
            copied += 1
            if copied % 1000 == 0:
                print(f"  copied {copied} images...")

    print(f"\nDone. Copied {copied} images into {out}" + (f"  ({missing} not found)" if missing else ""))
    print("Now train:  python scripts/train_image_model.py --data datasets/skin_images --epochs 12")


if __name__ == "__main__":
    main()
