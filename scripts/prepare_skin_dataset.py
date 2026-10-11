"""
CareBot AI - build a unified skin-image training manifest.

Merges two public datasets into ONE label space so the CNN covers both the
pigmented/neoplastic lesions and the common inflammatory conditions the old
heuristic targeted:

  Pigmented / neoplastic  (HAM10000, dermoscopy images)
      Melanoma, Melanocytic Nevus, Basal Cell Carcinoma, Actinic Keratosis,
      Benign Keratosis, Dermatofibroma, Vascular Lesion

  Common inflammatory    (DermNet, ordinary clinical photos)
      Acne, Eczema, Psoriasis, Fungal Skin Infection

  Optional
      Likely Normal / Healthy Skin   (--normal <folder>)

Instead of copying several GB of images around, this writes a MANIFEST that
points at the originals:

    datasets/skin_manifest.csv
        filepath,label,group_id,source

`group_id` is the part that makes the evaluation honest. HAM10000 contains
multiple photographs of the SAME lesion (~7.5k unique lesion_id across ~10k
images). Split randomly and the same lesion lands in both train and
validation, inflating accuracy by several points. train_image_model.py splits
on group_id, never on individual rows.

DOWNLOADS (both free, Kaggle account required)
    HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    DermNet:  https://www.kaggle.com/datasets/shubhamgoel27/dermnet

USAGE
    python scripts/prepare_skin_dataset.py \
        --ham  path/to/ham10000 \
        --dermnet path/to/dermnet \
        [--normal path/to/healthy_skin_images]
"""
import argparse
import csv
import glob
import os
from collections import Counter

# HAM10000 `dx` code -> readable class name.
HAM_DX = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

# DermNet folder names are long and inconsistent, so match on a lowercase
# substring. Anything not listed here is deliberately skipped.
#
# DermNet's own "Melanoma Skin Cancer Nevi and Moles" folder is excluded on
# purpose: HAM10000 already covers those classes with controlled dermoscopy
# images, and feeding one class from two very different imaging modalities
# teaches the model the modality instead of the disease.
DERMNET_MAP = [
    ("acne and rosacea", "Acne"),
    ("eczema", "Eczema"),
    ("psoriasis", "Psoriasis"),
    ("tinea ringworm candidiasis", "Fungal Skin Infection"),
]

NORMAL_LABEL = "Likely Normal / Healthy Skin"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _find_ham_image(src, image_id):
    for part in ("", "HAM10000_images_part_1", "HAM10000_images_part_2",
                 "images", "ham10000_images", "HAM10000_images"):
        hit = glob.glob(os.path.join(src, part, image_id + ".jpg"))
        if hit:
            return hit[0]
    return None


def collect_ham(src):
    """Rows from HAM10000, keyed by lesion_id so repeat shots group together."""
    meta = os.path.join(src, "HAM10000_metadata.csv")
    if not os.path.exists(meta):
        raise SystemExit("HAM10000_metadata.csv not found in " + src)

    rows, missing = [], 0
    with open(meta, newline="") as f:
        for r in csv.DictReader(f):
            label = HAM_DX.get(r["dx"])
            if not label:
                continue
            path = _find_ham_image(src, r["image_id"])
            if not path:
                missing += 1
                continue
            # lesion_id groups the duplicate shots of a single lesion.
            rows.append((os.path.abspath(path), label, r["lesion_id"], "ham10000"))
    if missing:
        print("  ! %d HAM10000 images listed in the CSV were not found on disk" % missing)
    return rows


def collect_dermnet(src):
    """Rows from DermNet. Each image is its own group (no known duplicates)."""
    rows = []
    # The Kaggle release ships train/ and test/; we re-split ourselves, so pull
    # from both and treat them as a single pool.
    roots = [os.path.join(src, d) for d in ("train", "test")]
    roots = [r for r in roots if os.path.isdir(r)] or [src]

    for root in roots:
        for folder in sorted(os.listdir(root)):
            fpath = os.path.join(root, folder)
            if not os.path.isdir(fpath):
                continue
            low = folder.lower()
            label = None
            for key, lab in DERMNET_MAP:
                if key in low:
                    label = lab
                    break
            if not label:
                continue
            for name in os.listdir(fpath):
                if name.lower().endswith(IMG_EXT):
                    p = os.path.join(fpath, name)
                    rows.append((os.path.abspath(p), label,
                                 os.path.splitext(name)[0], "dermnet"))
    return rows


def collect_normal(src):
    rows = []
    for name in sorted(os.listdir(src)):
        if name.lower().endswith(IMG_EXT):
            p = os.path.join(src, name)
            rows.append((os.path.abspath(p), NORMAL_LABEL,
                         os.path.splitext(name)[0], "normal"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham", help="Folder with HAM10000_metadata.csv + image folders")
    ap.add_argument("--dermnet", help="DermNet root (containing train/ and test/)")
    ap.add_argument("--normal", help="Optional folder of healthy-skin photos")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "datasets", "skin_manifest.csv"))
    args = ap.parse_args()

    if not (args.ham or args.dermnet or args.normal):
        raise SystemExit("Give at least one of --ham / --dermnet / --normal")

    rows = []
    if args.ham:
        print("Scanning HAM10000 in %s ..." % args.ham)
        r = collect_ham(args.ham)
        print("  %d images" % len(r))
        rows += r
    if args.dermnet:
        print("Scanning DermNet in %s ..." % args.dermnet)
        r = collect_dermnet(args.dermnet)
        print("  %d images" % len(r))
        rows += r
    if args.normal:
        print("Scanning healthy-skin folder %s ..." % args.normal)
        r = collect_normal(args.normal)
        print("  %d images" % len(r))
        rows += r

    if not rows:
        raise SystemExit("No images collected - check the paths and folder names.")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "group_id", "source"])
        w.writerows(rows)

    counts = Counter(r[1] for r in rows)
    groups = len(set(r[2] for r in rows))
    print("\nWrote %d rows (%d distinct lesions/groups) -> %s\n" % (len(rows), groups, out))
    print("%-32s%8s%8s" % ("class", "images", "share"))
    print("-" * 48)
    for label, n in counts.most_common():
        print("%-32s%8d%7.1f%%" % (label, n, n / len(rows) * 100))

    top = counts.most_common()[0]
    bottom = counts.most_common()[-1]
    ratio = top[1] / bottom[1]
    print("\nImbalance ratio %.0f:1  (%s vs %s)" % (ratio, top[0], bottom[0]))
    if ratio > 5:
        print("  -> training applies class weights; judge the result on per-class\n"
              "     recall in model_metrics.json, NOT on overall accuracy.")
    if NORMAL_LABEL not in counts:
        print("\nNote: no '%s' class, so every photo will be assigned some\n"
              "condition. The API compensates with a confidence-abstention\n"
              "threshold, but supplying --normal images is the better fix." % NORMAL_LABEL)
    print("\nNext:  python scripts/train_image_model.py")


if __name__ == "__main__":
    main()
