"""
Copies test images into subfolders by dominant class ID.

Output layout inside dataset/test/:
    bird/      <- class 0 is most frequent in the label
    drone/     <- class 1 is most frequent
    unknown/   <- class 2 is most frequent
    no_label/  <- label file missing or empty

Images are copied; originals are not touched.
"""
import shutil
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = PROJECT_ROOT / "dataset 2" / "test" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset 2" / "test" / "labels"
OUTPUT_DIR  = PROJECT_ROOT / "dataset 2" / "test"

CLASS_NAMES = {0: "bird", 1: "drone", 2: "unknown"}

# ── Toggle ─────────────────────────────────────────────────────────────────
# True  → split into subfolders named by dominant class (bird / drone / unknown)
# False → split into subfolders named by the first PREFIX_LENGTH characters of
#         the image filename (e.g. "20190" for "20190925_111757…")
SPLIT_BY_CLASS = True
PREFIX_LENGTH  = 7          # only used when SPLIT_BY_CLASS = False
# ───────────────────────────────────────────────────────────────────────────

if SPLIT_BY_CLASS:
    for name in list(CLASS_NAMES.values()) + ["no_label"]:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

print(f"IMAGES_DIR: {IMAGES_DIR}")
print(f"LABELS_DIR: {LABELS_DIR}")

image_files = sorted(f for f in IMAGES_DIR.iterdir() if f.is_file())
total = len(image_files)
start = time.time()
counts = Counter()

for idx, img_path in enumerate(image_files, start=1):
    label_path = LABELS_DIR / (img_path.stem + ".txt")

    if SPLIT_BY_CLASS:
        if not label_path.exists():
            folder = OUTPUT_DIR / "no_label"
            counts["no_label"] += 1
        else:
            class_counter = Counter()
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # bbox (5) or polygon (7+)
                        class_counter[int(parts[0])] += 1

            if not class_counter:
                folder = OUTPUT_DIR / "no_label"
                counts["no_label"] += 1
            else:
                dominant = class_counter.most_common(1)[0][0]
                folder_name = CLASS_NAMES.get(dominant, f"class{dominant}")
                folder = OUTPUT_DIR / folder_name
                counts[folder_name] += 1
    else:
        folder_name = img_path.stem[:PREFIX_LENGTH]
        folder = OUTPUT_DIR / folder_name
        counts[folder_name] += 1

    folder.mkdir(exist_ok=True)
    shutil.copy2(img_path, folder / img_path.name)
    if idx % 50 == 0 or idx == total:
        elapsed = time.time() - start
        m, s = divmod(int(elapsed), 60)
        print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)  Elapsed: {m}:{s:02d}", end="\r")

print()
print("Done.")
for name, n in sorted(counts.items()):
    print(f"  {name:>10}: {n} images")
