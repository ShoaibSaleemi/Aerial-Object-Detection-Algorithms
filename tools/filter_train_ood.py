"""
filter_train_ood.py

Removes label lines with class ID 0 (airplane) or 3 (helicopter) from the
train dataset. If a label file becomes empty after removal, both the label
file and its matching image are deleted.

Must be run BEFORE labels_train.py, which remaps the remaining IDs.
"""

import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Class IDs to strip from every label file.
REMOVE_CLASSES = {0, 3}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# (labels_dir, images_dir) pairs to filter.
DATASET_SPLITS = [
    (
        PROJECT_ROOT / "dataset" / "train" / "labels",
        PROJECT_ROOT / "dataset" / "train" / "images",
    ),
]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    total_files = sum(
        len([f for f in os.listdir(label_dir) if f.endswith(".txt")])
        for label_dir, _ in DATASET_SPLITS
        if label_dir.exists()
    )

    processed = 0
    total_removed = 0
    start_time = time.time()

    for label_dir, images_dir in DATASET_SPLITS:
        if not label_dir.exists():
            print(f"[skip] Labels directory not found: {label_dir}")
            continue

        split_removed = 0
        label_files = sorted(label_dir.glob("*.txt"))

        for label_path in label_files:
            with label_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            kept_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                if int(parts[0]) not in REMOVE_CLASSES:
                    kept_lines.append(line)

            if len(kept_lines) == len(lines):
                # Nothing removed — skip rewrite.
                pass
            elif kept_lines:
                # Some lines removed but file is non-empty — rewrite.
                with label_path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(kept_lines) + "\n")
                split_removed += 1
            else:
                # All lines removed — delete label and image.
                label_path.unlink()
                img_path = find_image(images_dir, label_path.stem)
                if img_path is not None:
                    img_path.unlink()
                split_removed += 1

            processed += 1
            elapsed = time.time() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            print(
                f"Progress: {processed}/{total_files} "
                f"({processed / total_files * 100:.1f}%)  "
                f"Removed: {total_removed + split_removed}  "
                f"Elapsed: {minutes}:{seconds:02d}",
                end="\r",
            )

        total_removed += split_removed
        split_name = label_dir.parents[0].name
        print(f"\n[{split_name}] Removed {split_removed} OOD image/label pair(s).")

    print(f"\nDone. Total removed: {total_removed} pair(s).")


if __name__ == "__main__":
    main()
