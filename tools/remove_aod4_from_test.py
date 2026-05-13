"""
Remove AOD4 images (and their labels) from the thesis train, validation, and
test splits to prevent data leakage in evaluation.

Moves contaminated files to dataset/<split>_aod4_removed/ so they can be
inspected or restored if needed.

Usage:
    python tools/remove_aod4_from_test.py [--dry-run]

    --dry-run: only print what would be removed, without moving anything.
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AOD4_DIR = Path(r"C:\Users\shoai\project\AOD4 dataset")
DATASET_DIR = PROJECT_ROOT / "dataset"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "validation", "test"]


def main():
    dry_run = "--dry-run" in sys.argv

    # Build set of AOD4 filenames once
    aod4_names = {p.name for p in AOD4_DIR.rglob("*") if p.suffix.lower() in EXTS}
    print(f"AOD4 unique image filenames: {len(aod4_names)}\n")

    for split in SPLITS:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"
        removed_images_dir = DATASET_DIR / f"{split}_aod4_removed" / "images"
        removed_labels_dir = DATASET_DIR / f"{split}_aod4_removed" / "labels"

        if not images_dir.exists():
            print(f"[{split}] images dir not found, skipping.")
            continue

        all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in EXTS]
        contaminated = [p for p in all_images if p.name in aod4_names]

        print(f"[{split}] total: {len(all_images)}  |  AOD4 overlap: {len(contaminated)}")

        if dry_run:
            for p in contaminated[:5]:
                print(f"  would move: {p.name}")
            if len(contaminated) > 5:
                print(f"  ... and {len(contaminated) - 5} more")
            continue

        if len(contaminated) == 0:
            print(f"  No contaminated files. Nothing to do.")
            continue

        removed_images_dir.mkdir(parents=True, exist_ok=True)
        removed_labels_dir.mkdir(parents=True, exist_ok=True)

        moved_images = 0
        moved_labels = 0
        for img_path in contaminated:
            shutil.move(str(img_path), str(removed_images_dir / img_path.name))
            moved_images += 1
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.move(str(label_path), str(removed_labels_dir / label_path.name))
                moved_labels += 1

        remaining = len(all_images) - moved_images
        print(f"  Moved {moved_images} images and {moved_labels} labels -> {split}_aod4_removed/")
        print(f"  Clean {split} split now has {remaining} images.")

    if dry_run:
        print("\n[DRY RUN] No files were moved. Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
