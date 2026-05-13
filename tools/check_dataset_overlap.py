"""
Check for duplicate images between two dataset directories.

Uses two methods:
  1. Exact hash (MD5 of raw bytes) — finds pixel-perfect duplicates
  2. Perceptual hash (average hash via PIL) — finds visually identical images
     even if they were re-exported, resized, or recompressed

Usage:
    python tools/check_dataset_overlap.py <dir_a> <dir_b>

Example — check if AOD4 images appear in the large dataset test split:
    python tools/check_dataset_overlap.py path/to/aod4/images dataset/test/images

The script prints overlapping filenames and saves results to
runs/overlap_check_<timestamp>.txt
"""

import hashlib
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PHASH_SIZE = 16  # larger = more sensitive; 16 is a good balance


def md5_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def perceptual_hash(path: Path) -> str:
    """Average hash: resize to PHASH_SIZE x PHASH_SIZE grayscale, threshold at mean."""
    with Image.open(path) as img:
        img = img.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float32)
    mean = pixels.mean()
    bits = (pixels >= mean).flatten()
    # pack bits into hex string
    value = int("".join("1" if b else "0" for b in bits), 2)
    return f"{value:0{PHASH_SIZE * PHASH_SIZE // 4}x}"


def hamming_distance(h1: str, h2: str) -> int:
    i1 = int(h1, 16)
    i2 = int(h2, 16)
    x = i1 ^ i2
    return bin(x).count("1")


def index_directory(directory: Path, method: str) -> dict:
    """Returns {hash: [path, ...]} for all images in directory."""
    index = {}
    files = [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    total = len(files)
    print(f"  Indexing {total} images in {directory} ...")
    for i, path in enumerate(files, 1):
        if i % 500 == 0 or i == total:
            print(f"  {i}/{total}", end="\r")
        try:
            h = md5_hash(path) if method == "exact" else perceptual_hash(path)
            index.setdefault(h, []).append(path)
        except Exception as e:
            print(f"  Warning: could not hash {path.name}: {e}")
    print()
    return index


def find_exact_duplicates(dir_a: Path, dir_b: Path):
    print("\n[1/2] Exact hash check (pixel-perfect duplicates)...")
    index_a = index_directory(dir_a, "exact")
    index_b = index_directory(dir_b, "exact")

    matches = []
    for h, paths_b in index_b.items():
        if h in index_a:
            for pb in paths_b:
                for pa in index_a[h]:
                    matches.append((pa, pb))

    return matches


def find_perceptual_duplicates(dir_a: Path, dir_b: Path, max_distance: int = 5):
    """
    Finds visually similar images even if re-exported/recompressed.
    max_distance: Hamming distance threshold (0=identical, lower=stricter).
    5 out of 256 bits is ~98% similar.
    """
    print(f"\n[2/2] Perceptual hash check (similar images, max Hamming distance={max_distance})...")
    index_a = index_directory(dir_a, "phash")
    index_b = index_directory(dir_b, "phash")

    hashes_a = list(index_a.keys())
    matches = []
    total = len(index_b)
    for i, (hb, paths_b) in enumerate(index_b.items(), 1):
        if i % 200 == 0 or i == total:
            print(f"  Comparing {i}/{total}", end="\r")
        for ha in hashes_a:
            if hamming_distance(ha, hb) <= max_distance:
                for pb in paths_b:
                    for pa in index_a[ha]:
                        matches.append((pa, pb))
    print()
    return matches


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    dir_a = Path(sys.argv[1])
    dir_b = Path(sys.argv[2])

    if not dir_a.exists():
        print(f"Error: directory A not found: {dir_a}")
        sys.exit(1)
    if not dir_b.exists():
        print(f"Error: directory B not found: {dir_b}")
        sys.exit(1)

    print(f"Checking overlap between:")
    print(f"  A: {dir_a}")
    print(f"  B: {dir_b}")

    exact_matches = find_exact_duplicates(dir_a, dir_b)
    perceptual_matches = find_perceptual_duplicates(dir_a, dir_b)

    # Deduplicate perceptual matches (exact matches are a subset)
    exact_b_names = {str(pb) for _, pb in exact_matches}
    perceptual_only = [(pa, pb) for pa, pb in perceptual_matches if str(pb) not in exact_b_names]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlap_check_{timestamp}.txt"

    lines = []
    lines.append(f"Overlap check: {dir_a}  vs  {dir_b}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append(f"=== Exact duplicates: {len(exact_matches)} ===")
    for pa, pb in exact_matches:
        lines.append(f"  A: {pa.name}  <->  B: {pb.name}")
    lines.append("")
    lines.append(f"=== Perceptual-only duplicates (visually similar, not pixel-perfect): {len(perceptual_only)} ===")
    for pa, pb in perceptual_only:
        lines.append(f"  A: {pa.name}  <->  B: {pb.name}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Exact duplicates found:              {len(exact_matches)}")
    print(f"Perceptual-only duplicates found:    {len(perceptual_only)}")
    print(f"Results saved to: {out_path}")

    if len(exact_matches) == 0 and len(perceptual_only) == 0:
        print("\nNo overlap detected. Datasets appear to be independent.")
    elif len(exact_matches) > 0:
        print(f"\nWARNING: {len(exact_matches)} exact duplicates found between the two directories.")
        print("If dir_b is your test split, this is data leakage and must be resolved.")


if __name__ == "__main__":
    main()
