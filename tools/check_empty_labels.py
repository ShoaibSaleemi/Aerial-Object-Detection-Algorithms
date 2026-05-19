"""
check_empty_labels.py

Scans all .txt label files under a given root directory and reports
any that are empty (zero bytes or only whitespace).
"""

import time
from pathlib import Path

DATASET_ROOT = Path(r"C:\Users\shoai\project\dataset")


def main():
    label_files = sorted(DATASET_ROOT.rglob("*.txt"))

    if not label_files:
        print(f"No .txt files found under {DATASET_ROOT}")
        return

    total = len(label_files)
    empty = []
    start_time = time.time()

    for i, p in enumerate(label_files, 1):
        if p.stat().st_size == 0 or not p.read_text(encoding="utf-8").strip():
            empty.append(p)

        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Progress: {i}/{total} ({i / total * 100:.1f}%)  "
            f"Empty so far: {len(empty)}  "
            f"Elapsed: {minutes}:{seconds:02d}",
            end="\r",
        )

    print()
    print(f"\nScanned : {total} label file(s)")
    print(f"Empty   : {len(empty)}\n")

    for p in empty:
        print(p)


if __name__ == "__main__":
    main()
