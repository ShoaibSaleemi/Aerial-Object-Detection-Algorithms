"""
Dataset label statistics — train / val / test.

For each split and each class (0=bird, 1=drone, 2=unknown) reports:
  - Total label lines  (object instances)
  - Number of image files containing at least one of that class

Warns immediately when a class other than 0, 1, 2 is encountered, naming the file.
Writes results to tools/label_stats.csv alongside the terminal table.
"""

import csv
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALL_SPLITS = {
    "train":      PROJECT_ROOT / "dataset 2" / "train"      / "labels",
    "validation": PROJECT_ROOT / "dataset 2" / "validation" / "labels",
    "test":       PROJECT_ROOT / "dataset 2" / "test"   / "labels",
}

# Set RUN_ALL_SPLITS = True to scan every split regardless of the flags below.
RUN_ALL_SPLITS = True
ENABLED_SPLITS = {
    "train":      True,
    "validation": True,
    "test":       True,
}

SPLITS = ALL_SPLITS if RUN_ALL_SPLITS else {
    k: v for k, v in ALL_SPLITS.items() if ENABLED_SPLITS.get(k, False)
}

KNOWN_CLASSES = {0, 1, 2}
CLASS_NAMES   = {0: "bird", 1: "drone", 2: "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-split analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_split(split_name: str, label_dir: Path) -> dict:
    instance_counts: dict[int, int] = defaultdict(int)
    file_counts:     dict[int, int] = defaultdict(int)
    unknown_warnings: list[tuple]   = []

    label_files = sorted(p for p in label_dir.iterdir() if p.suffix == ".txt")
    total_files = len(label_files)
    start = time.time()

    for idx, fpath in enumerate(label_files, start=1):
        classes_in_file: set[int] = set()

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                instance_counts[cls] += 1
                classes_in_file.add(cls)

                if cls not in KNOWN_CLASSES:
                    unknown_warnings.append((cls, fpath.name))

        for cls in classes_in_file:
            file_counts[cls] += 1

        if idx % 50 == 0 or idx == total_files:
            elapsed = time.time() - start
            m, s = divmod(int(elapsed), 60)
            print(
                f"  [{split_name}] {idx}/{total_files}"
                f" ({idx / total_files * 100:.1f}%)  Elapsed: {m}:{s:02d}",
                end="\r",
            )

    print()
    return {
        "instance_counts":  dict(instance_counts),
        "file_counts":      dict(file_counts),
        "total_files":      total_files,
        "unknown_warnings": unknown_warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Terminal table
# ─────────────────────────────────────────────────────────────────────────────
def print_table(results: dict[str, dict], all_classes: list[int]):
    col_w = 14
    hdr_w = 12

    header = f"{'Split':<{hdr_w}}"
    for c in all_classes:
        name = CLASS_NAMES.get(c, f"cls{c}")
        header += f"  {'['+str(c)+'] '+name+' inst':>{col_w}}  {'files':>{col_w}}"
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for split, r in results.items():
        row = f"{split:<{hdr_w}}"
        for cls in all_classes:
            inst  = r["instance_counts"].get(cls, 0)
            files = r["file_counts"].get(cls, 0)
            row += f"  {inst:>{col_w},}  {files:>{col_w},}"
        print(row)

    print(sep)
    total_row = f"{'TOTAL':<{hdr_w}}"
    for cls in all_classes:
        inst  = sum(r["instance_counts"].get(cls, 0) for r in results.values())
        files = sum(r["file_counts"].get(cls, 0)     for r in results.values())
        total_row += f"  {inst:>{col_w},}  {files:>{col_w},}"
    print(total_row)
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(results: dict[str, dict], all_classes: list[int], out_path: Path):
    rows = []
    for split, r in results.items():
        for cls in all_classes:
            rows.append({
                "split":               split,
                "class_id":            cls,
                "class_name":          CLASS_NAMES.get(cls, "OTHER"),
                "instances":           r["instance_counts"].get(cls, 0),
                "files_with_class":    r["file_counts"].get(cls, 0),
                "total_files_in_split": r["total_files"],
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results: dict[str, dict] = {}
    all_warnings: list[tuple[str, int, str]] = []

    for split, label_dir in SPLITS.items():
        if not label_dir.exists():
            print(f"  [SKIP] {split}: labels directory not found ({label_dir})")
            continue
        print(f"\nScanning {split}  ({label_dir}) ...")
        r = analyse_split(split, label_dir)
        results[split] = r
        for cls, fname in r["unknown_warnings"]:
            all_warnings.append((split, cls, fname))

    if not results:
        print("No splits found. Check SPLITS paths.")
        return

    # Unexpected class warnings
    if all_warnings:
        print(f"\n{'!' * 60}")
        print(f"  WARNING: {len(all_warnings)} line(s) with unexpected class IDs:")
        for split, cls, fname in all_warnings:
            print(f"    [{split}]  class {cls}  →  {fname}")
        print(f"{'!' * 60}")
    else:
        print("\n  OK — only classes 0, 1, 2 found across all splits.")

    all_classes = sorted(
        KNOWN_CLASSES | {cls for r in results.values() for cls in r["instance_counts"]}
    )

    print_table(results, all_classes)

    csv_path = Path(__file__).parent / "label_stats.csv"
    write_csv(results, all_classes, csv_path)


if __name__ == "__main__":
    main()
