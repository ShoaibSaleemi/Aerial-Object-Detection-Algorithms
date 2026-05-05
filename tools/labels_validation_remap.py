import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

label_dirs = [
    str(PROJECT_ROOT / "dataset" / "validation" / "labels"),
]

# Count only .txt files for progress tracking
total_files = sum(
    len([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    for label_dir in label_dirs
)
processed = 0
start_time = time.time()
updated_files = 0

for label_dir in label_dirs:
    for file in sorted(os.listdir(label_dir)):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])

            if cls == 1:  # bird
                parts[0] = "0"
            elif cls == 2:  # drone
                parts[0] = "1"
            elif cls in [0, 3]:  # airplane + helicopter
                parts[0] = "2"  # unknown
            else:
                parts[0] = "2"  # map any unexpected class to unknown for stability

            new_lines.append(" ".join(parts) + "\n")

        if lines != new_lines:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            updated_files += 1

        # Update progress
        processed += 1
        current_tenths = int((processed * 1000) / total_files)
        if current_tenths != int(((processed - 1) * 1000) / total_files) or processed == total_files:
            elapsed = time.time() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            print(
                f"Progress: {processed}/{total_files} ({processed / total_files * 100:.1f}%) "
                f"Elapsed: {minutes}:{seconds:02d}",
                end="\r",
            )

print()  # Newline after completion