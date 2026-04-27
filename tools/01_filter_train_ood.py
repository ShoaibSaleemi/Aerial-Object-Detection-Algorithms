import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
label_dir = str(PROJECT_ROOT / "dataset" / "train" / "labels")
image_dir = str(PROJECT_ROOT / "dataset" / "train" / "images")

# Get list of label files
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
total_files = len(label_files)
processed = 0
start_time = time.time()

for file in label_files:
    path = os.path.join(label_dir, file)

    with open(path, 'r') as f:
        lines = f.readlines()

    # Check if any line has class 0 (helicopter) or 3 (plane)
    has_unwanted = any(int(line.strip().split()[0]) in [0, 3] for line in lines if line.strip())

    if has_unwanted:
        # Remove label file
        os.remove(path)

        # Remove corresponding image file
        img_file = file.replace('.txt', '.jpg')
        img_path = os.path.join(image_dir, img_file)
        if os.path.exists(img_path):
            os.remove(img_path)

    # Update progress
    processed += 1
    if processed % 10 == 0 or processed == total_files:
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Progress: {processed}/{total_files} ({processed / total_files * 100:.2f}%) "
            f"Elapsed: {minutes}:{seconds:02d}",
            end='\r',
        )

print()  # Newline after completion
