import os
import time


dataset_splits = [
    ("dataset/train/labels", "dataset/train/images"),
    ("dataset/validation/labels", "dataset/validation/images"),
]

# Count label files across all configured splits for a single progress bar.
total_files = sum(
    len([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    for label_dir, _ in dataset_splits
)
processed = 0
start_time = time.time()

for label_dir, image_dir in dataset_splits:
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    for file in label_files:
        path = os.path.join(label_dir, file)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check if any line has class 0 (helicopter) or 3 (plane)
        has_unwanted = any(int(line.strip().split()[0]) in [0, 3] for line in lines if line.strip())

        if has_unwanted:
            # Remove label file
            os.remove(path)

            # Remove corresponding image file
            img_file = file.replace(".txt", ".jpg")
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
                end="\r",
            )

print()  # Newline after completion
