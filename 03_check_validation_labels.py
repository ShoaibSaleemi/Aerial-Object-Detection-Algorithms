import os
import time

label_dir = "C:/Users/shoai/project/thesis/dataset/validation/labels"

classes = set()

label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
total_files = len(label_files)
start_time = time.time()

for idx, file in enumerate(label_files, start=1):
    with open(os.path.join(label_dir, file)) as f:
        for line in f:
            if line.strip():
                cls = int(line.split()[0])
                classes.add(cls)

    if idx % 10 == 0 or idx == total_files:
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Progress: {idx}/{total_files} ({idx / total_files * 100:.2f}%) "
            f"Elapsed: {minutes}:{seconds:02d}",
            end="\r",
        )

print()
print("Classes found:", classes)
