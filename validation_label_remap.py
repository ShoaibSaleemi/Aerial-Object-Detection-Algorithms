import os
import time

label_dirs = [
    "C:/Users/shoai/thesis/YOLOv8/data/validation/labels",
]

# Count total files for progress tracking
total_files = sum(len(os.listdir(label_dir)) for label_dir in label_dirs)
processed = 0
start_time = time.time()

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])

            if cls == 1: # bird
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")
            elif cls == 2: # drone
                parts[0] = "1"
                new_lines.append(" ".join(parts) + "\n")
            elif cls in [0, 3]:  # airplane + helicopter
                parts[0] = "2"   # unknown  
  
        new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

        # Update progress
        processed += 1
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"Progress: {processed}/{total_files} ({processed / total_files * 100:.2f}%) Elapsed: {minutes}:{seconds:02d}", end='\r')

print()  # Newline after completion