import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"

scripts = [
    ("filter_train_validation_ood.py", "Filtering OOD samples from train dataset"),
    ("labels_train_remap.py", "Remapping train label IDs for training"),
    ("labels_validation_remap.py", "Remapping validation label IDs for evaluation"),
    ("convert_labels_fasterrcnn.py", "Converting labels for Faster R-CNN format"),
    ("labels_validation_check.py", "Checking class IDs in validation labels"),
]


def main():
    total_parts = len(scripts)

    for i, (script, description) in enumerate(scripts, 1):
        print(f"Part {i}/{total_parts}: {description}")
        try:
            subprocess.run([sys.executable, str(TOOLS_DIR / script)], check=True, cwd=str(TOOLS_DIR))
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
