import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent

scripts = [
    ("filter_train_ood.py", "Filtering OOD samples from train dataset"),
    ("remap_train_validation_labels.py", "Remapping validation label IDs for evaluation"),
    ("check_validation_labels.py", "Checking class IDs in validation labels"),
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
