import subprocess
import sys

scripts = [
    ("01_filter_train_ood.py", "Filtering OOD samples from train dataset"),
    ("02_remap_validation_labels.py", "Remapping validation label IDs for evaluation"),
    ("03_check_validation_labels.py", "Checking class IDs in validation labels"),
]


def main():
    total_parts = len(scripts)

    for i, (script, description) in enumerate(scripts, 1):
        print(f"Part {i}/{total_parts}: {description}")
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
