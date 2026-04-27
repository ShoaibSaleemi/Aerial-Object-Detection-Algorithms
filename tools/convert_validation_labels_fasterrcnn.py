import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_labels_for_images(
    images_dir: Path, src_labels_dir: Path, dst_labels_dir: Path
) -> Tuple[int, int]:
    """Convert YOLO normalized labels to absolute-pixel Faster R-CNN labels."""
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )

    total = len(image_files)
    missing = 0
    processed = 0
    start_time = time.time()

    for img_path in image_files:
        processed += 1
        src_label = src_labels_dir / f"{img_path.stem}.txt"
        dst_label = dst_labels_dir / f"{img_path.stem}.txt"

        if not src_label.exists():
            missing += 1
            dst_label.write_text("")
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        out_lines: List[str] = []
        with src_label.open("r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        for raw_line in raw_lines:
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0])) + 1
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2.0) * width
            y1 = (cy - h / 2.0) * height
            x2 = (cx + w / 2.0) * width
            y2 = (cy + h / 2.0) * height
            out_lines.append(f"{cls_id} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}")

        dst_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

        # Update progress
        current_tenths = int((processed * 1000) / total)
        if current_tenths != int(((processed - 1) * 1000) / total) or processed == total:
            elapsed = time.time() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            print(
                f"Progress: {processed}/{total} ({processed / total * 100:.2f}%) "
                f"Elapsed: {minutes}:{seconds:02d}",
                end="\r",
            )

    print()
    return total, missing


def parse_data_yaml(data_yaml_path: Path) -> Tuple[Path, Path, Dict[int, str]]:
    """Parse minimal train/val and names fields from YOLO data.yaml."""
    try:
        import yaml  # type: ignore

        with data_yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        root = Path(cfg["path"])
        train_images = root / cfg["train"]
        val_images = root / cfg["val"]

        names_raw = cfg.get("names", {})
        if isinstance(names_raw, dict):
            names = {int(k): str(v) for k, v in names_raw.items()}
        elif isinstance(names_raw, list):
            names = {i: str(v) for i, v in enumerate(names_raw)}
        else:
            names = {}

        return train_images, val_images, names
    except Exception:
        root = None
        train = None
        val = None
        names: Dict[int, str] = {}

        with data_yaml_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("path:"):
                    root = Path(line.split(":", 1)[1].strip())
                elif line.startswith("train:"):
                    train = line.split(":", 1)[1].strip()
                elif line.startswith("val:"):
                    val = line.split(":", 1)[1].strip()
                elif line.startswith("test:"):
                    continue
                elif ":" in line and line[0].isdigit():
                    k, v = line.split(":", 1)
                    names[int(k.strip())] = v.strip().strip("\"'")

        if root is None or train is None or val is None:
            raise ValueError(
                f"Could not parse required fields from {data_yaml_path}. "
                "Please install pyyaml or verify data.yaml format."
            )

        return root / train, root / val, names


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO validation labels into Faster R-CNN label files in-place"
    )
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()
    _, val_images, _ = parse_data_yaml(data_yaml_path)

    val_labels = val_images.parent / "labels"

    print(f"Converting validation labels in-place: {val_labels}")
    val_total, val_missing = convert_labels_for_images(val_images, val_labels, val_labels)

    print("\nConversion finished")
    print(f"Validation: converted {val_total} images (missing source labels: {val_missing})")


if __name__ == "__main__":
    main()