import argparse
from pathlib import Path
from typing import Dict, Tuple

from label_conversion_fasterrcnn import convert_labels_for_images


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
        description="Convert YOLO labels into Faster R-CNN label files for train/validation"
    )
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--dst-name", type=str, default="labels_fasterrcnn", help="Output labels folder name under each split")
    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()
    train_images, val_images, _ = parse_data_yaml(data_yaml_path)

    train_src = train_images.parent / "labels"
    val_src = val_images.parent / "labels"

    train_dst = train_images.parent / args.dst_name
    val_dst = val_images.parent / args.dst_name

    print(f"Converting train labels: {train_src} -> {train_dst}")
    train_total, train_missing = convert_labels_for_images(train_images, train_src, train_dst)

    print(f"Converting validation labels: {val_src} -> {val_dst}")
    val_total, val_missing = convert_labels_for_images(val_images, val_src, val_dst)

    print("\nConversion finished")
    print(f"Train: converted {train_total} images (missing source labels: {train_missing})")
    print(f"Validation: converted {val_total} images (missing source labels: {val_missing})")


if __name__ == "__main__":
    main()
