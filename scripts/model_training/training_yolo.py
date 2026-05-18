from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def unique_run_name(base: str, runs_dir: Path) -> str:
    """Return base if runs_dir/base doesn't exist, otherwise base 2, base 3, …"""
    candidate = base
    n = 2
    while (runs_dir / candidate).exists():
        candidate = f"{base} {n}"
        n += 1
    return candidate


if __name__ == "__main__":
    # 1. Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from YAML (from scratch, not pretrained)
    # model = YOLO("yolov8n.yaml").load("yolo8n.pt")  # build from YAML and transfer weights
    model = YOLO(str(PROJECT_ROOT / "yolov10n.pt"))  # pretrained yolo10 nano

    run_name = unique_run_name("yolo10n", PROJECT_ROOT / "runs" / "detect")

    # 2. Train the model, https://docs.ultralytics.com/modes/train/#musgd-optimizer
    results = model.train(
        data=str(PROJECT_ROOT / "data.yaml"),
        name=run_name,  # save to runs/detect/yolo8n (not train/train2/...)
        epochs=50, # pass over the entire dataset, affect training duration and model performance
        imgsz=640, # input image size is 1024, but resized to 640 for training
        batch=8,
        workers=2, # avoid Windows shared-memory mapping failures in multi-worker dataloading
        save=True, # save the training checkpoints
        save_period=1, # frequency of saving a checkpoint, specified in epochs
        optimizer="SGD",
        lr0=0.01, # default SGD learning rate
        resume=False, # fresh start — never resume from a previous run's checkpoint
    )
