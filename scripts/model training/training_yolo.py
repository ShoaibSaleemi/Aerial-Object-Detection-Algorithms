from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    # 1. Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from YAML (from scratch, not pretrained)
    # model = YOLO("yolov8n.yaml").load("yolo8n.pt")  # build from YAML and transfer weights
    model_file = PROJECT_ROOT / "yolo26n.pt"
    model = YOLO(str(model_file))  # load a pretrained model

    # 2. Train the model, https://docs.ultralytics.com/modes/train/#musgd-optimizer
    results = model.train(
        data=str(PROJECT_ROOT / "data.yaml"),
        name=model_file.stem,  # save to runs/detect/yolo26n (not train/train2/...)
        epochs=50, # pass over the entire dataset, affect training duration and model performance
        imgsz=640, # input image size is 1024, but resized to 640 for training
        batch=16,
        workers=0, # avoid Windows shared-memory mapping failures in multi-worker dataloading
        save=True, # save the training checkpoints
        save_period=10, # frequency of saving a checkpoint, specified in epochs
        optimizer="SGD", # We are using Stochastic Gradient Descent (SGD) optimizer
        resume=False, # resume training from the latest saved checkpoint
    )
