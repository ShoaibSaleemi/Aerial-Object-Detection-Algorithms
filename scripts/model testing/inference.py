from pathlib import Path
from ultralytics import YOLO
import cv2
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ["bird", "drone", "unknown"]

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train9" / "weights" / "best.pt"
IMAGE_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "train9" / "inference_results"
CONF_THRESH = 0.70
IMG_SIZE = 640

COLORS = {
    "bird": (0, 255, 0),      # green
    "drone": (0, 0, 255),     # red
    "unknown": (0, 165, 255), # orange
}

model = YOLO(str(MODEL_PATH))

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loaded model: {MODEL_PATH}")
print(f"Found {len(image_paths)} images in {IMAGE_DIR}")
print(f"Saving results to {OUTPUT_DIR}\n")

total_files = len(image_paths)
processed = 0
start_time = time.time()

for img_path in image_paths:
    result = model.predict(
        source=str(img_path),
        device=0,
        half=True,
        conf=CONF_THRESH,
        imgsz=IMG_SIZE,
        batch=1,
        save=False,
        show=False,
        verbose=False,
    )
    r = result[0] if isinstance(result, list) else result

    boxes = r.boxes.xyxy.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    img = cv2.imread(str(img_path))

    for box, cls_id, conf in zip(boxes, class_ids, confs):
        label = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "unknown"
        color = COLORS.get(label, (255, 255, 255))
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    save_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(save_path), img)

    # Update progress
    processed += 1
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"Progress: {processed}/{total_files} ({processed / total_files * 100:.2f}%) Elapsed: {minutes}:{seconds:02d}", end='\r')

print(f"\nDone. Saved {len(image_paths)} images to {OUTPUT_DIR}")
