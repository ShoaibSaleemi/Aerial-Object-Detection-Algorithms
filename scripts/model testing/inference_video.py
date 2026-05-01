from pathlib import Path
import sys
import time

import cv2
import questionary
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["bird", "drone", "unknown"]
VIDEO_DIR = PROJECT_ROOT / "dataset" / "test" / "videos"
CONF_THRESH = 0.25
IMG_SIZE = 640
DEVICE = ""  # Set to "cpu" or "0" if you want to force a device.

COLORS = {
    "bird": (0, 255, 0),      # green
    "drone": (0, 0, 255),     # red
    "unknown": (0, 165, 255), # orange
}


def choose_run_folder() -> str:
    detect_root_dir = PROJECT_ROOT / "runs" / "detect"
    available_runs = sorted([path.name for path in detect_root_dir.iterdir() if path.is_dir()])

    if len(available_runs) == 0:
        raise ValueError(f"No folders found in {detect_root_dir}")

    if len(sys.argv) > 1:
        run_name = sys.argv[1]
        if run_name not in available_runs:
            available_text = ", ".join(available_runs)
            raise ValueError(
                f"Unknown folder '{run_name}'. Choose one from runs/detect: {available_text}"
            )
        return run_name

    run_name = questionary.select(
        "Choose a folder from runs/detect:",
        choices=available_runs,
    ).ask()
    if not run_name:
        raise ValueError("No folder selected from runs/detect")

    return run_name


def choose_video_file() -> Path:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
    if len(video_paths) == 0:
        raise ValueError(f"No .mp4 video files found in {VIDEO_DIR}")

    if len(sys.argv) > 2:
        video_name = sys.argv[2]
        if not video_name.lower().endswith(".mp4"):
            video_name = f"{video_name}.mp4"
        video_path = VIDEO_DIR / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path

    if len(video_paths) == 1:
        return video_paths[0]

    video_name = questionary.select(
        "Choose a video from dataset/test/videos:",
        choices=[p.name for p in video_paths],
    ).ask()
    if not video_name:
        raise ValueError("No video selected from dataset/test/videos")

    return VIDEO_DIR / video_name


def draw_detections(frame, result):
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, class_ids, confs):
        cls_raw = int(cls_id)
        if cls_raw == 0:
            cls_int = 0
        elif cls_raw == 1:
            cls_int = 1
        else:
            cls_int = 2
        label = CLASS_NAMES[cls_int] if cls_int < len(CLASS_NAMES) else "unknown"
        color = COLORS.get(label, (255, 255, 255))

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_bg_y1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, text_bg_y1), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    run_name = choose_run_folder()
    video_path = choose_video_file()

    detect_run_dir = PROJECT_ROOT / "runs" / "detect" / run_name
    model_path = detect_run_dir / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_path = detect_run_dir / f"{video_path.stem}_inference.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(f"{'Loaded model:':<22}{model_path}")
    print(f"{'Input video:':<22}{video_path}")
    print(f"{'Saving output to:':<22}{output_path}\n")

    processed = 0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        predict_kwargs = {
            "source": frame,
            "conf": CONF_THRESH,
            "imgsz": IMG_SIZE,
            "batch": 1,
            "save": False,
            "show": False,
            "verbose": False,
        }
        if DEVICE:
            predict_kwargs["device"] = DEVICE

        result = model.predict(**predict_kwargs)
        r = result[0] if isinstance(result, list) else result

        draw_detections(frame, r)
        writer.write(frame)

        processed += 1
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)

        if total_frames > 0:
            pct = processed / total_frames * 100
            print(
                f"Progress: {processed}/{total_frames} ({pct:.2f}%) Elapsed: {minutes}:{seconds:02d}",
                end="\r",
            )
        else:
            print(
                f"Progress: {processed} frames Elapsed: {minutes}:{seconds:02d}",
                end="\r",
            )

    cap.release()
    writer.release()

    print()
    if total_frames > 0:
        print(f"Done. Processed {processed}/{total_frames} frames.")
    else:
        print(f"Done. Processed {processed} frames.")


if __name__ == "__main__":
    main()
