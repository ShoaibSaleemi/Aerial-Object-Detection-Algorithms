from pathlib import Path
import csv
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import questionary
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["bird", "drone", "unknown"]
VIDEO_DIR = PROJECT_ROOT / "dataset" / "test" / "videos"
CONF_THRESH = 0.7
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


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_gt_labels(video_path: Path):
    """Load <stem>.json GT alongside the video if present."""
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get("exist", []), data.get("gt_rect", [])


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

    output_path = PROJECT_ROOT / "runs" / "detect" / "inference" / f"{video_path.stem}_{run_name}_inference.mp4"
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
    print(f"{'Saving output to:':<22}{output_path}")

    # Ground-truth labels (optional)
    gt_data   = load_gt_labels(video_path)
    has_gt    = gt_data is not None
    gt_exist: list = []
    gt_rect:  list = []
    if has_gt:
        gt_exist, gt_rect = gt_data
        print(f"{'GT labels:':<22}{video_path.stem}.json  ({len(gt_exist)} frames)")

    print()

    frame_ious:     list[float] = []
    frame_dists:    list[float] = []
    frame_confs:    list[float] = []
    frame_detected: list[int]   = []
    frame_numbers:  list[int]   = []
    covered_frames: int = 0
    exist1_frames:  int = 0

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

        # ── GT overlay + per-frame evaluation ──────────────────────────────
        if has_gt and processed < len(gt_exist):
            if gt_exist[processed] == 1:
                rx, ry, rw, rh = gt_rect[processed]
                gx1, gy1 = int(rx), int(ry)
                gx2, gy2 = int(rx + rw), int(ry + rh)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 255, 255), 2)
                cv2.putText(
                    frame, "GT", (gx1, max(12, gy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                )

                gt_xyxy = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
                gt_cx   = rx + rw / 2.0
                gt_cy   = ry + rh / 2.0
                exist1_frames += 1

                best_iou  = 0.0
                best_dist = float("nan")
                best_conf = 0.0
                detected  = 0
                if hasattr(r, "boxes") and len(r.boxes) > 0:
                    covered_frames += 1
                    detected = 1
                    for box, conf in zip(
                        r.boxes.xyxy.cpu().numpy(),
                        r.boxes.conf.cpu().numpy(),
                    ):
                        iou_val = _iou(box.tolist(), gt_xyxy)
                        if iou_val > best_iou:
                            best_iou  = iou_val
                            tcx = (box[0] + box[2]) / 2.0
                            tcy = (box[1] + box[3]) / 2.0
                            best_dist = float(np.hypot(tcx - gt_cx, tcy - gt_cy))
                        if float(conf) > best_conf:
                            best_conf = float(conf)

                frame_ious.append(best_iou)
                frame_dists.append(best_dist)
                frame_confs.append(best_conf)
                frame_detected.append(detected)
                frame_numbers.append(processed + 1)

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

    # ── Evaluation summary ──────────────────────────────────────────────────
    if has_gt and exist1_frames > 0:
        iou_arr  = np.array(frame_ious,  dtype=np.float32)
        dist_arr = np.array(frame_dists, dtype=np.float32)

        thr_iou  = np.linspace(0.0, 1.0, 101)
        success  = np.array([(iou_arr >= t).mean() for t in thr_iou], dtype=np.float32)
        auc      = float(np.trapezoid(success, thr_iou))
        sr50     = float((iou_arr >= 0.5).mean())

        thr_dist  = np.arange(0, 51, dtype=np.float32)
        precision = np.array([(dist_arr <= t).mean() for t in thr_dist], dtype=np.float32)
        prec20    = float((dist_arr <= 20.0).mean())
        coverage  = covered_frames / exist1_frames
        mean_iou  = float(iou_arr.mean())

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  Evaluation  ({exist1_frames} present frames / {processed} total)")
        print(sep)
        print(f"  {'Mean IoU:':<32}{mean_iou * 100:.2f}%")
        print(f"  {'Success Rate  @IoU≥0.5:':<32}{sr50 * 100:.2f}%")
        print(f"  {'AUC  (success curve 0→1):':<32}{auc * 100:.2f}%")
        print(f"  {'Precision  @20 px:':<32}{prec20 * 100:.2f}%")
        print(f"  {'Coverage:':<32}{coverage * 100:.2f}%  ({covered_frames}/{exist1_frames})")
        print(sep)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(thr_iou, success, linewidth=2)
        ax1.set_xlabel("IoU threshold")
        ax1.set_ylabel("Success rate")
        ax1.set_title(f"Success curve  AUC={auc:.3f}")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.4)

        ax2.plot(thr_dist, precision, linewidth=2)
        ax2.axvline(20, color="gray", linestyle="--", linewidth=1, label="20 px")
        ax2.set_xlabel("Centre distance threshold (px)")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision curve  @20px={prec20:.3f}")
        ax2.set_xlim(0, 50); ax2.set_ylim(0, 1)
        ax2.legend(); ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        plot_path = output_path.parent / f"{video_path.stem}_{run_name}_eval.png"
        fig.savefig(str(plot_path), dpi=120)
        plt.close(fig)
        print(f"\n  Eval plot → {plot_path}")

        # ── Per-frame metrics plot ──────────────────────────────────────────
        frames_x = np.array(frame_numbers, dtype=np.float32)
        metrics = [
            ("IoU",                     np.array(frame_ious,     dtype=np.float32), (0.0, 1.0)),
            ("Center distance (px)",    np.array(frame_dists,    dtype=np.float32), None),
            ("Detection confidence",    np.array(frame_confs,    dtype=np.float32), (0.0, 1.0)),
            ("Detected (0/1)",          np.array(frame_detected, dtype=np.float32), (-0.1, 1.1)),
        ]

        fig2, axes = plt.subplots(
            len(metrics), 1,
            figsize=(12, 3 * len(metrics)),
            sharex=True,
        )
        fig2.suptitle(f"{run_name}  —  per-frame metrics", fontsize=13, fontweight="bold")

        for ax, (label, values, ylim) in zip(axes, metrics):
            ax.plot(frames_x, values, linewidth=1.0)
            ax.set_ylabel(label, fontsize=10)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.35)

        axes[-1].set_xlabel("Frame", fontsize=10)
        fig2.tight_layout()
        perframe_plot_path = output_path.parent / f"{video_path.stem}_{run_name}_perframe.png"
        fig2.savefig(str(perframe_plot_path), dpi=120)
        plt.close(fig2)
        print(f"  Per-frame plot → {perframe_plot_path}")

        csv_path = output_path.parent / f"{video_path.stem}_{run_name}_eval.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value", "value_percent", "note"])
            writer.writerow(["mean_iou", mean_iou, mean_iou * 100.0, ""])
            writer.writerow(["success_rate_iou_ge_0_5", sr50, sr50 * 100.0, "IoU >= 0.5"])
            writer.writerow(["auc_success_curve_0_to_1", auc, auc * 100.0, ""])
            writer.writerow(["precision_at_20px", prec20, prec20 * 100.0, "center distance <= 20"])
            writer.writerow(["coverage", coverage, coverage * 100.0, f"{covered_frames}/{exist1_frames}"])
            writer.writerow(["covered_frames", covered_frames, "", ""])
            writer.writerow(["present_frames", exist1_frames, "", ""])
            writer.writerow(["processed_frames", processed, "", ""])
        print(f"  Eval csv  → {csv_path}")


if __name__ == "__main__":
    main()
