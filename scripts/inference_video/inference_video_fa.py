"""
inference_video_fa.py

Runs inference on every visible.mp4 + visible.json pair found in the
subfolders of dataset/test/videos/, treats all clips as one stitched
sequence and reports False Alarms per Hour for that stitched sequence.

A frame is a FALSE ALARM if it has at least one detection that is:
  - gt_exist == 0  (no drone present)  -> any detection
  - gt_exist == 1  (drone present)     -> detection classified as bird/unknown,
                                          OR drone detection with IoU < FA_IOU_THRESH
  - beyond GT data / no GT file        -> any detection

Outputs (saved to runs/detect/inference_video/):
  all_visible_<run>_eval.csv      -- per-subfolder rows + AGGREGATE row
  all_visible_<run>_eval.png      -- stitched success curve
  all_visible_<run>_perframe.png  -- stitched per-frame metrics (5 subplots)
  all_visible_<run>_perframe.npz  -- raw arrays
"""

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
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_ROOT  = PROJECT_ROOT / "dataset" / "test" / "videos"
CONF_THRESH = 0.7
IMG_SIZE    = 640
DEVICE      = ""   # e.g. "cpu" or "0"

# IoU threshold for matching a drone detection to the GT box.
# Drone detections below this are counted as false alarms.
FA_IOU_THRESH = 0.5


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


def find_visible_pairs() -> list[tuple[Path, Path]]:
    """Return sorted (visible.mp4, visible.json) pairs from all subfolders."""
    if not VIDEO_ROOT.exists():
        raise FileNotFoundError(f"Video root directory not found: {VIDEO_ROOT}")
    pairs: list[tuple[Path, Path]] = []
    for subdir in sorted(VIDEO_ROOT.iterdir()):
        if not subdir.is_dir():
            continue
        mp4 = subdir / "visible.mp4"
        js  = subdir / "visible.json"
        if mp4.exists() and js.exists():
            pairs.append((mp4, js))
        else:
            missing = [n for n, p in [("visible.mp4", mp4), ("visible.json", js)] if not p.exists()]
            print(f"  [skip] {subdir.name}: missing {', '.join(missing)}")
    return pairs


def load_gt_from_json(json_path: Path) -> tuple[list, list]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("exist", []), data.get("gt_rect", [])


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def _eiou(a, b) -> float:
    iou = _iou(a, b)
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    cx1 = min(a[0], b[0]); cy1 = min(a[1], b[1])
    cx2 = max(a[2], b[2]); cy2 = max(a[3], b[3])
    wc = cx2 - cx1; hc = cy2 - cy1
    center_p = ((ax - bx) ** 2 + (ay - by) ** 2) / (wc ** 2 + hc ** 2 + 1e-9)
    width_p  = (aw - bw) ** 2 / (wc ** 2 + 1e-9)
    height_p = (ah - bh) ** 2 / (hc ** 2 + 1e-9)
    return iou - center_p - width_p - height_p


def _compute_ap(preds: list, gt_count: int, iou_thresh: float) -> float:
    """AP with greedy matching (one TP per GT box per frame)."""
    if not preds or gt_count == 0:
        return 0.0
    sorted_preds = sorted(preds, key=lambda x: x[0], reverse=True)
    matched: set[int] = set()
    tp_list: list[int] = []
    for _conf, iou, fidx in sorted_preds:
        if iou >= iou_thresh and fidx not in matched:
            tp_list.append(1)
            matched.add(fidx)
        else:
            tp_list.append(0)
    cum_tp = np.cumsum(tp_list, dtype=np.float64)
    cum_fp = np.cumsum([1 - t for t in tp_list], dtype=np.float64)
    recall = cum_tp / gt_count
    precision_vals = cum_tp / (cum_tp + cum_fp + 1e-9)
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 101):
        mask = recall >= thr
        ap  += (precision_vals[mask].max() if mask.any() else 0.0)
    return ap / 101


def main():
    run_name = choose_run_folder()

    detect_run_dir = PROJECT_ROOT / "runs" / "detect" / run_name
    model_path = detect_run_dir / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_dir = PROJECT_ROOT / "runs" / "detect" / "inference_video"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_visible_pairs()
    if not pairs:
        raise ValueError(f"No visible.mp4 + visible.json pairs found under {VIDEO_ROOT}")

    print(f"\n{'Loaded model:':<24}{model_path}")
    print(f"{'Video pairs found:':<24}{len(pairs)}")
    print(f"{'Output directory:':<24}{out_dir}\n")

    model = YOLO(str(model_path))

    # ── Global accumulators ────────────────────────────────────────────────
    total_exist1_frames:  int   = 0
    total_covered_frames: int   = 0
    total_fa_frames:      int   = 0
    total_processed:      int   = 0
    total_duration_h:     float = 0.0

    per_video_rows: list[dict] = []

    prefix     = out_dir / f"all_visible_{run_name}"
    csv_path   = out_dir / f"{prefix.name}_eval.csv"
    fieldnames = [
        "subfolder", "n_frames", "duration_h", "exist1_frames",
        "covered_frames", "fa_frames", "fa_per_hour",
        "mean_iou", "sr50", "auc", "coverage", "ap50", "map50_95",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as _f:
        csv.DictWriter(_f, fieldnames=fieldnames).writeheader()

    loop_start_time = time.time()

    # ── Per-video loop ─────────────────────────────────────────────────────
    for video_idx, (video_path, json_path) in enumerate(pairs, 1):
        subfolder = video_path.parent.name

        gt_exist, gt_rect = load_gt_from_json(json_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("[ERROR] could not open video -- skipping")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Local accumulators for this video
        loc_frame_ious: list[float] = []
        loc_map_preds:  list[tuple] = []

        loc_exist1_frames:  int = 0
        loc_covered_frames: int = 0
        loc_fa_frames:      int = 0
        processed:          int = 0
        start_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            predict_kwargs = {
                "source":  frame,
                "conf":    CONF_THRESH,
                "imgsz":   IMG_SIZE,
                "batch":   1,
                "save":    False,
                "show":    False,
                "verbose": False,
            }
            if DEVICE:
                predict_kwargs["device"] = DEVICE

            result = model.predict(**predict_kwargs)
            r = result[0] if isinstance(result, list) else result

            has_boxes   = hasattr(r, "boxes") and len(r.boxes) > 0
            is_fa_frame = False

            if processed < len(gt_exist):
                if gt_exist[processed] == 1:
                    rx, ry, rw, rh = gt_rect[processed]
                    gt_xyxy = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
                    loc_exist1_frames += 1

                    best_iou = 0.0

                    if has_boxes:
                        loc_covered_frames += 1
                        for box, conf, cls_id in zip(
                            r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy(),
                        ):
                            iou_val = _iou(box.tolist(), gt_xyxy)
                            cls_int = int(cls_id) if int(cls_id) < 3 else 2

                            if iou_val > best_iou:
                                best_iou = iou_val
                            loc_map_preds.append((float(conf), iou_val, processed))

                            # FA: wrong class OR drone that doesn't overlap GT
                            if cls_int != 1 or iou_val < FA_IOU_THRESH:
                                is_fa_frame = True

                    loc_frame_ious.append(best_iou)

                elif gt_exist[processed] == 0:
                    # GT absent: any detection is a false alarm
                    if has_boxes:
                        is_fa_frame = True
            else:
                # Beyond GT data length: any detection is a false alarm
                if has_boxes:
                    is_fa_frame = True

            loc_fa_frames += int(is_fa_frame)
            processed += 1

            elapsed_v = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_v), 60)
            if total_frames_v > 0:
                pct = processed / total_frames_v * 100
                print(
                    f"Progress: {processed}/{total_frames_v} ({pct:.2f}%) Elapsed: {minutes}:{seconds:02d}",
                    end="\r", flush=True,
                )
            else:
                print(
                    f"Progress: {processed} frames Elapsed: {minutes}:{seconds:02d}",
                    end="\r", flush=True,
                )

        cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        video_duration_h = processed / fps / 3600.0
        video_fa_per_h   = loc_fa_frames / video_duration_h if video_duration_h > 0 else 0.0

        # Per-video evaluation metrics
        row: dict = {
            "subfolder":      subfolder,
            "n_frames":       processed,
            "duration_h":     video_duration_h,
            "exist1_frames":  loc_exist1_frames,
            "covered_frames": loc_covered_frames,
            "fa_frames":      loc_fa_frames,
            "fa_per_hour":    video_fa_per_h,
        }
        if loc_exist1_frames > 0:
            v_iou_arr = np.array(loc_frame_ious, dtype=np.float32)
            v_thr_iou = np.linspace(0.0, 1.0, 101)
            v_success = np.array([(v_iou_arr >= t).mean() for t in v_thr_iou], dtype=np.float32)
            row["mean_iou"] = float(v_iou_arr.mean())
            row["sr50"]     = float((v_iou_arr >= 0.5).mean())
            row["auc"]      = float(np.trapezoid(v_success, v_thr_iou))
            row["coverage"] = loc_covered_frames / loc_exist1_frames
            row["ap50"]     = _compute_ap(loc_map_preds, loc_exist1_frames, 0.50)
            row["map50_95"] = float(np.mean([
                _compute_ap(loc_map_preds, loc_exist1_frames, t)
                for t in np.arange(0.50, 1.00, 0.05)
            ]))
        else:
            for key in ("mean_iou", "sr50", "auc", "coverage", "ap50", "map50_95"):
                row[key] = ""
        per_video_rows.append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as _f:
            csv.DictWriter(_f, fieldnames=fieldnames).writerow(row)

        total_exist1_frames  += loc_exist1_frames
        total_covered_frames += loc_covered_frames
        total_fa_frames      += loc_fa_frames
        total_processed      += processed
        total_duration_h     += video_duration_h

        print(f"\r{' ' * 60}", end="\r")  # clear progress line
        v_min, v_sec = divmod(int(elapsed), 60)
        total_elapsed_s = int(time.time() - loop_start_time)
        total_minutes, total_seconds = divmod(total_elapsed_s, 60)
        w = len(str(len(pairs)))
        print(f"Video {video_idx:0{w}}/{len(pairs)}  FA={loc_fa_frames}  FA/h={video_fa_per_h:.1f}  Elapsed: {v_min}:{v_sec:02d}  Total: {total_minutes}:{total_seconds:02d}")

    # ── Aggregate evaluation ───────────────────────────────────────────────
    total_fa_per_h = total_fa_frames / total_duration_h if total_duration_h > 0 else 0.0

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  AGGREGATE  ({len(per_video_rows)} videos / {total_processed} frames / "
          f"{total_duration_h * 60:.1f} min)")
    print(sep)

    agg_auc = agg_sr50 = agg_mean_iou = agg_coverage = agg_ap50 = agg_map50_95 = 0.0

    if total_exist1_frames > 0:
        rows_w = [r for r in per_video_rows if r.get("mean_iou") != ""]
        if rows_w:
            w = np.array([r["exist1_frames"] for r in rows_w], dtype=np.float64)
            agg_mean_iou = float(np.average([r["mean_iou"] for r in rows_w], weights=w))
            agg_sr50     = float(np.average([r["sr50"]     for r in rows_w], weights=w))
            agg_auc      = float(np.average([r["auc"]      for r in rows_w], weights=w))
            agg_ap50     = float(np.average([r["ap50"]     for r in rows_w], weights=w))
            agg_map50_95 = float(np.average([r["map50_95"] for r in rows_w], weights=w))
        agg_coverage = total_covered_frames / total_exist1_frames

        print(f"  {'Mean IoU:':<38}{agg_mean_iou * 100:.2f}%")
        print(f"  {'Success Rate  @IoU>=0.5:':<38}{agg_sr50 * 100:.2f}%")
        print(f"  {'AUC  (success curve 0->1):':<38}{agg_auc * 100:.2f}%")
        print(f"  {'Coverage:':<38}{agg_coverage * 100:.2f}%  "
              f"({total_covered_frames}/{total_exist1_frames})")
        print(f"  {'False Alarms / hour (stitched):':<38}{total_fa_per_h:.2f}  "
              f"({total_fa_frames} FA frames)")
        print(f"  {'AP@50:':<38}{agg_ap50 * 100:.2f}%")
        print(f"  {'mAP@50:95:':<38}{agg_map50_95 * 100:.2f}%")
    else:
        print(f"  No GT present frames found across all videos.")
        print(f"  {'False Alarms / hour (stitched):':<38}{total_fa_per_h:.2f}  "
              f"({total_fa_frames} FA frames)")

    print(sep)

    # ── CSV: append aggregate row ──────────────────────────────────────────
    with open(csv_path, "a", newline="", encoding="utf-8") as _f:
        csv.DictWriter(_f, fieldnames=fieldnames).writerow({
            "subfolder":      "AGGREGATE",
            "n_frames":       total_processed,
            "duration_h":     total_duration_h,
            "exist1_frames":  total_exist1_frames,
            "covered_frames": total_covered_frames,
            "fa_frames":      total_fa_frames,
            "fa_per_hour":    total_fa_per_h,
            "mean_iou":       agg_mean_iou if total_exist1_frames > 0 else "",
            "sr50":           agg_sr50     if total_exist1_frames > 0 else "",
            "auc":            agg_auc      if total_exist1_frames > 0 else "",
            "coverage":       agg_coverage if total_exist1_frames > 0 else "",
            "ap50":           agg_ap50     if total_exist1_frames > 0 else "",
            "map50_95":       agg_map50_95 if total_exist1_frames > 0 else "",
        })
    print(f"  Eval csv       -> {csv_path}")


if __name__ == "__main__":
    main()
