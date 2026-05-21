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
import gc
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

# â”€â”€ Model selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Set a model to True to include it, False to skip it.
# Set RUN_ALL_MODELS = True to override and run every model regardless.
RUN_ALL_MODELS = True
ENABLED_MODELS = {
    "yolo8n":     False,
    "yolo8m":     False,
    "yolo9t":     False,
    "yolo10n":    False,
    "yolo11n":    False,
    "yolo12n":    False,
    "yolo26n":    False,
    "fasterrcnn": False,
}

# Preferred evaluation order; models not listed are appended alphabetically.
MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n", "fasterrcnn"]

# CSV column names shared by per-model and combined output files.
FIELDNAMES = [
    "subfolder", "n_frames", "duration_h", "exist1_frames",
    "covered_frames", "fa_frames", "fa_per_hour",
    "mean_iou", "sr50", "auc", "coverage", "ap50", "map50_95",
]

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

    # Priority: ENABLED_MODELS constant â†’ command-line arg â†’ interactive prompt
    enabled = [name for name, on in ENABLED_MODELS.items() if on]
    if len(enabled) > 1:
        raise ValueError(f"Only one model may be enabled at a time, got: {enabled}")
    if len(enabled) == 1:
        run_name = enabled[0]
        if run_name not in available_runs:
            available_text = ", ".join(available_runs)
            raise ValueError(
                f"ENABLED_MODELS '{run_name}' not found in runs/detect. "
                f"Available: {available_text}"
            )
        return run_name

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


def _get_run_names() -> list[str]:
    """Resolve which model run folders to evaluate (respects RUN_ALL_MODELS / ENABLED_MODELS)."""
    detect_root = PROJECT_ROOT / "runs" / "detect"
    available_runs = sorted([p.name for p in detect_root.iterdir() if p.is_dir()])
    if not available_runs:
        raise ValueError(f"No folders found in {detect_root}")

    if RUN_ALL_MODELS:
        available_set = set(available_runs)
        ordered = [
            r for r in MODEL_ORDER
            if r in available_set and (detect_root / r / "weights" / "best.pt").exists()
        ]
        extras = sorted([
            r for r in available_runs
            if r not in MODEL_ORDER and (detect_root / r / "weights" / "best.pt").exists()
        ])
        run_names = ordered + extras
        if not run_names:
            raise ValueError(f"No model folders with best.pt found in {detect_root}")
        return run_names

    return [choose_run_folder()]


def run_model(run_name: str, pairs: list[tuple[Path, Path]], out_dir: Path) -> dict | None:
    """
    Run inference for one model over all video pairs.
    Loads the model once, processes every video sequentially, saves a per-model
    CSV, then unloads the model and frees GPU/CPU memory before returning.
    Returns an aggregate metrics dict, or None if best.pt is missing.
    """
    detect_run_dir = PROJECT_ROOT / "runs" / "detect" / run_name
    model_path = detect_run_dir / "weights" / "best.pt"
    if not model_path.exists():
        print(f"  [skip] {run_name}: best.pt not found at {model_path}")
        return None

    print(f"\n  {'Model path:':<22}{model_path}")
    print(f"  {'Video pairs:':<22}{len(pairs)}")
    print(f"  {'Output dir:':<22}{out_dir}\n")

    model = YOLO(str(model_path))

    # â”€â”€ Global accumulators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total_exist1_frames:  int   = 0
    total_covered_frames: int   = 0
    total_fa_frames:      int   = 0
    total_processed:      int   = 0
    total_duration_h:     float = 0.0
    per_video_rows: list[dict]  = []

    csv_path = out_dir / f"all_visible_{run_name}_eval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as _f:
        csv.DictWriter(_f, fieldnames=FIELDNAMES).writeheader()

    loop_start_time = time.time()

    # â”€â”€ Per-video loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for video_idx, (video_path, json_path) in enumerate(pairs, 1):
        subfolder = video_path.parent.name
        gt_exist, gt_rect = load_gt_from_json(json_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [ERROR] Could not open {video_path.name} â€” skipping")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

            # Move tensors to CPU numpy arrays before freeing the YOLO result.
            has_boxes = hasattr(r, "boxes") and len(r.boxes) > 0
            if has_boxes:
                boxes_np  = r.boxes.xyxy.cpu().numpy()
                confs_np  = r.boxes.conf.cpu().numpy()
                clsids_np = r.boxes.cls.cpu().numpy()
            else:
                boxes_np = confs_np = clsids_np = None
            del r, result  # release GPU tensors immediately

            is_fa_frame = False

            if processed < len(gt_exist):
                if gt_exist[processed] == 1:
                    rx, ry, rw, rh = gt_rect[processed]
                    gt_xyxy = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
                    loc_exist1_frames += 1
                    best_iou = 0.0

                    if has_boxes:
                        loc_covered_frames += 1
                        for box, conf, cls_id in zip(boxes_np, confs_np, clsids_np):
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

            del boxes_np, confs_np, clsids_np  # drop numpy refs

            loc_fa_frames += int(is_fa_frame)
            processed += 1

            elapsed_v = time.time() - start_time
            m, s = divmod(int(elapsed_v), 60)
            if total_frames_v > 0:
                pct = processed / total_frames_v * 100
                print(
                    f"  [{run_name}] {subfolder}  {processed}/{total_frames_v} ({pct:.1f}%)  {m}:{s:02d}",
                    end="\r", flush=True,
                )
            else:
                print(
                    f"  [{run_name}] {subfolder}  {processed} frames  {m}:{s:02d}",
                    end="\r", flush=True,
                )

        cap.release()
        # Free any cached GPU/CPU state between videos
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        video_duration_h = processed / fps / 3600.0
        video_fa_per_h   = loc_fa_frames / video_duration_h if video_duration_h > 0 else 0.0

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
            del v_iou_arr, v_thr_iou, v_success
        else:
            for key in ("mean_iou", "sr50", "auc", "coverage", "ap50", "map50_95"):
                row[key] = ""

        loc_frame_ious.clear()  # discard per-video arrays immediately
        loc_map_preds.clear()

        per_video_rows.append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as _f:
            csv.DictWriter(_f, fieldnames=FIELDNAMES).writerow(row)

        total_exist1_frames  += loc_exist1_frames
        total_covered_frames += loc_covered_frames
        total_fa_frames      += loc_fa_frames
        total_processed      += processed
        total_duration_h     += video_duration_h

        print(f"\r{' ' * 80}", end="\r")
        v_min, v_sec = divmod(int(elapsed), 60)
        total_elapsed_s = int(time.time() - loop_start_time)
        total_minutes, total_seconds = divmod(total_elapsed_s, 60)
        w = len(str(len(pairs)))
        print(f"  [{run_name}] Video {video_idx:0{w}}/{len(pairs)} {subfolder}  FA={loc_fa_frames}  FA/h={video_fa_per_h:.1f}  {v_min}:{v_sec:02d}  Total: {total_minutes}:{total_seconds:02d}")

    # â”€â”€ Unload model and free all GPU/CPU memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # â”€â”€ Aggregate evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total_fa_per_h = total_fa_frames / total_duration_h if total_duration_h > 0 else 0.0

    sep = "â”€" * 60
    print(f"\n{sep}")
    print(f"  [{run_name}] AGGREGATE  ({len(per_video_rows)} videos / {total_processed} frames / "
          f"{total_duration_h * 60:.1f} min)")
    print(sep)

    agg: dict = {
        "run_name":       run_name,
        "n_videos":       len(per_video_rows),
        "total_frames":   total_processed,
        "duration_h":     total_duration_h,
        "fa_frames":      total_fa_frames,
        "fa_per_hour":    total_fa_per_h,
        "exist1_frames":  total_exist1_frames,
        "covered_frames": total_covered_frames,
        "mean_iou":  None,
        "sr50":      None,
        "auc":       None,
        "coverage":  None,
        "ap50":      None,
        "map50_95":  None,
    }

    if total_exist1_frames > 0:
        rows_w = [r for r in per_video_rows if r.get("mean_iou") != ""]
        if rows_w:
            w = np.array([r["exist1_frames"] for r in rows_w], dtype=np.float64)
            agg["mean_iou"]  = float(np.average([r["mean_iou"]  for r in rows_w], weights=w))
            agg["sr50"]      = float(np.average([r["sr50"]      for r in rows_w], weights=w))
            agg["auc"]       = float(np.average([r["auc"]       for r in rows_w], weights=w))
            agg["ap50"]      = float(np.average([r["ap50"]      for r in rows_w], weights=w))
            agg["map50_95"]  = float(np.average([r["map50_95"]  for r in rows_w], weights=w))
        agg["coverage"] = total_covered_frames / total_exist1_frames

        print(f"  {'Mean IoU:':<38}{agg['mean_iou'] * 100:.2f}%")
        print(f"  {'Success Rate  @IoU>=0.5:':<38}{agg['sr50'] * 100:.2f}%")
        print(f"  {'AUC  (success curve 0->1):':<38}{agg['auc'] * 100:.2f}%")
        print(f"  {'Coverage:':<38}{agg['coverage'] * 100:.2f}%  "
              f"({total_covered_frames}/{total_exist1_frames})")
        print(f"  {'False Alarms / hour (stitched):':<38}{total_fa_per_h:.2f}  "
              f"({total_fa_frames} FA frames)")
        print(f"  {'AP@50:':<38}{agg['ap50'] * 100:.2f}%")
        print(f"  {'mAP@50:95:':<38}{agg['map50_95'] * 100:.2f}%")
    else:
        print(f"  No GT present frames found across all videos.")
        print(f"  {'False Alarms / hour (stitched):':<38}{total_fa_per_h:.2f}  "
              f"({total_fa_frames} FA frames)")

    print(sep)

    # Append aggregate row to the per-model CSV
    with open(csv_path, "a", newline="", encoding="utf-8") as _f:
        csv.DictWriter(_f, fieldnames=FIELDNAMES).writerow({
            "subfolder":      "AGGREGATE",
            "n_frames":       total_processed,
            "duration_h":     total_duration_h,
            "exist1_frames":  total_exist1_frames,
            "covered_frames": total_covered_frames,
            "fa_frames":      total_fa_frames,
            "fa_per_hour":    total_fa_per_h,
            "mean_iou":  agg["mean_iou"]  if agg["mean_iou"]  is not None else "",
            "sr50":      agg["sr50"]      if agg["sr50"]      is not None else "",
            "auc":       agg["auc"]       if agg["auc"]       is not None else "",
            "coverage":  agg["coverage"]  if agg["coverage"]  is not None else "",
            "ap50":      agg["ap50"]      if agg["ap50"]      is not None else "",
            "map50_95":  agg["map50_95"]  if agg["map50_95"]  is not None else "",
        })
    print(f"  Eval csv       -> {csv_path}")

    return agg


def _save_combined_csv(all_aggs: list[dict], out_dir: Path) -> Path:
    """Write aggregate results for all models into a single combined CSV."""
    combined_csv = out_dir / "all_models_fa_eval.csv"
    combined_fieldnames = [
        "model", "n_videos", "total_frames", "duration_h",
        "fa_frames", "fa_per_hour",
        "mean_iou", "sr50", "auc", "coverage", "ap50", "map50_95",
    ]
    with open(combined_csv, "w", newline="", encoding="utf-8") as _f:
        writer = csv.DictWriter(_f, fieldnames=combined_fieldnames)
        writer.writeheader()
        for agg in all_aggs:
            writer.writerow({
                "model":        agg["run_name"],
                "n_videos":     agg["n_videos"],
                "total_frames": agg["total_frames"],
                "duration_h":   agg["duration_h"],
                "fa_frames":    agg["fa_frames"],
                "fa_per_hour":  agg["fa_per_hour"],
                "mean_iou":  agg["mean_iou"]  if agg["mean_iou"]  is not None else "",
                "sr50":      agg["sr50"]      if agg["sr50"]      is not None else "",
                "auc":       agg["auc"]       if agg["auc"]       is not None else "",
                "coverage":  agg["coverage"]  if agg["coverage"]  is not None else "",
                "ap50":      agg["ap50"]      if agg["ap50"]      is not None else "",
                "map50_95":  agg["map50_95"]  if agg["map50_95"]  is not None else "",
            })
    return combined_csv


def _print_comparison_table(all_aggs: list[dict]) -> None:
    """Print a formatted side-by-side comparison of all model aggregate results."""

    def _pct(v) -> str:
        return f"{v * 100:.1f}%" if isinstance(v, float) else "â€”"

    def _f2(v) -> str:
        return f"{v:.2f}" if isinstance(v, float) else "â€”"

    col_w = max((len(a["run_name"]) for a in all_aggs), default=5)
    col_w = max(col_w, 5)

    header = (
        f"{'Model':<{col_w}}  "
        f"{'FA/h':>7}  {'FA frm':>6}  "
        f"{'IoU':>6}  {'SR50':>6}  {'AUC':>6}  "
        f"{'Cov':>6}  {'AP50':>6}  {'mAP':>6}  {'Dur(h)':>7}"
    )
    sep_line = "â”€" * len(header)
    print(f"\n{sep_line}")
    print("  ALL-MODEL COMPARISON")
    print(sep_line)
    print(f"  {header}")
    print(f"  {sep_line}")
    for agg in all_aggs:
        print(
            f"  {agg['run_name']:<{col_w}}  "
            f"{_f2(agg['fa_per_hour']):>7}  {agg['fa_frames']:>6}  "
            f"{_pct(agg['mean_iou']):>6}  {_pct(agg['sr50']):>6}  {_pct(agg['auc']):>6}  "
            f"{_pct(agg['coverage']):>6}  {_pct(agg['ap50']):>6}  {_pct(agg['map50_95']):>6}  "
            f"{agg['duration_h']:>7.3f}"
        )
    print(f"  {sep_line}")


def main() -> None:
    out_dir = PROJECT_ROOT / "runs" / "detect" / "inference_video"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_names = _get_run_names()

    pairs = find_visible_pairs()
    if not pairs:
        raise ValueError(f"No visible.mp4 + visible.json pairs found under {VIDEO_ROOT}")

    print(f"Video pairs:  {len(pairs)}")
    print(f"Models:       {', '.join(run_names)}\n")

    all_aggs: list[dict] = []

    for i, run_name in enumerate(run_names, 1):
        if len(run_names) > 1:
            print(f"\n{'=' * 60}")
            print(f"  Model {i}/{len(run_names)}: {run_name}")
            print(f"{'=' * 60}")
        agg = run_model(run_name, pairs, out_dir)
        if agg is not None:
            all_aggs.append(agg)

    if len(all_aggs) > 1:
        _print_comparison_table(all_aggs)
        combined_csv = _save_combined_csv(all_aggs, out_dir)
        print(f"\n  Combined CSV   -> {combined_csv}")


if __name__ == "__main__":
    main()
