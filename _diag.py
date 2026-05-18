import json
from pathlib import Path

cache = Path(r"C:\Users\shoai\project\thesis\runs\detect\weights\pred_cache_yolo12n_test2_imgsz640.json")
if not cache.exists():
    print("Cache not found:", cache)
else:
    data = json.load(cache.open())
    print(f"Images in cache: {len(data)}")
    total_gt = sum(len(d["gt_labels"]) for d in data)
    total_pred = sum(len(d["preds"]) for d in data)
    from collections import Counter
    gt_cls = Counter(c for d in data for c in d["gt_labels"])
    pred_cls = Counter(p[1] for d in data for p in d["preds"])
    print(f"Total GT boxes: {total_gt}  -> {dict(gt_cls)}")
    print(f"Total pred boxes (conf>0.001): {total_pred}  -> {dict(pred_cls)}")
    # sample confidence range
    all_confs = [p[0] for d in data for p in d["preds"]]
    if all_confs:
        print(f"Pred conf range: {min(all_confs):.4f} – {max(all_confs):.4f}")
        print(f"Preds with conf>0.3: {sum(1 for c in all_confs if c > 0.3)}")
