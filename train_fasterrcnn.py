import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from label_conversion_fasterrcnn import load_fasterrcnn_label_file


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_data_yaml(data_yaml_path: Path) -> Tuple[Path, Path, Dict[int, str]]:
    """Parse the minimal fields needed from a YOLO data.yaml file."""
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


class FasterRCNNTxtDetectionDataset(Dataset):
    """Dataset that reads precomputed Faster R-CNN txt labels."""

    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        )
        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _label_path_for_image(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def __getitem__(self, idx: int):
        image_path = self.image_files[idx]
        label_path = self._label_path_for_image(image_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = torch.from_numpy(np.array(image, dtype="uint8")).permute(2, 0, 1).float() / 255.0

        boxes, labels = load_fasterrcnn_label_file(label_path)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
            iscrowd = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def infer_max_class_id(labels_dir: Path) -> int:
    max_class = -1
    if not labels_dir.exists():
        return max_class

    for txt_path in labels_dir.glob("*.txt"):
        if txt_path.stat().st_size == 0:
            continue
        with txt_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                parts = raw_line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id > max_class:
                    max_class = cls_id
    return max_class


def build_model(num_classes: int, use_pretrained: bool = True):
    model = None

    if use_pretrained:
        try:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception as e:
            print(f"Could not load pretrained weights ({e}). Falling back to random init.")

    if model is None:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0.0

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_value = total_loss.item()
        epoch_loss += loss_value

        if step % 20 == 0 or step == len(data_loader):
            loss_items = {k: float(v.item()) for k, v in loss_dict.items()}
            print(
                f"Epoch {epoch} | Step {step}/{len(data_loader)} | "
                f"Loss: {loss_value:.4f} | Parts: {loss_items}"
            )

    return epoch_loss / max(len(data_loader), 1)


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-16)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def evaluate_detection_metrics(model, data_loader, device, num_classes):
    """
    Return metrics close to Ultralytics detect metrics:
    precision, recall, mAP50, mAP50-95, and fitness = 0.1*mAP50 + 0.9*mAP50-95.
    """
    model.eval()

    iouv = np.arange(0.5, 0.96, 0.05)
    niou = len(iouv)

    class_stats = {
        c: {
            "scores": [],
            "tp": [],
            "n_gt": 0,
        }
        for c in range(1, num_classes)
    }

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)

                pred_boxes = output["boxes"].to(device)
                pred_labels = output["labels"].to(device)
                pred_scores = output["scores"].to(device)

                for c in range(1, num_classes):
                    gt_mask = gt_labels == c
                    pb_mask = pred_labels == c

                    gt_c = gt_boxes[gt_mask]
                    pb_c = pred_boxes[pb_mask]
                    ps_c = pred_scores[pb_mask]

                    class_stats[c]["n_gt"] += int(gt_c.shape[0])

                    if pb_c.shape[0] == 0:
                        continue

                    order = torch.argsort(ps_c, descending=True)
                    pb_c = pb_c[order]
                    ps_c = ps_c[order]

                    tp = torch.zeros((pb_c.shape[0], niou), dtype=torch.bool, device=device)

                    if gt_c.shape[0] > 0:
                        iou = box_iou_matrix(pb_c, gt_c)
                        for iou_idx, thr in enumerate(iouv):
                            matched_gt = set()
                            for p_idx in range(pb_c.shape[0]):
                                best_iou, best_gt = torch.max(iou[p_idx], dim=0)
                                gt_j = int(best_gt.item())
                                if best_iou.item() >= float(thr) and gt_j not in matched_gt:
                                    tp[p_idx, iou_idx] = True
                                    matched_gt.add(gt_j)

                    class_stats[c]["scores"].append(ps_c.detach().cpu().numpy())
                    class_stats[c]["tp"].append(tp.detach().cpu().numpy().astype(np.float32))

    ap_all = np.zeros((max(num_classes - 1, 1), niou), dtype=np.float32)
    precision_all = np.zeros((max(num_classes - 1, 1),), dtype=np.float32)
    recall_all = np.zeros((max(num_classes - 1, 1),), dtype=np.float32)

    valid_cls_idx = 0
    for c in range(1, num_classes):
        scores_list = class_stats[c]["scores"]
        tp_list = class_stats[c]["tp"]
        n_gt = class_stats[c]["n_gt"]

        if len(scores_list) == 0:
            valid_cls_idx += 1
            continue

        scores = np.concatenate(scores_list, axis=0)
        tp = np.concatenate(tp_list, axis=0)

        sort_idx = np.argsort(-scores)
        scores = scores[sort_idx]
        tp = tp[sort_idx]

        fp = 1.0 - tp

        for iou_idx in range(niou):
            tp_cum = np.cumsum(tp[:, iou_idx])
            fp_cum = np.cumsum(fp[:, iou_idx])

            recall_curve = tp_cum / max(n_gt, 1)
            precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
            ap_all[valid_cls_idx, iou_idx] = compute_ap(recall_curve, precision_curve)

            if iou_idx == 0:
                precision_all[valid_cls_idx] = float(precision_curve[-1]) if precision_curve.size else 0.0
                recall_all[valid_cls_idx] = float(recall_curve[-1]) if recall_curve.size else 0.0

        valid_cls_idx += 1

    mp = float(np.mean(precision_all)) if precision_all.size else 0.0
    mr = float(np.mean(recall_all)) if recall_all.size else 0.0
    map50 = float(np.mean(ap_all[:, 0])) if ap_all.size else 0.0
    map5095 = float(np.mean(ap_all)) if ap_all.size else 0.0
    fitness = 0.1 * map50 + 0.9 * map5095

    return {
        "precision": mp,
        "recall": mr,
        "map50": map50,
        "map50_95": map5095,
        "fitness": float(fitness),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train torchvision Faster R-CNN directly from YOLO txt labels"
    )
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--output", type=str, default="runs/fasterrcnn/train")
    parser.add_argument(
        "--labels-dir-name",
        type=str,
        default="labels_fasterrcnn",
        help="Label folder name under each split (e.g., labels_fasterrcnn)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume from output/last.pt if it exists",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()
    train_images, val_images, names = parse_data_yaml(data_yaml_path)

    train_labels_dir = train_images.parent / args.labels_dir_name
    val_labels_dir = val_images.parent / args.labels_dir_name

    train_ds = FasterRCNNTxtDetectionDataset(train_images, train_labels_dir)

    train_max_class = infer_max_class_id(train_ds.labels_dir)
    val_max_class = infer_max_class_id(val_labels_dir)
    max_class_seen = max(train_max_class, val_max_class)

    num_named_classes = len(names)
    num_dataset_classes = max_class_seen + 1 if max_class_seen >= 0 else 0
    num_foreground_classes = max(num_named_classes, num_dataset_classes)
    num_classes = num_foreground_classes + 1

    if max_class_seen >= num_named_classes:
        print(
            "Warning: dataset contains class ids not present in data.yaml names. "
            f"Max class id seen: {max_class_seen}, names count: {num_named_classes}. "
            "Model head size was expanded to include all seen ids."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train images: {train_images}")
    print(f"Val images:   {val_images}")
    print(f"Train labels: {train_labels_dir}")
    print(f"Val labels:   {val_labels_dir}")
    print(f"Foreground classes: {num_foreground_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    val_ds = FasterRCNNTxtDetectionDataset(val_images, val_labels_dir)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=num_classes, use_pretrained=not args.no_pretrained)
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_fitness = -1.0
    last_ckpt_path = output_dir / "last.pt"

    if last_ckpt_path.exists() and not args.no_resume:
        print(f"Resuming from checkpoint: {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_fitness = float(checkpoint.get("best_fitness", best_fitness))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

        print(
            f"Resume state | next epoch: {start_epoch}, "
            f"best fitness: {best_fitness:.4f}"
        )

        if start_epoch > args.epochs:
            print(
                f"Checkpoint is already at epoch {start_epoch - 1}, "
                f"which is >= requested epochs ({args.epochs}). Nothing to do."
            )
            return

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        print(f"Epoch {epoch} done. Average loss: {avg_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_fitness": best_fitness,
        }

        val_metrics = evaluate_detection_metrics(model, val_loader, device, num_classes)
        checkpoint["val_metrics"] = val_metrics

        print(
            "Validation | "
            f"P: {val_metrics['precision']:.4f} "
            f"R: {val_metrics['recall']:.4f} "
            f"mAP50: {val_metrics['map50']:.4f} "
            f"mAP50-95: {val_metrics['map50_95']:.4f} "
            f"fitness: {val_metrics['fitness']:.4f}"
        )

        if val_metrics["fitness"] > best_fitness:
            best_fitness = val_metrics["fitness"]
            checkpoint["best_fitness"] = best_fitness
            best_ckpt_path = output_dir / "best.pt"
            torch.save(checkpoint, best_ckpt_path)
            print(f"Saved best checkpoint: {best_ckpt_path}")

        torch.save(checkpoint, last_ckpt_path)

        if epoch % args.save_period == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"fasterrcnn_epoch_{epoch}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    # Avoid extra OpenMP threads competing with DataLoader workers on Windows.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
