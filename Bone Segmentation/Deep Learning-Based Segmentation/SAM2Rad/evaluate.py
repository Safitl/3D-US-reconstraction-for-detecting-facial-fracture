"""
SAM2Rad evaluation script — computes the same metrics as the U-Net baseline
(Dice, IoU, Precision, Recall, Hausdorff) on the Test/ split (Patient 2).

Usage (from SAM2Rad/ folder):
    python evaluate.py --config sam2rad/configs/bone_seg.yaml \
                       --checkpoint checkpoints/model_epoch=59-val_dice=0.69.ckpt \
                       --out_dir ../../runs/sam2rad_bone_seg_eval
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from sam2rad import DATASETS, DotDict, build_sam2rad, build_samrad, convert_to_semantic


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(config):
    if "sam2" in config.image_encoder:
        return build_sam2rad(config)
    return build_samrad(config)


class SegmentationModel(torch.nn.Module):
    def __init__(self, config, prompts: Dict[str, torch.nn.Parameter]):
        super().__init__()
        self.model = build_model(config)
        self.dataset_names = list(prompts.keys())
        self.num_classes = list(prompts.values())[0].shape[0]
        self.learnable_prompts = torch.nn.ParameterDict(prompts)
        # Use learned prompts only at inference
        self.model.prompt_sampler.p[0] = 1.0
        self.model.prompt_sampler.p[1] = 0.0
        self.model.prompt_sampler.p[2] = 0.0
        self.model.prompt_sampler.p[3] = 0.0

    def forward(self, batch, dataset_index=0):
        prompts = self.learnable_prompts[self.dataset_names[dataset_index]]
        return self.model(batch, prompts, inference=True)

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Metrics (identical implementations to U-Net train.py)
# ---------------------------------------------------------------------------

def compute_metrics(pred_bin: np.ndarray, target_bin: np.ndarray):
    """Compute Dice, IoU, Precision, Recall on flat binary arrays."""
    pred   = pred_bin.astype(np.float32).ravel()
    target = target_bin.astype(np.float32).ravel()
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    eps = 1e-6
    return {
        "dice":      (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou":       (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall":    (tp + eps) / (tp + fn + eps),
    }


def hausdorff_distance(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    p = np.argwhere(pred_bin)
    t = np.argwhere(target_bin)
    if len(p) == 0 or len(t) == 0:
        return float("nan")
    return max(directed_hausdorff(p, t)[0], directed_hausdorff(t, p)[0])


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

_PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_PIXEL_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(images: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 RGB (B, H, W, 3)."""
    imgs = (images.cpu() * _PIXEL_STD + _PIXEL_MEAN).clamp(0, 1)
    return (imgs.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def evaluate(model, dataloader, out_dir: Path):
    model.eval()
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # BoneSegDataset returns a dict
        images    = batch["images"].to(model.device)   # (B, 3, H, W)
        gt        = batch["masks"].to(model.device)    # (B, C, H, W)
        boxes     = batch["boxes"].to(model.device).view(-1, 4)
        filenames = batch["filename"]

        _, num_classes, h, w = gt.shape

        outputs = model({"images": images, "masks": gt, "boxes": boxes})
        pred    = outputs["pred"]                      # (B*C, 1, H, W)
        pred    = convert_to_semantic(pred.view(-1, num_classes, h, w))
        gt_sem  = convert_to_semantic(gt)

        imgs_rgb = denormalize(images)

        # Per-sample metrics
        for i in range(pred.shape[0]):
            pred_np = pred[i].cpu().numpy().astype(np.uint8)
            gt_np   = gt_sem[i].cpu().numpy().astype(np.uint8)
            fname   = filenames[i] if isinstance(filenames[i], str) else filenames[i][0]

            m = compute_metrics(pred_np, gt_np)
            m["hausdorff"] = hausdorff_distance(pred_np, gt_np)
            m["filename"]  = Path(fname).name
            results.append(m)

            # Save overlay: image | ground truth | prediction
            img_bgr  = cv2.cvtColor(imgs_rgb[i], cv2.COLOR_RGB2BGR)
            gt_bgr   = cv2.cvtColor((gt_np   * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            pred_bgr = cv2.cvtColor((pred_np  * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            panel = np.concatenate([img_bgr, gt_bgr, pred_bgr], axis=1)
            cv2.imwrite(str(overlay_dir / f"{Path(fname).stem}_overlay.png"), panel)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,
                        help="Path to bone_seg.yaml config.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .ckpt file (e.g. checkpoints/model_epoch=59-...).")
    parser.add_argument("--out_dir",    default="../../runs/sam2rad_bone_seg_eval",
                        help="Directory to save metrics CSV and overlays.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = DotDict(yaml.safe_load(f))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset_obj = DATASETS[config.dataset.name]
    test_ds = dataset_obj.from_path(config.dataset, mode="Test")
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )
    print(f"Test samples: {len(test_ds)}")

    # Model
    class_tokens = torch.nn.Parameter(
        torch.randn(config.dataset.num_classes, config.dataset.num_tokens, 256)
        / math.sqrt(256)
    )
    model = SegmentationModel(config, {config.dataset.name: class_tokens})

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint from epoch {epoch}")

    model = model.to(device)

    # Evaluate
    results = evaluate(model, test_dl, out_dir)

    # Aggregate metrics
    metrics = ["dice", "iou", "precision", "recall", "hausdorff"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results if not (isinstance(r[m], float) and np.isnan(r[m]))]
        agg[m] = float(np.mean(vals)) if vals else float("nan")

    # Save per-sample CSV
    csv_path = out_dir / "metrics_per_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + metrics)
        writer.writeheader()
        for r in results:
            row = {"filename": r["filename"]}
            for m in metrics:
                v = r[m]
                row[m] = round(v, 6) if not (isinstance(v, float) and np.isnan(v)) else "nan"
            writer.writerow(row)

    # Save summary
    summary_path = out_dir / "metrics_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + metrics)
        writer.writeheader()
        row = {"model": f"sam2rad_epoch{epoch}"}
        row.update({m: round(agg[m], 4) for m in metrics})
        writer.writerow(row)

    print(f"\n{'='*50}")
    print(f"SAM2Rad evaluation — epoch {epoch}")
    print(f"{'='*50}")
    for m in metrics:
        print(f"  {m:<12} {agg[m]:.4f}")
    print(f"\nPer-sample CSV : {csv_path}")
    print(f"Summary CSV    : {summary_path}")
    print(f"Overlays       : {out_dir / 'overlays'}")


if __name__ == "__main__":
    main()
