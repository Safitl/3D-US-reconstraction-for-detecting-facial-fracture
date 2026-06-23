# CLAUDE.md — Project Briefing

> This file is read automatically at every Claude Code session start.  
> After reading this file, also read `progress.md` and say: "Ready. Here's where we left off: …"

---

## What this project is

**3D Ultrasound Reconstruction for Detecting Facial Fractures**  
A two-part research pipeline developed as a Final Project A/B at the Technion, Electrical Engineering Faculty.

- **Project 1 (COMPLETE):** Preprocessing and 2D bone segmentation in facial ultrasound frames
- **Project 2 (ACTIVE):** Sparse 3D reconstruction and CT-guided alignment of the zygomatic arch surface

The long-term goal is a non-ionizing, portable, low-cost alternative to CT for initial or repeated facial fracture assessment.

**Project 1 is COMPLETE (report submitted 2026-06-06). Project 2 is now active.**

### Project 2 scope
CT-guided sparse 3D reconstruction and alignment — NOT a full dense sensorless reconstruction.
No external probe tracking exists, so z is estimated from frame order × assumed spacing Δ.
CT is used as the anatomical reference and evaluation target (rigid ICP alignment).

**Clinical data:**
- Case 1 (Patient 1): healthy left zygomatic arch US + CT
- Case 2 (Patient 2): right fractured zygomatic arch + contralateral healthy arch US + CT
  - scan `image_172731958799` — labeled frames
  - scan `image_441560463491` — labeled frames
  - Total: **60 labeled frames** across both scans (expanded from 30 in v1)

### Project 2 milestones
1. ✅ Milestone 1: Extract bone surface points from 2D masks → sparse 3D PLY  
   Script: `3D-Reconstruction/build_point_cloud.py`
2. ⬜ Milestone 2: Extract zygomatic arch surface from CT (mesh/point cloud)
3. ⬜ Milestone 3: Initial rough alignment + rigid ICP to CT surface
4. ⬜ Milestone 4: Quantitative evaluation (mean/median/RMSE/P95 surface distances)
5. ⬜ Milestone 5 (optional): Compare manual vs U-Net vs SAM2Rad masks; Δ sensitivity analysis

---

## Repository name

```
3D-US-reconstraction-for-detecting-facial-fracture
```

Note: "reconstraction" is a typo in the original repo name. Do not rename mid-project.

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3 |
| Image I/O | OpenCV (`cv2`), `pydicom` |
| Classical segmentation | `scikit-image` (`flood`, `active_contour`), `scipy`, `skimage.morphology` |
| Deep learning | PyTorch, PyTorch Lightning |
| DL training framework | Custom config-driven (YAML), `wandb` (SAM2Rad only, offline mode) |
| Data management | pandas, CSV, JSON |
| Visualization | matplotlib, OpenCV overlays |
| File formats | PNG (frames + masks), DICOM (raw), JSON (metadata per mask), CSV (dataset index), YAML (configs) |

---

## Folder structure

```
3D-US-reconstraction-for-detecting-facial-fracture/
├── 3D-Reconstruction/
│   ├── build_point_cloud.py      ← Milestone 1: masks → sparse 3D PLY
│   └── output/                   ← generated PLY files and overview figures
├── patient1_image_index.json
├── Patient2_image_index.json
├── Pre-processing/
│   ├── build_metadata_labeled.py
│   ├── dcm_to_png_batch.py
│   └── dcm_extract_frames_manual.py
├── Dataset/
│   ├── metadata_labeled.csv          ← source of truth for ML training
│   ├── Patient1/
│   │   ├── DCM_frames/               ← READ ONLY
│   │   ├── IMG_frames/               ← ~83 frames
│   │   └── Masks/                    ← ~83 binary masks + meta JSONs
│   └── Patient2/
│       ├── DCM_frames/               ← READ ONLY
│       ├── IMG_frames/               ← extracted frames
│       └── Masks/                    ← 60 labeled masks + meta JSONs
└── Bone Segmentation/
    └── Deep Learning-Based Segmentation/
        ├── dataset.py                ← shared, crop-aware PyTorch Dataset
        ├── augmentation.py           ← shared online augmentation
        ├── plot_training.py          ← training curve plots (--run_dir)
        ├── prepare_sam2rad_data.py   ← converts CSV+masks → SAM2Rad format
        ├── UNet/
        │   ├── unet_model.py         ← 2D U-Net architecture
        │   ├── train.py              ← config-driven training
        │   ├── evaluate.py           ← standalone evaluation script
        │   └── configs/default.yaml
        ├── SAM2Rad/                  ← cloned from github.com/aswahd/SamRadiology
        │   ├── train.py              ← modified for Patient2 val set
        │   ├── evaluate.py           ← evaluation with all 5 metrics
        │   ├── checkpoints/          ← saved model checkpoints
        │   ├── weights/sam2_hiera_tiny.pt
        │   └── sam2rad/
        │       ├── configs/bone_seg.yaml
        │       └── datasets/known_datasets.py  ← bone_seg registered here
        └── runs/
            ├── unet_baseline_20260531_143022/              ← v1 no-aug (30 val frames)
            ├── unet_with_augmentation_20260531_143449/     ← v1 U-Net (30 val frames)
            ├── unet_with_augmentation_v2_20260612_143014/  ← BEST U-Net v2 (60 val frames)
            └── unet_no_augmentation_v2_20260614_131100/    ← v2 no-aug baseline (overfitting figure, same v2 data)
Bone Segmentation/
└── runs/
    ├── sam2rad_bone_seg_v3/                  ← CANONICAL v3 checkpoints (best ep85 + last)
    ├── sam2rad_bone_seg_v3_eval_ep85/        ← BEST SAM2Rad v3 (epoch 85) — canonical results
    ├── sam2rad_bone_seg_v3_eval_plots/       ← v3 training curves (merged ep 0–99)
    ├── sam2rad_bone_seg_v2_eval_ep79/        ← SAM2Rad v2 (epoch 79) — superseded reference
    └── sam2rad_bone_seg_v2_eval_ep39/        ← SAM2Rad v2 epoch 39 (comparison)
```

---

## Final evaluation results (for report)

**Canonical results — evaluated on all 60 Patient 2 frames** (metadata_labeled.csv: 143 total, Patient1=83, Patient2=60). U-Net = v2, SAM2Rad = v3 (clean 100-epoch run, val every epoch):

| Model | Dice | IoU | Precision | Recall | Hausdorff (px) | HD (%diag) | Scale |
|-------|------|-----|-----------|--------|----------------|------------|-------|
| U-Net + Augmentation (ep 28) | **0.674 ± 0.155** | **0.527 ± 0.160** | **0.648 ± 0.162** | **0.730 ± 0.192** | 128.0 ± 76.5 | 17.7% | 512px |
| SAM2Rad (epoch 85, v3) | 0.648 ± 0.158 | 0.498 ± 0.162 | 0.631 ± 0.173 | 0.694 ± 0.177 | 191.3 ± 124.9 | **13.2%** | 1024px |

Hausdorff normalized by diagonal (512px→724px, 1024px→1448px): SAM2Rad better spatial accuracy (13.2% vs 17.7%).

**SAM2Rad run-to-run variance:** v2 ep79 = 0.6585, v3 ep85 = 0.6480 on the same 60 frames — ~0.01 Dice apart (≈½ SEM), statistically equivalent. Training was unseeded; v3 is canonical because number + curve come from the same per-epoch-validated run. `pl.seed_everything` added afterward for reproducible future runs.

**v1 results — 30 frames (reference):** U-Net Dice=0.678, SAM2Rad Dice=0.671 (epoch 59, HD%diag: U-Net=12.3%, SAM2Rad=7.6%)

---

## Model architectures (for report)

### U-Net
- 2D U-Net, 3 encoder levels, base_ch=32 → channels: 32→64→128, bottleneck=256
- ~1.9M parameters, trained from scratch
- Input: 512×512, 1-channel grayscale
- Loss: 0.5×BCE + 0.5×Dice loss
- Optimizer: Adam lr=1e-3, ReduceLROnPlateau (patience=10)
- Augmentation: horizontal flip, rotation ±15°, brightness/contrast, Gaussian noise
- **v2 best run**: `unet_with_augmentation_v2_20260612_143014`, epoch 28, val Dice 0.6742 on 60 Patient 2 frames

### SAM2Rad
- Base: SAM2 Tiny Hiera (Meta), pretrained on SA-1B
- Transfer learning: encoder FROZEN (38M params), fine-tuned: LoRA decoder (rank=8) + prompt learner + class tokens (6.2M params)
- Learnable prompts: 1 class × 10 tokens × 256 dims (no manual seed prompts)
- Input: 1024×1024, 3-channel (grayscale→RGB)
- Loss: Dice + Focal (1.0 + 10.0) + box regression + object score
- Optimizer: AdamW lr=1e-4, CosineAnnealingLR
- v3 batch_size=1 + accumulate_grad_batches=4 (effective batch 4; fits 4 GB GPU), validation **every epoch**
- **v3 best checkpoint (canonical)**: `runs/sam2rad_bone_seg_v3/model_epoch=85-val_dice=0.69.ckpt` (val_dice 0.6942 torchmetrics), evaluate.py Dice 0.6480 on 60 Patient 2 frames
- v3 was resumed once (interrupted at ep37→99); checkpoint saving froze at global-best ep85, so `last.ckpt` == ep85 weights
- v2 (superseded): `model_epoch=79` evaluate.py Dice 0.6585 — within noise of v3
- CSVLogger + `pl.seed_everything` in train.py; v3 curve merges logs/csv_metrics/version_6 (ep0–37) + version_7 (ep38–99)

---

## Key scripts and how to run them

```bash
# Rebuild metadata CSV
python "Pre-processing/build_metadata_labeled.py" --dataset_root Dataset

# Segmentation labeling tool (interactive)
python "Bone Segmentation/Region Growing Segmentation/seg/run_patient1_by_index.py"
python "Bone Segmentation/Region Growing Segmentation/seg/run_patient1_by_index.py" --patient_dir Patient2

# U-Net training
python "Bone Segmentation/Deep Learning-Based Segmentation/UNet/train.py"
python "Bone Segmentation/Deep Learning-Based Segmentation/UNet/train.py" --run_name my_name

# U-Net evaluation on Patient 2
python "Bone Segmentation/Deep Learning-Based Segmentation/UNet/evaluate.py" \
  --checkpoint "Bone Segmentation/Deep Learning-Based Segmentation/runs/unet_with_augmentation_v2_20260612_143014/best_model.pth" \
  --csv_path Dataset/metadata_labeled.csv --base_dir .

# SAM2Rad training
cd "Bone Segmentation/Deep Learning-Based Segmentation/SAM2Rad"
set WANDB_MODE=offline
python train.py --config sam2rad/configs/bone_seg.yaml

# SAM2Rad evaluation (run from SAM2Rad/ folder) — canonical v3 ep85
python evaluate.py --config sam2rad/configs/bone_seg.yaml \
  --checkpoint ../../runs/sam2rad_bone_seg_v3/model_epoch=85-val_dice=0.69.ckpt \
  --out_dir ../../runs/sam2rad_bone_seg_v3_eval_ep85

# SAM2Rad training curves plot (run from SAM2Rad/ folder)
python plot_sam2rad_training.py

# Plot training curves
python "Bone Segmentation/Deep Learning-Based Segmentation/plot_training.py" \
  --run_dir "Bone Segmentation/Deep Learning-Based Segmentation/runs/<run_name>"
```

---

## Segmentation pipeline parameters (for report — Materials & Methods)

The classical segmentation pipeline applies to each frame:
1. **Preprocessing:** CLAHE (clip=0.01) + Gaussian blur (7×7 default)
2. **Crop:** y=[100,700], x=[200,800] → 600×600 working region
3. **Seed selection:** Interactive clicks, snapped to nearest bright ridge (window=7px)
4. **Region growing:** `skimage.segmentation.flood` with per-seed spatial constraints
   - `seed_x_band` — horizontal flood window (±N cols per seed)
   - `seed_y_band` — symmetric vertical window (±N rows per seed)
   - `seed_y_band_up` — asymmetric upward-only restriction
5. **Morphological cleaning:** opening (speckle removal) + horizontal closing (gap bridging)
6. **Pre-snake dilation:** isotropic disk expansion to bridge diagonal gaps on curved bone
7. **Active contour (snake):** Sobel edge-based, LoRA refinement
   - `snake_alpha` (tension), `snake_beta` (stiffness), `snake_sigma` (edge smoothing)
8. **Post-processing trims:** `post_trim_up` / `post_trim_down` — hard clip above/below interpolated seed line; uses linear interpolation → follows bone curve smoothly
9. **Output:** Binary PNG (0/255) + meta JSON with all parameters logged

---

## Architectural decisions

1. **Segmentation before reconstruction.** No 3D work until 2D masks are complete.
2. **Segmentation target:** Visible cortical hyperechoic bone interface only.
3. **Two-route comparison completed:** U-Net (Route A) + SAM2Rad (Route C). UltraSAM (Route B) skipped.
4. **Patient-wise split only.** Patient 1 → train (83 frames), Patient 2 → val/test (60 frames).
5. **Patient 2 dual use:** val for model selection AND test for final evaluation. Disclose in report.
6. **Online augmentation** (U-Net) — nearly eliminated overfitting.
7. **Transfer learning** (SAM2Rad) — frozen SAM2 encoder, fine-tuned decoder only.
8. **`metadata_labeled.csv` is the single source of truth** for labeled data.
9. **Masks are binary PNG, uint8, 0/255.** No float masks.
10. **Raw DICOMs never modified.** DCM_frames/ is read-only.

---

## Constraints

- **Never modify files under `DCM_frames/`.**
- **Never commit raw patient data to a public repository.**
- **Never use random frame-level train/test splits.**
- **Never train on Patient 2 frames.**
- **Never produce float masks.**

---

## Report structure (Project 1) — SUBMITTED 2026-06-06

File: `Project1_Report_v1_docx.docx`  
All sections written and submitted. v2 results (60 frames) generated after submission for reference.

**Results summary in report:**
- Comparison table (both models on Patient 2, v2: 60 frames)
- Training curves (loss, Dice plots) — U-Net from CSV, SAM2Rad from wandb binary via `plot_sam2rad_training.py`
- Overlay figures: `report_figures/fig_65_best_cases.png`, `fig_65_worst_cases.png`
- Hausdorff scale difference noted and normalized values reported

**Key discussion points in report:**
- U-Net competitive despite simpler architecture and no pretraining
- SAM2Rad advantage: better normalized Hausdorff (boundary accuracy), less overfitting by design
- Limitation: only 2 patients, Patient 2 used as both val and test
- Future: UltraSAM (ultrasound-specific pretraining), 3D reconstruction (Project 2)

---

## Literature reviewed

| Paper | Relevance |
|-------|-----------|
| Arsenescu et al. 2023 | Closest analog: 2D US segmentation → 3D reconstruction, MultiResUNet |
| Victoria et al. 2023 | Real-time octree-based 3D US reconstruction |
| Dou et al. 2024 | Sensorless physics-guided DL reconstruction — future direction |
| Solberg et al. 2011 | Classical freehand 3D US reconstruction algorithms |
| SAM2 (Meta 2024) | Base model for SAM2Rad — Segment Anything Model 2 |
| SAM2Rad (Wahd et al.) | Medical image segmentation with learnable prompts on SAM2 |

---

## Session startup checklist

1. Read this file (`CLAUDE.md`)
2. Read `progress.md`
3. Do not touch `DCM_frames/`
4. Confirm task before writing code
