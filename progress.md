# 3D US Reconstruction — Project Progress
**Project:** Preprocessing, Segmentation and 3D Reconstruction in Facial Ultrasound for Fracture Detection  
**Student:** Safit Levy | **Supervisor:** Dr. Eli Appelboim  
**Last updated:** 2026-06-13  

---

## Current status summary

**Project 1 is COMPLETE** (report deadline was 2026-06-06).  
**Project 2 is now active** — sparse 3D reconstruction and CT-guided alignment.

### Project 2 pipeline milestones
| # | Milestone | Status |
|---|-----------|--------|
| 1 | Extract bone surface points from 2D masks → sparse 3D PLY | ✅ Done |
| 2 | Extract zygomatic arch surface from CT (mesh / point cloud) | ⬜ TODO |
| 3 | Initial rough alignment + rigid ICP to CT surface | ⬜ TODO |
| 4 | Quantitative evaluation (mean/median/RMSE/P95 surface distances) | ⬜ TODO |
| 5 | (Optional) Compare manual vs U-Net vs SAM2Rad masks; Δ sensitivity | ⬜ TODO |

### Milestone 1 results (2026-06-07, v2 refined)
Script: `3D-Reconstruction/build_point_cloud.py`  
Test run — Patient 2, scan `image_172731958799`, Δ=1.0 px/frame:
- **20 frames**, **9,119 bone-surface points** (top_boundary) / 9,066 (skeleton)
- X (US lateral): 216–799 px | Y (US depth): 224–323 px | Z (sweep): 0–N×Δ
- Outputs in `3D-Reconstruction/output/` — open PLY in MeshLab / CloudCompare

**Key script parameters:**
- `--scan_id` — which cine scan to process  
- `--patient_dir` — Patient1 or Patient2  
- `--delta D [D ...]` — single or multiple Δ values for sensitivity analysis  
- `--extraction_method` — `top_boundary` (default) | `all_mask_pixels` | `skeleton`  
- `--midframe F` — DICOM frame index of sweep turnaround; folds forward+backward sweeps  
- `--qc` — generates 5-frame QC figure showing US frame | mask | extracted curve  
- `--no_ply`, `--no_figure`, `--no_csv` — suppress individual outputs  

**Summary CSV:** `3D-Reconstruction/output/metrics/pointcloud_summary.csv`  
**QC figures:** `3D-Reconstruction/output/qc/{scan_id}_curve_extraction_qc.png`

---

- **Patient 1**: ~83 frames labeled across 7 scans → training set (some frames re-labeled for consistency in v2)
- **Patient 2**: 60 frames labeled across 2 scans → validation/test set (expanded from 30 in v1)
- **Route A (U-Net)**: ✅ Trained and evaluated (v2 run on 143-frame dataset)
- **Route B (UltraSAM)**: ❌ SKIPPED — too complex to implement within deadline (requires Python 3.8 env + OpenMMLab + COCO conversion, estimated 1–2 days setup)
- **Route C (SAM2Rad)**: ✅ Trained and evaluated (v2 run on 143-frame dataset)
- **Report**: Submitted 2026-06-06 (Project 1 complete)

---

## FINAL EVALUATION RESULTS — Patient 2 test set, 60 frames (U-Net v2 + SAM2Rad v3)

Both models trained on the full 143-frame dataset (Patient1=83, Patient2=60) and evaluated with identical metric implementations on all 60 Patient 2 frames. SAM2Rad updated to the v3 run (clean 100-epoch run, validated every epoch).

| Model | Dice | IoU | Precision | Recall | Hausdorff (px) | HD (%diag) | Scale |
|-------|------|-----|-----------|--------|----------------|------------|-------|
| **U-Net + Augmentation** | **0.674 ± 0.155** | **0.527 ± 0.160** | **0.648 ± 0.162** | **0.730 ± 0.192** | 128.0 ± 76.5 | 17.7% | 512×512 |
| **SAM2Rad (epoch 85, v3)** | 0.648 ± 0.158 | 0.498 ± 0.162 | 0.631 ± 0.173 | 0.694 ± 0.177 | **191.3 ± 124.9** | **13.2%** | 1024×1024 |

**Hausdorff normalization note**: Values are in different pixel spaces (512px vs 1024px input). Diagonal: U-Net = 724 px, SAM2Rad = 1448 px. When normalized, **SAM2Rad has better spatial boundary accuracy** (13.2% vs 17.7%).

**Run-to-run variance note**: SAM2Rad v2 (epoch 79) scored Dice = 0.6585 and v3 (epoch 85) scored 0.6480 on the same 60 frames. The ~0.01 Dice difference is ≈½ a standard error (per-frame std ≈ 0.158, SEM ≈ 0.020) — statistically indistinguishable. Training is unseeded, so the gap is pure run-to-run variance; both runs use identical architecture/data/hyperparameters. v3 is canonical because it was validated every epoch, so the reported number and the training-curve figure come from the **same** run.

**Result CSV locations:**
- U-Net (v2): `Deep Learning-Based Segmentation/runs/unet_with_augmentation_v2_20260612_143014/eval_patient2/`
- SAM2Rad (v3 ep85): `Bone Segmentation/runs/sam2rad_bone_seg_v3_eval_ep85/`
- SAM2Rad training curves (merged 0–99): `Bone Segmentation/runs/sam2rad_bone_seg_v3_eval_plots/`
- Overlay figures (image | GT | prediction): in `overlays/` subfolders of each
- Report figures: `report_figures/fig_metrics_comparison.png`, `fig_65_best_cases.png`, `fig_65_worst_cases.png`

**Key findings:**
- U-Net ahead on all four region metrics (Dice/IoU/Precision/Recall); SAM2Rad better on normalized Hausdorff (boundary accuracy)
- Both models achieve Dice in the 0.65–0.67 range — competitive on a dataset of only 83 training frames
- No overfitting in either model (SAM2Rad: train≈val Dice across the whole run; U-Net+aug: minimal gap)
- U-Net: 1.9M params trained from scratch; SAM2Rad: 6.2M fine-tuned from 44.1M pretrained

### v1 results (30 frames, for reference only — superseded by v2)
| Model | Dice | IoU | Precision | Recall | HD (px) | HD (%diag) |
|-------|------|-----|-----------|--------|---------|------------|
| U-Net + Augmentation (epoch 47) | 0.678 | 0.532 | 0.658 | 0.714 | 88.9 | 12.3% |
| SAM2Rad (epoch 59) | 0.671 | 0.522 | 0.645 | 0.717 | 109.6 | 7.6% |

---

## Dataset details (for report)

### Patient 1 (training set)
- 7 scan files: `image_105551296540`, `image_152993002660`, `image_258976846007`, `image_283536217682`, `image_383229031802`, `image_406314327901`, `image_452503334599`
- ~83 frames total across all scans
- Frame types: single-frame DICOMs and multi-frame cine DICOMs
- All manually labeled using the classical segmentation pipeline

### Patient 2 (validation/test set — never used in training)
- **60 frames** labeled across 2 scans: `image_172731958799` (20 frames) + `image_441560463491` (10 frames) + additional re-labeled and expanded frames
- Completely held out until final evaluation
- **Caveat for report**: Patient 2 serves as both validation (checkpoint selection) and test set due to having only 2 patients — this is unavoidable and must be disclosed

### Image properties
- Original size: 768×1024 px
- Crop box applied: y=[100,700], x=[200,800] → 600×600 working region
- Masks: binary PNG, uint8, values 0 (background) and 255 (bone cortex)
- Each mask accompanied by meta JSON with seeds, params, crop box

---

## Model details (for report)

### U-Net (Route A) — v2
- Architecture: 2D U-Net with encoder depth 3 (32→64→128→256 bottleneck)
- Parameters: ~1.9M total, all trained
- Input: 512×512 grayscale (1-channel)
- Loss: 0.5×BCE + 0.5×Dice
- Optimizer: Adam, lr=0.001, ReduceLROnPlateau (patience=10, factor=0.5)
- Augmentation: RandomHorizontalFlip(p=0.5), RandomRotation(±15°, p=0.5), RandomBrightnessContrast(p=0.5), GaussianNoise(p=0.3)
- Training: 100 epochs, batch=4, Patient1 (83 frames)→train, Patient2 (60 frames)→val
- **v2 best run**: `unet_with_augmentation_v2_20260612_143014`, best epoch **28**, training-time val Dice = 0.6759
- evaluate.py result on 60 frames: Dice = 0.6742 ± 0.1552
- Train/val loss gap: minimal (augmentation effectively reduced overfitting)

### SAM2Rad (Route C) — v3 (canonical)
- Base model: SAM2 Tiny Hiera (Meta), pretrained on SA-1B
- Fine-tuning strategy: transfer learning — image encoder FROZEN (38M params), only decoder (LoRA adapters, rank=8) + prompt learner + learnable class tokens trained (6.2M params)
- Learnable prompts: 1 class × 10 tokens × 256 dims (bone_seg class tokens)
- Input: 1024×1024 RGB (grayscale repeated 3× to match SAM2 input format)
- Loss: Dice + Focal (weights 1.0 + 10.0) + box regression + object score
- Optimizer: AdamW, lr=1e-4, CosineAnnealingLR
- Training: 100 epochs, batch_size=1 + accumulate_grad_batches=4 (effective batch 4; fits 4 GB GPU), Patient1 (83 frames)→train, Patient2 (60 frames)→val
- Validation cadence: **every epoch** (check_val_every_n_epoch=1) → dense, publishable training curve
- **v3 best checkpoint** (by training-time val_dice): `model_epoch=85-val_dice=0.69.ckpt`, val_dice=0.6942 (torchmetrics global), evaluate.py Dice = 0.6480 ± 0.1578 on 60 frames
- Run was resumed once (interrupted at epoch 37 → resumed to 99); checkpoint saving froze at the global-best epoch 85, so `last.ckpt` == epoch 85 (epoch-99 weights not saved — harmless, 85 was best)
- Metric caveat: checkpoint selected by torchmetrics global Dice, which doesn't perfectly track per-sample Dice; with save_top_k=1 other epochs weren't saved, so the per-sample-optimal epoch can't be verified
- No overfitting: train Dice ≈ val Dice across the entire 0–99 run (see `sam2rad_bone_seg_v3_eval_plots/dice.png`)
- SAM2Rad does NOT use the seed points from meta JSONs — it uses learnable class tokens shared across all samples
- **v2 (superseded reference)**: `sam2rad_bone_seg_v2_eval_ep79`, epoch 79, evaluate.py Dice = 0.6585 ± 0.1486 (validated every 20 epochs). Within noise of v3; kept only as a variance reference.
- Reproducibility: `pl.seed_everything(seed, workers=True)` added to `train.py` after v3 — future runs are reproducible (v2/v3 were unseeded)
- CSVLogger added to train.py for future runs (metrics accessible without wandb binary parsing)

---

## What's done

### Infrastructure & Dataset
- [x] Dataset folder structure for Patient 1 and Patient 2
- [x] `metadata_labeled.csv` — contains all Patient 1 + Patient 2 labeled pairs
- [x] `build_metadata_labeled.py` — rebuilds CSV from folder structure
- [x] `dcm_to_png_batch.py` — batch DICOM → PNG extraction
- [x] `dcm_extract_frames_manual.py` — interactive frame picker from single DICOM

### Classical Segmentation Pipeline
- [x] `ultrasound_bone_segmentation_cli.py` — full parameter set including: `seed_x_band`, `seed_y_band`, `seed_y_band_up`, `pre_snake_dilate`, `post_trim_up`, `post_trim_down`
- [x] `run_patient1_by_index.py` — `view` command, `--patient_dir` support

### Dataset Labeling — v2
- [x] Patient 1: fully labeled (~83 frames, 7 scans); some frames re-labeled for mask consistency
- [x] Patient 2: **60 frames** labeled across 2 scans (expanded from 30 in v1)
- [x] `metadata_labeled.csv` rebuilt — 143 entries (Patient1=83, Patient2=60)

### Deep Learning — U-Net (Route A) ✓ COMPLETE
- [x] `dataset.py` — crop-aware, patient/scan filtering
- [x] `augmentation.py` — online augmentation pipeline
- [x] `UNet/train.py` — config-driven, all metrics, per-run output dirs
- [x] `UNet/evaluate.py` — standalone evaluation on any patient split
- [x] `plot_training.py` — training curve plots
- [x] **v2 best run (with aug)**: `unet_with_augmentation_v2_20260612_143014` (best epoch 28, val Dice 0.674)
- [x] **v2 no-aug baseline**: `unet_no_augmentation_v2_20260614_131100` (best epoch 34, val Dice 0.675) — same v2 dataset/hyperparameters, augmentation OFF. Used for the overfitting figure: train loss → 0.076 while val loss diverges to ~0.23 after ep 34 (classic overfitting). Controlled A/B vs the with-aug run on identical data (supersedes the old v1 no-aug curve). Plot: `runs/unet_no_augmentation_v2_20260614_131100/plots/loss.png`
- [x] v1 reference runs: `unet_with_augmentation_20260531_143449` (best ep 47), `unet_baseline_20260531_143022` (v1 no-aug, 30-frame)

### Deep Learning — SAM2Rad (Route C) ✓ COMPLETE
- [x] Repo cloned, `bone_seg` dataset registered in `known_datasets.py`
- [x] `bone_seg.yaml` config (SAM2 tiny, 1 class, 10 tokens, 100 epochs)
- [x] `prepare_sam2rad_data.py` — converts CSV+masks to Train/Test folder structure
- [x] **v3 training (canonical)**: 100 epochs, val every epoch; best `runs/sam2rad_bone_seg_v3/model_epoch=85-val_dice=0.69.ckpt`, evaluate.py Dice = 0.6480
- [x] v2 training (superseded reference): `model_epoch=79`, evaluate.py Dice = 0.6585 — within noise of v3
- [x] `evaluate.py` — standalone evaluation with all 5 metrics + overlays
- [x] `plot_sam2rad_training.py` — reads CSVLogger; merges version_6 (ep 0–37) + version_7 (ep 38–99) for full curve
- [x] CSVLogger added to `train.py` (saves per-epoch metrics to `logs/csv_metrics/`)
- [x] `pl.seed_everything` added to `train.py` for reproducible future runs

### Evaluation ✓ COMPLETE
- [x] Both models evaluated on all **60 Patient 2 frames** with identical metric implementations
- [x] Per-sample CSVs + summary CSVs + overlay figures saved for U-Net v2 + SAM2Rad v3
- [x] Report figures regenerated from v3 ep85: `fig_metrics_comparison.png`, `fig_65_best_cases.png`, `fig_65_worst_cases.png`
- [x] SAM2Rad training-curve plots (full 0–99) in `runs/sam2rad_bone_seg_v3_eval_plots/`

---

## Remaining tasks (this week)

### Project 2 — next steps
- [ ] Milestone 2: Extract zygomatic arch surface from CT (mesh/point cloud)
- [ ] Milestone 3: Initial rough alignment + rigid ICP to CT surface
- [ ] Milestone 4: Quantitative evaluation (mean/median/RMSE/P95 surface distances)
- [ ] Milestone 5 (optional): Compare manual vs U-Net vs SAM2Rad masks; Δ sensitivity analysis

---

## Key decisions made

1. **Staged strategy:** Project 1 = 2D segmentation. Project 2 = 3D reconstruction.
2. **Segmentation target:** Visible cortical hyperechoic bone interface — not full bone volume.
3. **Two-route comparison completed:** Route A (U-Net ✓) + Route C (SAM2Rad ✓). Route B (UltraSAM) skipped due to deadline.
4. **Patient-wise splitting only.** Patient 1 (83 frames) → train, Patient 2 (60 frames) → val/test.
5. **Patient 2 dual use:** serves as both val (checkpoint selection) and test (final evaluation). Must disclose in report.
6. **Online augmentation** used for U-Net — nearly eliminated overfitting.
7. **SAM2Rad transfer learning** — frozen encoder, LoRA decoder fine-tuning. No manual prompts — uses learned class tokens.
8. **UltraSAM skipped** — too heavy (OpenMMLab framework, Python 3.8 env, COCO conversion) for remaining time.
9. **Hausdorff comparison caveat** — U-Net at 512px, SAM2Rad at 1024px. Normalize by image diagonal for fair comparison.

---

## Known issues / limitations to disclose in report

- Only 2 patients — Patient 2 is simultaneously val and test set (no truly held-out test)
- 60 Patient 2 evaluation frames is still a relatively small test set from a single patient
- Crop coordinates hard-coded (y=[100,700], x=[200,800]) — may not generalize to new patients/scanners
- Hausdorff computed at different image scales — normalized comparison needed
- SAM2Rad does not use manually placed seed points — purely data-driven via class tokens
- Classical segmentation pipeline parameters tuned per-frame — not fully automated

---

## File map (key project files)

| File | Purpose |
|------|---------|
| `ultrasound_bone_segmentation_cli.py` | Classical interactive bone mask labeling tool |
| `run_patient1_by_index.py` | Runner for any patient |
| `Pre-processing/build_metadata_labeled.py` | Builds `metadata_labeled.csv` |
| `Dataset/metadata_labeled.csv` | Source of truth for ML training |
| `Deep Learning.../dataset.py` | Shared crop-aware PyTorch Dataset |
| `Deep Learning.../augmentation.py` | Online augmentation |
| `Deep Learning.../UNet/train.py` | U-Net training (config-driven) |
| `Deep Learning.../UNet/evaluate.py` | U-Net evaluation on any patient split |
| `Deep Learning.../UNet/configs/default.yaml` | U-Net training config |
| `Deep Learning.../SAM2Rad/train.py` | SAM2Rad training (modified for Patient2 val) |
| `Deep Learning.../SAM2Rad/evaluate.py` | SAM2Rad evaluation with all 5 metrics |
| `Deep Learning.../SAM2Rad/sam2rad/configs/bone_seg.yaml` | SAM2Rad config |
| `Deep Learning.../runs/unet_with_augmentation_v2_20260612_143014/` | Best U-Net run (v2, 60 val frames) |
| `Deep Learning.../runs/unet_with_augmentation_20260531_143449/` | v1 U-Net run (30 val frames, reference) |
| `Bone Segmentation/runs/sam2rad_bone_seg_v3/` | SAM2Rad v3 checkpoints (best ep85 + last) |
| `Bone Segmentation/runs/sam2rad_bone_seg_v3_eval_ep85/` | SAM2Rad v3 evaluation results (canonical) |
| `Bone Segmentation/runs/sam2rad_bone_seg_v3_eval_plots/` | SAM2Rad v3 training curves (merged 0–99) |
| `Bone Segmentation/runs/sam2rad_bone_seg_v2_eval_ep79/` | SAM2Rad v2 evaluation (superseded reference) |
| `Deep Learning.../prepare_sam2rad_data.py` | Converts CSV+masks to SAM2Rad format |
| `Deep Learning.../plot_training.py` | U-Net training curve plots |
| `Deep Learning.../SAM2Rad/plot_sam2rad_training.py` | SAM2Rad training curves (parses wandb binary) |
| `report_figures/` | Final report figures (metrics comparison, overlay cases) |
