# 3D US Reconstruction — Project Progress
**Project:** Preprocessing and Bone Segmentation in Facial Ultrasound for Fracture Detection  
**Student:** Safit Levy | **Supervisor:** Dr. Eli Appelboim  
**Last updated:** 2026-05-25

---

## Current status summary

The project is in the preprocessing expansion stage. A physics-aware preprocessing pipeline has been built alongside the existing classical segmentation pipeline. Three preprocessing modules are complete (FFT filters, SVD/PCA denoising, wavelet denoising). Fourier-domain analysis of Patient 1 data has been completed and used to calibrate the filter parameters. `metadata_labeled.csv` has been rebuilt from disk and now contains 44 labeled image-mask pairs (was 26 — stale). The next milestone is Phase 5: unified preprocessing API + comparison driver to measure which preprocessing combination yields the best Dice/IoU on the labeled set, then begin U-Net training.

---

## Current blockers

- Need visual quality review of all existing masks before using them for training
- Phase 5 (preprocessing API + comparison driver) not yet built — can't objectively pick best preprocessing combination yet
- Need configurable crop parameters before the segmentation pipeline can scale beyond Patient 1
- ~~Patient 2 availability confirmed~~ Patient 2 exists and is designated as the val/test set — completely untouched until evaluation

---

## What's done

### Infrastructure & Dataset
- [x] Dataset folder structure defined and in use:
  ```
  Dataset/Patient1/DCM_frames/, IMG_frames/, Masks/
  ```
- [x] Naming convention established: `image_<id>.png` → `image_<id>_mask.png` + `image_<id>_meta.json`
- [x] `patient1_image_index.json` exists, mapping original DICOM/cine sources to extracted PNGs
- [x] `metadata_labeled.csv` rebuilt from disk — now has **44 rows** (was stale at 26)
- [x] `build_metadata_labeled.py` written and verified against current folder layout

### Classical Segmentation Pipeline
- [x] `ultrasound_bone_segmentation_cli.py` implemented and in active use
  - CLAHE preprocessing (clip=0.01) + Gaussian blur (7×7)
  - Hard-coded crop: y=[100,700], x=[200,800]
  - Interactive seed selection (max 5 seeds)
  - Seed snapping to nearest bright ridge (window=7px)
  - Region growing via `skimage.segmentation.flood` (tolerance=20)
  - Active contour (snake) refinement on Sobel edges
  - Final mask: snake + region-grow combined, dilated (radius=3)
  - Saves binary PNG (0/255) + meta JSON per mask
- [x] `run_patient1_by_index.py` — batch runner for Patient 1 frames

### Literature Review
- [x] Classical freehand 3D US reconstruction (pixel/voxel/function-based methods)
- [x] Real-time octree reconstruction (Victoria et al. 2023)
- [x] AI-guided segmentation → 3D reconstruction (Arsenescu et al. 2023) — closest analog to this project
- [x] Sensorless reconstruction: physics-guided DL (Dou et al. 2024), SPRAO, RapidVol
- [x] Decision made: staged approach — segmentation first, reconstruction later

### Preprocessing Pipeline (Physics-Aware)
- [x] **Phase 1 — Fourier domain analysis** (`Notebooks/fourier_analysis.ipynb`)
  - 2D FFT magnitude spectrum + radial PSD (log-scale, DC-centred)
  - Patch-based PSD for bone vs background comparison (Hanning-windowed, avoids DC contamination)
  - Angular power distribution (directional artifact detection)
  - Data-adaptive LP cutoff via log-log PSD knee detection
  - Parameters derived: LP=0.1663 cyc/px, BP=0.015–0.0975 cyc/px, dominant artifact angle=93°
  - Section 11: overlay comparison of all filters on real Patient 1 frames
- [x] **Phase 2 — FFT-based frequency filters** (`Bone Segmentation/Preprocessing/frequency_filters.py`)
  - Gaussian LP, HP, BP; directional notch filter
  - All defaults calibrated from Phase 1 analysis
  - Unified entry point: `apply_filter(img, method, **kwargs)`
- [x] **Phase 3 — SVD/PCA denoising** (`Bone Segmentation/Preprocessing/svd_denoising.py`)
  - Global rank-k SVD and patch-based PCA (overlapping patches, averaged reconstruction)
  - Rank selection: `rank_from_gap` (elbow) and `rank_from_energy` (cumulative energy)
  - Diagnostic: `svd_scree` plot with twin-axis log-SV + cumulative energy
  - Unified entry point: `apply_svd_filter(img, method, **kwargs)`
- [x] **Phase 4 — Wavelet denoising** (`Bone Segmentation/Preprocessing/wavelet_denoising.py`)
  - Multi-level 2D DWT (default: db6, 4 levels)
  - VisuShrink (universal) and BayesShrink (per-sub-band) thresholding
  - Soft and hard threshold modes; `sigma_scale` multiplier for tuning aggressiveness
  - Noise std estimated via MAD on finest-scale HH sub-band
  - Unified entry point: `apply_wavelet_filter(img, method, **kwargs)`

### Report (Project 1)
- [x] Report structure defined (`Project1_Report_v1_docx.docx` — template/outline stage)
- [x] Introduction and problem statement written
- [x] Project goals and scope written
- [x] Background section headings defined (content TBD)

---

## In progress

### Preprocessing Pipeline
- [ ] **Phase 5 — Unified API + comparison driver** (`preprocessing_api.py`, `compare_preprocessing.py`)
  - Single `preprocess(img, method, **kwargs)` entry point across all modules
  - Comparison driver: run all methods through existing segmentation pipeline, score Dice/IoU/Hausdorff vs ground-truth masks

### Labeling
- [ ] Visual overlay review of all existing masks (check for leakage/errors)
- [ ] Flag uncertain/low-quality masks in `notes` column of CSV

### Report (Project 1)
- [ ] Background & Literature Review — filling in content under existing headings
- [ ] Materials & Methods — documenting the segmentation pipeline formally
- [ ] Results section — requires finished masks + qualitative examples

---

## Next up

### Immediate (before model training)
- [ ] Build Phase 5: `preprocessing_api.py` + `compare_preprocessing.py`
- [ ] Run comparison driver on Patient 1 frames; pick best preprocessing combination by Dice/IoU
- [ ] Generate overlay preview images for all masks (image + mask side-by-side)
- [ ] Flag uncertain/low-quality masks in `notes` column of CSV
- [x] Patient-wise split confirmed: Patient 1 → train, Patient 2 → val/test (Patient 2 untouched until evaluation)
- [x] `metadata_labeled.csv` verified clean (44 pairs)

### Model Training — Route A (first priority)
- [ ] Rewrite `train_unet.py` — architecture exists (`unet_model.py`, `dataset.py`) but training script uses random frame-level split and has no eval metrics; needs patient-wise split and full metric suite
- [ ] DataLoader filters by patient from `metadata_labeled.csv` (Patient 1 → train, Patient 2 → val/test)
- [ ] Loss: Dice + BCE (already implemented, keep)
- [ ] Metrics: Dice, IoU, Precision/Recall, Hausdorff distance
- [ ] Save quantitative metrics + overlay figures for report

### Model Training — Route B/C (secondary, time permitting)
- [ ] Route B: UltraSAM fine-tuning setup
- [ ] Route C: SAM2Rad-style / prompt-based segmentation
- [ ] Compare both against U-Net baseline

### Report (Project 1) — remaining sections
- [ ] Fill Background & Literature Review content
- [ ] Write Materials & Methods (segmentation pipeline details)
- [ ] Write Evaluation Methodology
- [ ] Write Results (after U-Net baseline is done)
- [ ] Write Discussion + Conclusion

### Future — Project 2 (3D Reconstruction)
- [ ] 3D point-cloud / surface reconstruction from bone masks
- [ ] Approximate frame stacking (if controlled acquisition)
- [ ] Pose-aware reconstruction if tracking data becomes available
- [ ] Sensorless learned pose estimation (long-term)
- [ ] Fracture detection / visualization from reconstructed surface

---

## Key decisions made

1. **Staged strategy:** Full sensorless 3D reconstruction is too ambitious as the first step given current data scale. Project 1 = 2D segmentation. Project 2 = 3D reconstruction.

2. **Segmentation target:** The visible cortical hyperechoic bone interface in the ultrasound B-scan — not the full anatomical bone volume.

3. **First model = plain 2D U-Net.** Must be working before attempting UltraSAM or SAM2Rad.

4. **Patient-wise splitting is mandatory.** Never split by frame randomly — adjacent cine frames from the same patient must stay in the same split. Proposed: Patient 1 → train, Patient 2 → test.

5. **Mask cleaning is currently disabled** in the segmentation CLI (commented out at line ~685). Region growing output is used directly. Re-enable by uncommenting that block — there is no `--use_cleaning` flag.

6. **Crop coordinates are hard-coded** (y=[100,700], x=[200,800]). Needs to be made configurable per patient/scan in a future improvement.

7. **`metadata_labeled.csv` is the source of truth** for all training. Never train from raw file-system traversal alone.

8. **Raw DICOMs are never overwritten.** All processing outputs go to separate folders.

9. **Report frames the work honestly:** This is a research proof-of-concept pipeline, not a clinical tool. Segmentation is the necessary prerequisite before reconstruction and fracture analysis.

---

## Known issues / watch-outs

- Crop coordinates may not generalize to all frames, patients, or scanner exports
- Region growing can leak into bright non-bone structures (especially without cleaning)
- Snake refinement may fail on weak or fragmented bone edges
- Mask thickness is artificial (controlled by dilation radius, not true anatomy)
- No external pose/tracking data currently available → sensorless reconstruction is a future challenge

---

## File map (key project files)

| File | Purpose |
|------|---------|
| `ultrasound_bone_segmentation_cli.py` | Classical interactive bone mask labeling tool |
| `run_patient1_by_index.py` | Batch runner for Patient 1 segmentation |
| `build_metadata_labeled.py` | Auto-builds `metadata_labeled.csv` from folder |
| `metadata_labeled.csv` | Ground-truth labeled pairs for ML training (44 rows) |
| `dataset_structure.md` | Dataset folder layout reference |
| `Project1_Report_v1_docx.docx` | Report draft (template/outline stage) |
| `Notebooks/fourier_analysis.ipynb` | Phase 1 Fourier analysis — calibrates filter parameters |
| `Bone Segmentation/Preprocessing/frequency_filters.py` | Phase 2 FFT-based LP/HP/BP/notch filters |
| `Bone Segmentation/Preprocessing/svd_denoising.py` | Phase 3 global SVD + patch PCA denoising |
| `Bone Segmentation/Preprocessing/wavelet_denoising.py` | Phase 4 wavelet denoising (VisuShrink + BayesShrink) |
