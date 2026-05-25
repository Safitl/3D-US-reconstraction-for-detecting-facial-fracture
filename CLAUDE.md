# CLAUDE.md — Project Briefing

> This file is read automatically at every Claude Code session start.  
> After reading this file, also read `progress.md` and say: "Ready. Here's where we left off: …"

---

## What this project is

**3D Ultrasound Reconstruction for Detecting Facial Fractures**  
A two-part research pipeline developed as a Final Project A/B at the Technion, Electrical Engineering Faculty.

- **Project 1 (current):** Preprocessing and 2D bone segmentation in facial ultrasound frames
- **Project 2 (future):** 3D reconstruction and fracture analysis from segmented bone masks

The long-term goal is a non-ionizing, portable, low-cost alternative to CT for initial or repeated facial fracture assessment. The immediate engineering goal is a reliable, reproducible 2D bone segmentation pipeline that produces binary bone masks from freehand ultrasound B-scans.

**We are currently in Project 1.** Do not attempt to implement 3D reconstruction until segmentation is stable and evaluated.

---

## Repository name

```
3D-US-reconstraction-for-detecting-facial-fracture
```

Note: "reconstraction" is a typo in the original repo name. Do not rename mid-project — it would break paths and collaborator references.

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3 |
| Image I/O | OpenCV (`cv2`), `pydicom` |
| Classical segmentation | `scikit-image` (`flood`, `active_contour`), `scipy`, `skimage.morphology` |
| Deep learning (planned) | PyTorch |
| Data management | pandas, CSV |
| Visualization | matplotlib, OpenCV overlays |
| File formats | PNG (frames + masks), DICOM (raw), JSON (metadata per mask), CSV (dataset index) |

---

## Folder structure

```
3D-US-reconstraction-for-detecting-facial-fracture/
├── patient1_image_index.json         ← maps IMG_frames filenames to index; lives at repo root
└── Dataset/
    ├── dataset_structure.md
    ├── metadata_labeled.csv          ← source of truth for ML training
    └── Patient1/
        ├── DCM_frames/               ← raw DICOM files (READ ONLY — never modify)
        │   ├── image_<id>.dcm
        │   └── ...
        ├── IMG_frames/               ← extracted PNG frames
        │   ├── image_<id>.png
        │   ├── image_<id>_f000.png   ← cine-derived frames use _fNNN suffix
        │   └── ...
        └── Masks/
            ├── image_<id>_mask.png   ← binary mask (0 = background, 255 = bone)
            ├── image_<id>_meta.json  ← seed points + segmentation params used
            └── ...
```

Additional patients will follow the same structure as `Patient1/`.

---

## Key scripts

| Script | Role |
|--------|------|
| `ultrasound_bone_segmentation_cli.py` | Main interactive labeling tool — runs on a single frame |
| `run_patient1_by_index.py` | Batch runner: segments Patient 1 frames by index |
| `build_metadata_labeled.py` | Scans folder structure, builds/updates `metadata_labeled.csv` |

---

## How to run the segmentation tool

```bash
python ultrasound_bone_segmentation_cli.py \
  --image_path Dataset/Patient1/IMG_frames/image_<id>.png \
  --output_mask_path Dataset/Patient1/Masks/image_<id>_mask.png \
  --show
```

Optional flags:
```
--max_seeds      (default: 5)
--tolerance      (default: 20)
--snap_window    (default: 7)
--snake_dilate   (default: 3)
--y_band         (default: 35)
--min_area       (default: 200)
--open_r         (default: 1)
--close_w        (default: 25)
--use_cleaning   (flag does NOT exist — cleaning is disabled by commented-out code at line ~685 of the CLI; re-enable by uncommenting that block, not via a flag)
```

## How to rebuild the metadata CSV

```bash
python build_metadata_labeled.py
```

Verify output: `Dataset/metadata_labeled.csv` should list all image-mask pairs with correct relative paths.

---

## Architectural decisions

1. **Segmentation before reconstruction.** No 3D work until 2D masks are reliable and evaluated. This is a fixed project-level decision.

2. **The segmentation target is the visible cortical bone interface** — the hyperechoic bright line at the bone surface in the B-scan. It is NOT the full anatomical bone volume.

3. **Plain 2D U-Net is the first model.** It must be implemented and producing baseline results before any foundation model (UltraSAM, SAM2Rad) is attempted.

4. **Patient-wise splitting only.** Never split randomly by frame. Adjacent cine frames from the same patient/scan must stay in the same split. Planned split: Patient 1 → train, Patient 2 → test.

5. **`metadata_labeled.csv` is the single source of truth** for what counts as labeled data. Never train by crawling the filesystem directly.

6. **Masks are binary PNG, uint8, values 0 and 255.** No other format. No float masks.

7. **All arrays and volumes use float32** unless there is a specific reason otherwise.

8. **Raw DICOMs are never modified.** `DCM_frames/` is read-only. All pipeline outputs go to separate folders.

9. **Crop coordinates are currently hard-coded** (y=[100,700], x=[200,800]) in the segmentation CLI. They work for Patient 1 but are not guaranteed to generalize. Raise a warning before applying to a new patient without verifying the crop.

10. **Mask cleaning is currently disabled.** The morphological cleaning step in the segmentation CLI is commented out (line ~685). There is no `--use_cleaning` flag — re-enable by uncommenting that block. Do not silently re-enable it without flagging it.

---

## Coding conventions

- Use relative paths everywhere. Never hard-code absolute paths.
- All new scripts must accept `--input` / `--output` style CLI arguments via `argparse`.
- Log what you're doing: print the frame being processed, masks being saved, and any warnings.
- Do not silently overwrite existing masks. Warn if a mask file already exists and require `--overwrite` to proceed.
- Variable names: `image` for raw loaded array, `enhanced` after preprocessing, `mask` for binary output, `overlay` for visualization.
- Keep preprocessing, segmentation, and I/O in separate functions — not one large main block.
- Every script that modifies the dataset must update or be compatible with `metadata_labeled.csv`.

---

## Constraints — things to never do

- **Never modify files under `DCM_frames/`.**
- **Never commit raw patient data to a public repository.**
- **Never use random frame-level train/test splits.** Always split by patient or scan.
- **Never train a model using frames from the test patient**, even for sanity checks.
- **Never produce masks in float format.** Masks are always uint8 PNG (0/255).
- **Never rename the repository** mid-project without updating all internal path references.

---

## Evaluation plan (Project 1)

Metrics to compute on test set:
- Dice coefficient
- IoU (Jaccard index)
- Precision and recall
- Hausdorff distance (or mean contour distance) — important because the bone target is a thin line

Qualitative outputs required:
- Side-by-side overlay images (frame + predicted mask + ground-truth mask if available)
- Failure case examples with notes

---

## Report structure (Project 1)

The report is `Project1_Report_v1_docx.docx`. Chapters:
1. Introduction
2. Project Goals and Scope
3. Background and Literature Review
4. Materials and Methods
5. Evaluation Methodology
6. Results
7. Discussion
8. Conclusions and Future Work
9. Appendices

The report should clearly frame this as a proof-of-concept research pipeline, not a clinical tool.

---

## Literature reviewed (relevant papers)

| Paper | Relevance |
|-------|-----------|
| Arsenescu et al. 2023 | Closest analog: 2D US segmentation → 3D reconstruction, MultiResUNet |
| Victoria et al. 2023 | Real-time octree-based 3D US reconstruction |
| Dou et al. 2024 | Sensorless physics-guided DL reconstruction — future direction |
| Solberg et al. 2011 | Classical freehand 3D US reconstruction algorithms |

---

## Session startup checklist (for Claude Code)

1. Read this file (`CLAUDE.md`)
2. Read `progress.md` — report what is done, in progress, and blocked
3. Do not touch `DCM_frames/` for any reason
4. Confirm the task for this session before writing any code
