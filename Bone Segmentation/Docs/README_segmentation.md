# Segmentation Plan

Goal: Segment jaw bone in 2D ultrasound (B-scans) as input for later 3D reconstruction.

## Classical Stage

Code:
- `Bone Segmentation/Region Growing Segmentation/seg/ultrasound_bone_segmentation.py`

Pipeline:
- Load B-scan → crop ROI → CLAHE → Gaussian blur → interactive region growing → mask cleaning → active contour → save `frame_XXX_mask.png`.

Use this to create labeled data for a subset of frames (ground truth).

## Deep Learning Stage

Code (soon):
- `Bone Segmentation/Segmentation_DL/dataset.py`
- `Bone Segmentation/Segmentation_DL/unet_model.py`
- `Bone Segmentation/Segmentation_DL/train_unet.py`

Plan:
- Read pairs from `Dataset/metadata_labeled.csv`.
- Train 2D U-Net on labeled frames.
- Use trained model to speed up labeling and scale dataset.
