# Labeling Protocol – Jaw Bone Segmentation

- Use B-scan ultrasound frames of the facial/jaw region.
- "Bone" = bright, continuous cortical bone line of mandible/zygomatic arch.
- Exclude soft tissue, artifacts, isolated spots.

Procedure:
1. Open frame with `Bone Segmentation/Region Growing Segmentation/seg/ultrasound_bone_segmentation.py`.
2. Crop, enhance, and select several seeds along the bone.
3. Run region growing + mask cleaning + active contour.
4. Check overlay; if wrong, adjust seeds/params and rerun.
5. Save final mask as `frame_XXX_mask.png` in the same folder as the image.
6. Add row to `Dataset/metadata_labeled.csv`.

This will create consistent ground-truth masks for training and evaluation.
