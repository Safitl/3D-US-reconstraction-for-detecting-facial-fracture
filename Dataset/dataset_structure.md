# Dataset Structure

Root: `Dataset/`

Each patient:
- `Dataset/patientXX/`, e.g. `patient01`

Each scan:
- `Dataset/patientXX/scanYY/`, e.g. `patient01/scan01`

Each scan contains:
- Frames: `frame_000.png`, `frame_001.png`, ...
- Masks (when labeled): `frame_000_mask.png`, `frame_001_mask.png`, ...

Masks:
- Same size as frame
- Single-channel 8-bit PNG
- 0 = background / non-bone
- 255 = bone

All labeled pairs are listed in `metadata_labeled.csv`
with columns: `patient_id, scan_id, frame_id, image_path, mask_path, notes`.

