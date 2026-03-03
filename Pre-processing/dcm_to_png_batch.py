import os
import argparse

import numpy as np
import cv2
import pydicom

# Script to run this file from command line (PowerShell):
# & "C:\Users\safit\.conda\envs\ultrasound3d\python.exe" `
# >>   "Pre-processing\dcm_to_png_batch.py" `
# >>   --input_folder "C:\Users\safit\OneDrive\GitHub\3D-US-reconstraction-for-detecting-facial-fracture\Dataset\Patient1\DCM_frames" `
# >>   --output_folder "C:\Users\safit\OneDrive\GitHub\3D-US-reconstraction-for-detecting-facial-fracture\Dataset\Patient1\IMG_frames" `
# >>   --max_frames_per_dcm 20

def save_single_frame(img: np.ndarray, out_path: str, normalize: bool = True):
    """Save one 2D or 3D (RGB) frame as PNG."""
    img = img.astype(np.float32)

    if normalize:
        img_min = img.min()
        img -= img_min
        img_max = img.max()
        if img_max > 0:
            img /= img_max
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwrite(out_path, img)
    if not ok:
        print(f"!!! Failed to write PNG: {out_path}")
    else:
        print(f"Saved: {out_path}")

def convert_dcm(input_path: str,
                output_folder: str,
                max_frames_per_dcm: int = 10,
                normalize: bool = True):
    """Convert one DICOM to one or many PNGs."""
    ds = pydicom.dcmread(input_path)
    arr = ds.pixel_array
    base = os.path.splitext(os.path.basename(input_path))[0]

    print(f"{base}.dcm pixel_array shape: {arr.shape}, dtype: {arr.dtype}")

    # 2D: (H, W) or 3D: (H, W, C)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 3):
        out_path = os.path.join(output_folder, f"{base}.png")
        save_single_frame(arr, out_path, normalize=normalize)
        return

    # 3D: (N, H, W) or (C, H, W)
    if arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            # channels-first (3, H, W)
            frame = np.moveaxis(arr, 0, -1)  # -> (H, W, 3)
            out_path = os.path.join(output_folder, f"{base}.png")
            save_single_frame(frame, out_path, normalize=normalize)
            return
        else:
            # (N, H, W): cine grayscale
            N = arr.shape[0]
            indices = np.linspace(0, N - 1,
                                  num=min(max_frames_per_dcm, N),
                                  dtype=int)
            for idx in indices:
                frame = arr[idx, :, :]
                out_path = os.path.join(output_folder,
                                        f"{base}_f{idx:03d}.png")
                save_single_frame(frame, out_path, normalize=normalize)
            return

    # 4D: (N, H, W, C) cine RGB – this is your big files
    if arr.ndim == 4:
        N = arr.shape[0]
        indices = np.linspace(0, N - 1,
                              num=min(max_frames_per_dcm, N),
                              dtype=int)
        for idx in indices:
            frame = arr[idx, :, :, :]   # (H, W, C)
            out_path = os.path.join(output_folder,
                                    f"{base}_f{idx:03d}.png")
            save_single_frame(frame, out_path, normalize=normalize)
        return

    raise ValueError(f"Unsupported pixel_array ndim={arr.ndim}, shape={arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM ultrasound frames to PNG.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing .dcm files.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder to save PNG images.")
    parser.add_argument("--max_frames_per_dcm", type=int, default=10,
                        help="Max number of frames to extract per multi-frame DICOM.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    for fname in os.listdir(args.input_folder):
        if not fname.lower().endswith(".dcm"):
            continue
        in_path = os.path.join(args.input_folder, fname)
        convert_dcm(in_path,
                    args.output_folder,
                    max_frames_per_dcm=args.max_frames_per_dcm,
                    normalize=True)


if __name__ == "__main__":
    main()
