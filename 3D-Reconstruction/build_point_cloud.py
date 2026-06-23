#!/usr/bin/env python3
"""
Project B — Milestone 1 (v2): Sparse 3D bone-surface point cloud from 2D masks.

For each labeled frame in a cine sweep, extract the bone-surface contour from
the binary mask, map it back to original image coordinates using the crop_box
stored in the meta JSON, then assign a Z-coordinate based on frame order and
an assumed inter-frame spacing delta (Δ).

Coordinate convention (used everywhere in this script):
    X  =  US lateral image coordinate   (pixel column in original image)
    Y  =  US depth coordinate            (pixel row in original image; increases downward)
    Z  =  sweep-axis / frame-order       (frame_order_index × Δ)

No external probe tracking is available.  Z is therefore a relative/estimated
coordinate, NOT a true physical depth along the probe trajectory.  This is a
"baseline sparse reconstruction" suitable for CT-guided ICP alignment in later
milestones, not an independent sensorless 3D reconstruction.

─────────────────────────────────────────────────────────────────────────────
Command examples (run from project root):

  # Single delta run — produces PLY + overview figure:
  python "3D-Reconstruction/build_point_cloud.py" \\
      --scan_id image_172731958799 --patient_dir Patient2 --delta 1.0

  # Delta sensitivity run — produces one PLY per delta:
  python "3D-Reconstruction/build_point_cloud.py" \\
      --scan_id image_172731958799 --patient_dir Patient2 \\
      --delta 0.5 1.0 2.0 5.0 10.0

  # With sweep reversal (turnaround at DICOM frame 156):
  python "3D-Reconstruction/build_point_cloud.py" \\
      --scan_id image_172731958799 --patient_dir Patient2 \\
      --delta 1.0 --midframe 156

  # QC figure only — inspect curve extraction without writing PLY:
  python "3D-Reconstruction/build_point_cloud.py" \\
      --scan_id image_172731958799 --patient_dir Patient2 \\
      --qc --no_ply --no_figure

  # All outputs with skeleton extraction method:
  python "3D-Reconstruction/build_point_cloud.py" \\
      --scan_id image_172731958799 --patient_dir Patient2 \\
      --delta 1.0 --extraction_method skeleton --qc
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False

# ─── constants ────────────────────────────────────────────────────────────────

_DEFAULT_CROP_BOX: List[int] = [200, 100, 800, 700]  # [x_min, y_min, x_max, y_max]

EXTRACTION_METHODS = ("top_boundary", "all_mask_pixels", "skeleton")


# ─── per-frame data container ─────────────────────────────────────────────────

@dataclass
class FrameData:
    order_idx: int                        # position in sorted frame list
    frame_num: int                        # DICOM frame index (from filename)
    pts_working: np.ndarray               # (N, 2) float32  [row, col] in 600×600 working space
    pts_orig: np.ndarray                  # (N, 2) float32  [row, col] in original image space
    mask: np.ndarray                      # (H, W) uint8  binary mask (for QC)
    img_crop: Optional[np.ndarray]        # (H, W) uint8  grayscale cropped US frame (for QC)
    crop_box: List[int]                   # [x_min, y_min, x_max, y_max]


# ─── file helpers ─────────────────────────────────────────────────────────────

def _parse_frame_index(filename: str) -> int:
    """Extract numeric frame index from a name like image_ID_f015_mask.png."""
    m = re.search(r"_f(\d+)_mask", filename)
    return int(m.group(1)) if m else 0


def _find_mask_files(masks_dir: Path, scan_id: str) -> List[Path]:
    """All manual mask PNGs for *scan_id*, sorted by ascending frame index."""
    files = list(masks_dir.glob(f"{scan_id}_f*_mask.png"))
    files.sort(key=lambda p: _parse_frame_index(p.name))
    return files


def _find_img_file(imgs_dir: Path, mask_path: Path) -> Optional[Path]:
    """Derive the US frame path from its corresponding mask path."""
    img_name = mask_path.name.replace("_mask.png", ".png")
    p = imgs_dir / img_name
    return p if p.exists() else None


def _load_crop_box(meta_path: Path) -> List[int]:
    """Read crop_box from meta JSON; fall back to DEFAULT_CROP_BOX on failure."""
    if meta_path.exists():
        try:
            with open(meta_path) as fh:
                return json.load(fh).get("crop_box", list(_DEFAULT_CROP_BOX))
        except Exception:
            pass
    return list(_DEFAULT_CROP_BOX)


# ─── surface extraction ───────────────────────────────────────────────────────

def _extract_bone_surface(mask_gray: np.ndarray, method: str) -> np.ndarray:
    """
    Extract bone-interface points from a binary mask.

    Returns (N, 2) float32 array of [row, col] in the mask's own coordinate space.

    Methods:
        top_boundary   — topmost foreground pixel per column; represents the
                         outer cortical interface seen by the probe (default).
        all_mask_pixels — every foreground pixel; denser but noisier for thin masks.
        skeleton        — morphological skeleton of the mask; requires scikit-image.
    """
    if method == "top_boundary":
        pts: List[Tuple[float, float]] = []
        for col in range(mask_gray.shape[1]):
            rows = np.where(mask_gray[:, col] > 127)[0]
            if rows.size > 0:
                pts.append((float(rows[0]), float(col)))
        return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)

    if method == "all_mask_pixels":
        rows, cols = np.where(mask_gray > 127)
        if rows.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.column_stack([rows, cols]).astype(np.float32)

    if method == "skeleton":
        if not _SKIMAGE_OK:
            raise RuntimeError(
                "scikit-image is required for the 'skeleton' method.  "
                "Install it with:  pip install scikit-image"
            )
        binary = mask_gray > 127
        skel = _sk_skeletonize(binary)
        rows, cols = np.where(skel)
        if rows.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.column_stack([rows, cols]).astype(np.float32)

    raise ValueError(f"Unknown extraction_method '{method}'.  "
                     f"Choose from: {EXTRACTION_METHODS}")


def _working_to_original(pts_rc: np.ndarray, crop_box: List[int]) -> np.ndarray:
    """
    Map [row, col] from the cropped working space to original image space.

    crop_box = [x_min, y_min, x_max, y_max] in the original image:
        row_orig = row_working + y_min  (= crop_box[1])
        col_orig = col_working + x_min  (= crop_box[0])
    """
    if pts_rc.shape[0] == 0:
        return pts_rc
    out = pts_rc.copy()
    out[:, 0] += crop_box[1]   # row offset = y_min
    out[:, 1] += crop_box[0]   # col offset = x_min
    return out


# ─── frame extraction (runs once; reused across all delta values) ─────────────

def extract_frames(
    mask_files: List[Path],
    masks_dir: Path,
    imgs_dir: Path,
    extraction_method: str,
) -> List[FrameData]:
    """
    Load all mask files, extract bone-surface points, and collect per-frame data.

    This is separated from cloud assembly so that multiple delta values can reuse
    the same extracted points without re-reading the masks.
    """
    frames: List[FrameData] = []
    for order_idx, mask_path in enumerate(mask_files):
        meta_path = mask_path.parent / mask_path.name.replace("_mask.png", "_meta.json")
        crop_box = _load_crop_box(meta_path)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARNING: cannot load {mask_path.name} — skipping")
            continue

        pts_working = _extract_bone_surface(mask, extraction_method)
        if pts_working.shape[0] == 0:
            print(f"  WARNING: no foreground in {mask_path.name} — skipping")
            continue

        pts_orig = _working_to_original(pts_working, crop_box)

        img_path = _find_img_file(imgs_dir, mask_path)
        img_crop: Optional[np.ndarray] = None
        if img_path is not None:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                x0, y0, x1, y1 = crop_box
                img_crop = img[y0:y1, x0:x1]

        frame_num = _parse_frame_index(mask_path.name)
        frames.append(FrameData(
            order_idx=order_idx,
            frame_num=frame_num,
            pts_working=pts_working,
            pts_orig=pts_orig,
            mask=mask,
            img_crop=img_crop,
            crop_box=crop_box,
        ))

        print(f"  [{order_idx:02d}] frame={frame_num:4d}  "
              f"pts={len(pts_working):4d}  "
              f"X=[{pts_orig[:,1].min():.0f},{pts_orig[:,1].max():.0f}]  "
              f"Y=[{pts_orig[:,0].min():.0f},{pts_orig[:,0].max():.0f}]")

    return frames


# ─── cloud assembly for a given delta ────────────────────────────────────────

def assemble_cloud(
    frames: List[FrameData],
    delta: float,
    midframe: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble 3D point cloud from extracted frames and a given delta.

    Returns:
        xyz   : (N, 3) float32  — X=lateral, Y=depth, Z=sweep
        order : (N,)  int32    — frame order index for each point
    """
    all_xyz: List[np.ndarray] = []
    all_order: List[np.ndarray] = []
    frame_numbers = [fd.frame_num for fd in frames]

    for fd in frames:
        x = fd.pts_orig[:, 1].astype(np.float32)   # col → X (lateral)
        y = fd.pts_orig[:, 0].astype(np.float32)   # row → Y (depth)
        z = np.full(len(x), fd.order_idx * delta, dtype=np.float32)

        all_xyz.append(np.column_stack([x, y, z]))
        all_order.append(np.full(len(x), fd.order_idx, dtype=np.int32))

    xyz = np.vstack(all_xyz)
    order = np.concatenate(all_order)

    if midframe is not None:
        mid_order = next(
            (i for i, fn in enumerate(frame_numbers) if fn >= midframe),
            len(frame_numbers) // 2,
        )
        backward = order >= mid_order
        z_mid = float(mid_order) * delta
        xyz[backward, 2] = 2.0 * z_mid - xyz[backward, 2]
        print(f"  Sweep reversal: DICOM frame >= {midframe} "
              f"(order >= {mid_order}) mirrored around Z = {z_mid:.1f}")

    return xyz, order


# ─── PLY writer ───────────────────────────────────────────────────────────────

def write_ply(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    """Write an ASCII PLY file.  rgb: (N, 3) uint8, optional."""
    n = len(xyz)
    has_color = rgb is not None
    header_lines = [
        "ply", "format ascii 1.0", f"element vertex {n}",
        "property float x", "property float y", "property float z",
    ]
    if has_color:
        header_lines += ["property uchar red", "property uchar green", "property uchar blue"]
    header_lines.append("end_header")

    with open(path, "w") as fh:
        fh.write("\n".join(header_lines) + "\n")
        for i in range(n):
            line = f"{xyz[i,0]:.3f} {xyz[i,1]:.3f} {xyz[i,2]:.3f}"
            if has_color:
                line += f" {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}"
            fh.write(line + "\n")


# ─── summary CSV ─────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "scan_id", "patient_dir", "delta", "num_frames", "num_points",
    "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
    "extraction_method", "reversal_used", "midframe",
]


def append_to_summary_csv(csv_path: Path, row: Dict) -> None:
    """Append one row to the summary CSV, creating the file with headers if absent."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── overview figure ──────────────────────────────────────────────────────────

def save_overview_figure(
    xyz: np.ndarray,
    fig_path: Path,
    scan_id: str,
    delta: float,
    extraction_method: str,
    midframe: Optional[int],
) -> None:
    """
    Two-panel overview figure.
    Left:   X – Z  (top view: lateral vs. sweep axis)
    Right:  Y – Z  (side view: depth vs. sweep axis)
    Points colored by frame order (viridis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    z_norm = (xyz[:, 2] - xyz[:, 2].min()) / (xyz[:, 2].max() - xyz[:, 2].min() + 1e-9)
    colors = plt.cm.viridis(z_norm)

    ax = axes[0]
    ax.scatter(xyz[:, 2], xyz[:, 0], c=colors, s=0.5, linewidths=0, alpha=0.55)
    ax.set_xlabel(f"Z — sweep axis  (Δ = {delta:g})", fontsize=10)
    ax.set_ylabel("X — US lateral (px)", fontsize=10)
    ax.set_title("Top view:  X – Z", fontsize=11)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(xyz[:, 2], xyz[:, 1], c=colors, s=0.5, linewidths=0, alpha=0.55)
    ax.set_xlabel(f"Z — sweep axis  (Δ = {delta:g})", fontsize=10)
    ax.set_ylabel("Y — US depth (px)", fontsize=10)
    ax.set_title("Side view:  Y – Z", fontsize=11)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Frame order (0 = first, 1 = last)", shrink=0.8)

    reversal_note = f"  |  reversal @ frame {midframe}" if midframe is not None else ""
    fig.suptitle(
        f"Sparse 3D bone-surface cloud  —  {scan_id}\n"
        f"method: {extraction_method}  |  Δ = {delta:g}{reversal_note}  "
        f"|  {len(xyz):,} pts",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.88)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─── QC figure ────────────────────────────────────────────────────────────────

def save_qc_figure(
    frames: List[FrameData],
    fig_path: Path,
    scan_id: str,
    extraction_method: str,
) -> None:
    """
    QC figure: 5 representative frames (0%, 25%, 50%, 75%, 100%) × 3 columns.
    Col 0: original cropped US frame (grayscale)
    Col 1: binary bone mask
    Col 2: extracted bone surface overlaid on the US frame (cyan dots)
    """
    n = len(frames)
    if n == 0:
        print("  WARNING: no frames available for QC figure")
        return

    # Select 5 representative indices
    indices = sorted(set([
        0,
        max(0, round(n * 0.25) - 1) if n > 3 else 0,
        max(0, round(n * 0.50) - 1),
        max(0, round(n * 0.75) - 1) if n > 3 else n - 1,
        n - 1,
    ]))

    selected = [frames[i] for i in indices]
    n_rows = len(selected)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "US frame (cropped)",
        "Bone mask",
        f"Extracted surface\n({extraction_method})",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold", pad=6)

    for row_i, fd in enumerate(selected):
        h, w = fd.mask.shape

        # ── column 0: original US frame (or grey placeholder) ──
        ax = axes[row_i, 0]
        if fd.img_crop is not None:
            ax.imshow(fd.img_crop, cmap="gray", vmin=0, vmax=255, aspect="equal")
        else:
            ax.imshow(np.full((h, w), 128, dtype=np.uint8),
                      cmap="gray", vmin=0, vmax=255, aspect="equal")
            ax.text(w // 2, h // 2, "frame not found",
                    ha="center", va="center", color="white", fontsize=7)
        ax.set_ylabel(f"frame {fd.frame_num}\n(order {fd.order_idx})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # ── column 1: binary mask ──
        ax = axes[row_i, 1]
        ax.imshow(fd.mask, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # ── column 2: extracted surface overlay on US frame ──
        ax = axes[row_i, 2]
        bg = fd.img_crop if fd.img_crop is not None else np.full((h, w), 64, dtype=np.uint8)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255, aspect="equal")
        pts = fd.pts_working
        if pts.shape[0] > 0:
            # pts_working: [row, col] → plot as (col, row) in image axes
            ax.scatter(pts[:, 1], pts[:, 0],
                       s=0.8, c="cyan", linewidths=0, alpha=0.9, label=f"{len(pts)} pts")
            ax.legend(loc="upper right", fontsize=7, markerscale=4,
                      framealpha=0.6, handlelength=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Curve extraction QC  —  {scan_id}\n"
        f"5 representative frames  |  method: {extraction_method}",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.92)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  QC figure : {fig_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a sparse 3D bone-surface point cloud from 2D binary masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--scan_id", required=True,
                    help="Scan ID, e.g. image_172731958799")
    ap.add_argument("--patient_dir", default="Patient2",
                    help="Patient folder under Dataset/  (default: Patient2)")
    ap.add_argument("--base_dir", default=".",
                    help="Project root directory  (default: .)")

    ap.add_argument("--delta", type=float, nargs="+", default=[1.0], metavar="D",
                    help="Assumed inter-frame spacing along the sweep axis. "
                         "Accepts one or more values for a sensitivity run.  "
                         "Example: --delta 0.5 1.0 2.0 5.0  (default: 1.0)")
    ap.add_argument("--extraction_method", default="top_boundary",
                    choices=list(EXTRACTION_METHODS),
                    help="How to derive bone-surface points from each mask.  "
                         "top_boundary (default): topmost foreground pixel per column — "
                         "represents the outer cortical interface.  "
                         "all_mask_pixels: every foreground pixel.  "
                         "skeleton: morphological skeleton (requires scikit-image).")
    ap.add_argument("--midframe", type=int, default=None,
                    help="DICOM frame index of the sweep turnaround.  "
                         "Frames at or beyond this index are mirrored in Z so that both "
                         "the forward and backward passes occupy the same anatomical range.")

    ap.add_argument("--output_ply", default=None,
                    help="Custom PLY output path (only used for single-delta runs; "
                         "ignored when multiple deltas are given).")
    ap.add_argument("--qc", action="store_true",
                    help="Generate a curve-extraction QC figure.")
    ap.add_argument("--no_ply", action="store_true",
                    help="Skip writing PLY files (useful with --qc only).")
    ap.add_argument("--no_figure", action="store_true",
                    help="Skip saving the overview PNG(s).")
    ap.add_argument("--no_csv", action="store_true",
                    help="Skip writing the summary CSV.")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    patient_path = base / "Dataset" / args.patient_dir
    masks_dir = patient_path / "Masks"
    imgs_dir = patient_path / "IMG_frames"
    out_dir = base / "3D-Reconstruction" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not masks_dir.exists():
        print(f"ERROR: Masks directory not found: {masks_dir}", file=sys.stderr)
        sys.exit(1)

    # ── locate mask files ──
    mask_files = _find_mask_files(masks_dir, args.scan_id)
    if not mask_files:
        print(f"ERROR: No mask files found for '{args.scan_id}' in {masks_dir}",
              file=sys.stderr)
        sys.exit(1)

    deltas: List[float] = args.delta
    multi_delta = len(deltas) > 1

    print(f"Scan              : {args.scan_id}")
    print(f"Patient           : {args.patient_dir}")
    print(f"Frames found      : {len(mask_files)}")
    print(f"Extraction method : {args.extraction_method}")
    print(f"Delta(s)          : {deltas}")
    print(f"Turnaround frame  : "
          f"{args.midframe if args.midframe is not None else 'none (linear stack)'}")
    if multi_delta:
        print(f"Mode              : delta sensitivity  ({len(deltas)} runs)")
    print()

    if multi_delta and args.output_ply is not None:
        print("NOTE: --output_ply is ignored in multi-delta mode; "
              "filenames are set automatically.")

    # ── extract per-frame data once (shared across all delta values) ──
    print("Extracting bone surface points …")
    frames = extract_frames(mask_files, masks_dir, imgs_dir, args.extraction_method)
    if not frames:
        print("ERROR: No valid frames extracted.", file=sys.stderr)
        sys.exit(1)
    print(f"  -> {len(frames)} frames, "
          f"{sum(len(fd.pts_working) for fd in frames):,} surface points total\n")

    # ── QC figure (independent of delta) ──
    if args.qc:
        qc_dir = out_dir / "qc"
        qc_path = qc_dir / f"{args.scan_id}_curve_extraction_qc.png"
        print("Generating QC figure …")
        save_qc_figure(frames, qc_path, args.scan_id, args.extraction_method)

    if args.no_ply and args.no_figure and args.no_csv:
        print("All outputs disabled — done.")
        return

    # ── per-delta loop ──
    csv_path = out_dir / "metrics" / "pointcloud_summary.csv"

    for delta in deltas:
        tag = f"{args.scan_id}_delta_{delta:g}"
        print(f"--- delta = {delta:g} " + "-" * 45)

        xyz, order = assemble_cloud(frames, delta, args.midframe)

        # ── colors (viridis by frame order) ──
        n_frames = order.max() + 1
        colors = (plt.cm.viridis(order / max(n_frames - 1, 1))[:, :3] * 255).astype(np.uint8)

        # ── PLY ──
        if not args.no_ply:
            if not multi_delta and args.output_ply is not None:
                ply_path = Path(args.output_ply)
                ply_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                ply_path = out_dir / f"{tag}.ply"
            write_ply(ply_path, xyz, colors)
            print(f"  PLY      : {ply_path}  ({len(xyz):,} pts)")

        # ── overview figure ──
        if not args.no_figure:
            fig_path = out_dir / f"{tag}_overview.png"
            save_overview_figure(
                xyz, fig_path, args.scan_id, delta,
                args.extraction_method, args.midframe,
            )
            print(f"  Figure   : {fig_path}")

        # ── summary CSV row ──
        if not args.no_csv:
            row = {
                "scan_id":            args.scan_id,
                "patient_dir":        args.patient_dir,
                "delta":              delta,
                "num_frames":         len(frames),
                "num_points":         len(xyz),
                "x_min":              f"{xyz[:,0].min():.1f}",
                "x_max":              f"{xyz[:,0].max():.1f}",
                "y_min":              f"{xyz[:,1].min():.1f}",
                "y_max":              f"{xyz[:,1].max():.1f}",
                "z_min":              f"{xyz[:,2].min():.1f}",
                "z_max":              f"{xyz[:,2].max():.1f}",
                "extraction_method":  args.extraction_method,
                "reversal_used":      args.midframe is not None,
                "midframe":           args.midframe if args.midframe is not None else "",
            }
            append_to_summary_csv(csv_path, row)

        # ── per-delta extent summary ──
        print(
            f"  Extent (px):\n"
            f"    X (US lateral) : {xyz[:,0].min():.0f} - {xyz[:,0].max():.0f}  "
            f"range = {xyz[:,0].max()-xyz[:,0].min():.0f}\n"
            f"    Y (US depth)   : {xyz[:,1].min():.0f} - {xyz[:,1].max():.0f}  "
            f"range = {xyz[:,1].max()-xyz[:,1].min():.0f}\n"
            f"    Z (sweep axis) : {xyz[:,2].min():.1f} - {xyz[:,2].max():.1f}  "
            f"range = {xyz[:,2].max()-xyz[:,2].min():.1f}"
        )
        print()

    if not args.no_csv:
        print(f"Summary CSV : {csv_path}")

    print(
        "\nNext steps:\n"
        "  1. Open a PLY in MeshLab / CloudCompare to inspect the point cloud.\n"
        "  2. Use --qc to verify that extracted curves match the bone interface.\n"
        "  3. Set --midframe to fold the forward/backward sweep halves.\n"
        "  4. Calibrate --delta to physical mm/frame using DICOM pixel spacing.\n"
        "  5. Proceed to Milestone 2: CT surface extraction."
    )


if __name__ == "__main__":
    main()
