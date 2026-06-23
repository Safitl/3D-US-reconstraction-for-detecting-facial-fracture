import argparse
import os

import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def save_single_frame(img: np.ndarray, out_path: str, normalize: bool = True):
    """Identical to dcm_to_png_batch.py — same normalization and save logic."""
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


def load_frames(input_path: str):
    """Load all frames from a DICOM file. Returns (base_name, frames_array)."""
    ds = pydicom.dcmread(input_path)
    arr = ds.pixel_array
    base = os.path.splitext(os.path.basename(input_path))[0]

    if arr.ndim == 2:
        return base, arr[np.newaxis]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return base, arr[np.newaxis]
    if arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            return base, np.moveaxis(arr, 0, -1)[np.newaxis]
        return base, arr
    if arr.ndim == 4:
        return base, arr
    raise ValueError(f"Unsupported pixel_array ndim={arr.ndim}, shape={arr.shape}")


def _to_display(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    f -= f.min()
    if f.max() > 0:
        f /= f.max()
    return f


def browse_and_select(frames: np.ndarray, base: str) -> list:
    """
    Interactive frame browser.
    Navigation: ← → arrow keys or Prev/Next buttons.
    Selection:  Space or Toggle Select button.
    Finish:     Enter, q, or Done button.
    Returns sorted list of selected frame indices.
    """
    N = len(frames)
    state = {"idx": 0, "selected": set()}

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.15)

    im = ax.imshow(_to_display(frames[0]), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    title = ax.set_title("")

    def update():
        idx = state["idx"]
        im.set_data(_to_display(frames[idx]))
        sel_tag = "  [SELECTED]" if idx in state["selected"] else ""
        title.set_text(
            f"{base}  —  Frame {idx} / {N - 1}{sel_tag}"
            f"   |   {len(state['selected'])} selected"
        )
        fig.canvas.draw_idle()

    update()

    ax_prev = plt.axes([0.08, 0.04, 0.15, 0.07])
    ax_next = plt.axes([0.25, 0.04, 0.15, 0.07])
    ax_sel  = plt.axes([0.43, 0.04, 0.22, 0.07])
    ax_done = plt.axes([0.67, 0.04, 0.22, 0.07])

    btn_prev = Button(ax_prev, "← Prev")
    btn_next = Button(ax_next, "Next →")
    btn_sel  = Button(ax_sel,  "Toggle Select")
    btn_done = Button(ax_done, "Done")

    def prev(_):
        state["idx"] = (state["idx"] - 1) % N
        update()

    def next_(_):
        state["idx"] = (state["idx"] + 1) % N
        update()

    def toggle(_):
        idx = state["idx"]
        if idx in state["selected"]:
            state["selected"].discard(idx)
        else:
            state["selected"].add(idx)
        update()

    def done(_):
        plt.close(fig)

    btn_prev.on_clicked(prev)
    btn_next.on_clicked(next_)
    btn_sel.on_clicked(toggle)
    btn_done.on_clicked(done)

    def on_key(event):
        if event.key == "right":
            next_(None)
        elif event.key == "left":
            prev(None)
        elif event.key == " ":
            toggle(None)
        elif event.key in {"q", "enter"}:
            done(None)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return sorted(state["selected"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract manually chosen frames from a single DICOM file. "
            "Output format is identical to dcm_to_png_batch.py: "
            "normalized PNG named {base}_f{idx:03d}.png."
        )
    )
    parser.add_argument(
        "--input_dcm", type=str, required=True,
        help="Path to the input DICOM file.",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="Folder to save extracted PNG frames.",
    )
    parser.add_argument(
        "--frames", type=int, nargs="+", default=None,
        help=(
            "Frame indices to extract (0-based). "
            "If omitted, opens an interactive browser."
        ),
    )
    parser.add_argument(
        "--no_normalize", action="store_true",
        help="Skip normalization (default is to normalize, matching dcm_to_png_batch.py).",
    )
    args = parser.parse_args()

    normalize = not args.no_normalize
    base, frames = load_frames(args.input_dcm)
    N = len(frames)
    print(f"Loaded {N} frame(s) from: {args.input_dcm}  (shape per frame: {frames[0].shape})")

    if args.frames is not None:
        selected = args.frames
        invalid = [i for i in selected if not (0 <= i < N)]
        if invalid:
            print(f"Invalid frame indices (out of range 0–{N - 1}): {invalid}")
            return
    else:
        print(
            "Opening interactive browser.\n"
            "  ← → or Prev/Next to navigate\n"
            "  Space or Toggle Select to mark a frame\n"
            "  Enter, q, or Done to finish\n"
        )
        selected = browse_and_select(frames, base)
        if not selected:
            print("No frames selected. Nothing saved.")
            return

    print(f"\nSaving {len(selected)} frame(s): {selected}")
    for idx in selected:
        out_path = os.path.join(args.output_folder, f"{base}_f{idx:03d}.png")
        save_single_frame(frames[idx], out_path, normalize=normalize)

    print(f"\nDone. {len(selected)} frame(s) saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
