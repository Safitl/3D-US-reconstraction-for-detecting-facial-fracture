import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

PARAM_SPECS = [
    ("max_seeds", int, "Maximum number of manual seeds"),
    ("tolerance", int, "Region-growing tolerance"),
    ("snap_window", int, "Seed snapping window"),
    ("snake_dilate", int, "Final snake dilation radius"),
    ("crop_y_min", int, "Crop top boundary"),
    ("crop_y_max", int, "Crop bottom boundary"),
    ("crop_x_min", int, "Crop left boundary"),
    ("crop_x_max", int, "Crop right boundary"),
    ("clahe_clip_limit", float, "CLAHE clip limit"),
    ("gaussian_kernel_y", int, "Gaussian kernel height"),
    ("gaussian_kernel_x", int, "Gaussian kernel width"),
    ("snake_sigma", float, "Snake edge-extraction sigma"),
    ("snake_alpha", float, "Snake alpha"),
    ("snake_beta", float, "Snake beta"),
    ("snake_gamma", float, "Snake gamma"),
    ("snake_w_line", float, "Snake line weight"),
    ("snake_w_edge", float, "Snake edge weight"),
    ("final_mask_mode", str, "Final mask mode: union or snake_only"),
    ("final_open_r", int, "Final opening radius for bump smoothing"),
    ("boundary_smooth", bool, "Optional final boundary smoothing"),
    ("boundary_smooth_sigma", float, "Sigma for optional final boundary smoothing"),
    ("y_band", int, "Cleaning band height"),
    ("min_area", int, "Cleaning minimum area"),
    ("open_r", int, "Cleaning opening radius"),
    ("close_w", int, "Cleaning closing width"),
]


def _find_repo_root(start: Path) -> Path:
    """
    Find the git repo root by walking up until a .git folder is found.
    Falls back to the script directory if not found.
    """
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    return cur


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
INDEX_PATH_DEFAULT = REPO_ROOT / "patient1_image_index.json"
CLI_PATH = (
    REPO_ROOT
    / "Bone Segmentation"
    / "Region Growing Segmentation"
    / "seg"
    / "ultrasound_bone_segmentation_cli.py"
)


def _load_index(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _build_index(img_dir: Path, mask_dir: Path) -> dict:
    images = sorted([p.name for p in img_dir.glob("*.png")])
    return {
        "dataset_root": str(img_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "mask_root": str(mask_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "images": images,
    }


def _ensure_index(index_path: Path, refresh: bool) -> dict:
    if refresh or (not index_path.exists()):
        # default Patient1 paths
        img_dir = REPO_ROOT / "Dataset" / "Patient1" / "IMG_frames"
        mask_dir = REPO_ROOT / "Dataset" / "Patient1" / "Masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {img_dir}")
        mask_dir.mkdir(parents=True, exist_ok=True)

        data = _build_index(img_dir, mask_dir)
        _save_index(index_path, data)
        return data

    data = _load_index(index_path)
    if "dataset_root" not in data or "mask_root" not in data or "images" not in data:
        raise ValueError(f"Index file missing required keys: {index_path}")

    # If the index file exists but has no images (e.g. placeholder file),
    # rebuild it from the folder so the runner works out of the box.
    if not data["images"]:
        img_dir = REPO_ROOT / Path(data["dataset_root"])
        mask_dir = REPO_ROOT / Path(data["mask_root"])
        if not img_dir.exists():
            # Fall back to the default Patient1 paths
            img_dir = REPO_ROOT / "Dataset" / "Patient1" / "IMG_frames"
            mask_dir = REPO_ROOT / "Dataset" / "Patient1" / "Masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        data = _build_index(img_dir, mask_dir)
        _save_index(index_path, data)

    return data


def _run_cli(cli_path: Path, image_path: Path, output_mask_path: Path, args: argparse.Namespace) -> int:
    python_exe = args.python_exe
    if not python_exe:
        # Prefer the active conda env python if available, otherwise fall back to current interpreter.
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate = Path(conda_prefix) / "python.exe"
            python_exe = str(candidate) if candidate.exists() else sys.executable
        else:
            python_exe = sys.executable

    cmd = [
        python_exe,
        str(cli_path),
        "--image_path",
        str(image_path),
        "--output_mask_path",
        str(output_mask_path),
        "--output_overlay_path",
        str(args.output_overlay_path),
        "--max_seeds",
        str(args.max_seeds),
        "--tolerance",
        str(args.tolerance),
        "--snap_window",
        str(args.snap_window),
        "--snake_dilate",
        str(args.snake_dilate),
        "--crop_y_min",
        str(args.crop_y_min),
        "--crop_y_max",
        str(args.crop_y_max),
        "--crop_x_min",
        str(args.crop_x_min),
        "--crop_x_max",
        str(args.crop_x_max),
        "--clahe_clip_limit",
        str(args.clahe_clip_limit),
        "--gaussian_kernel_y",
        str(args.gaussian_kernel_y),
        "--gaussian_kernel_x",
        str(args.gaussian_kernel_x),
        "--snake_sigma",
        str(args.snake_sigma),
        "--snake_alpha",
        str(args.snake_alpha),
        "--snake_beta",
        str(args.snake_beta),
        "--snake_gamma",
        str(args.snake_gamma),
        "--snake_w_line",
        str(args.snake_w_line),
        "--snake_w_edge",
        str(args.snake_w_edge),
        "--final_mask_mode",
        str(args.final_mask_mode),
        "--final_open_r",
        str(args.final_open_r),
        "--boundary_smooth_sigma",
        str(args.boundary_smooth_sigma),
        "--y_band",
        str(args.y_band),
        "--min_area",
        str(args.min_area),
        "--open_r",
        str(args.open_r),
        "--close_w",
        str(args.close_w),
    ]
    if args.show:
        cmd.append("--show")
    if args.boundary_smooth:
        cmd.append("--boundary_smooth")
    if args.reuse_meta_path:
        cmd.extend(["--reuse_meta_path", str(args.reuse_meta_path)])

    print("\nCommand:")
    print(" ".join([f"\"{c}\"" if " " in c else c for c in cmd]))
    print()
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _show_params(args: argparse.Namespace) -> None:
    print("\nCurrent segmentation parameters:")
    for name, _, description in PARAM_SPECS:
        print(f"  {name:<18} = {_format_value(getattr(args, name)):<10}  {description}")
    print(f"  show{'':<14} = {args.show}        Show intermediate figures")
    print()


def _set_param(args: argparse.Namespace, name: str, raw_value: str) -> bool:
    if name in {"show", "boundary_smooth"}:
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            setattr(args, name, True)
        elif value in {"0", "false", "no", "off"}:
            setattr(args, name, False)
        else:
            print(f"{name} expects one of: true/false, yes/no, on/off, 1/0\n")
            return False
        return True

    spec = next((spec for spec in PARAM_SPECS if spec[0] == name), None)
    if spec is None:
        print(f"Unknown parameter: {name}\n")
        return False

    caster = spec[1]
    if name == "final_mask_mode":
        value = raw_value.strip().lower()
        if value not in {"union", "snake_only"}:
            print("final_mask_mode expects one of: union, snake_only\n")
            return False
        setattr(args, name, value)
        return True

    try:
        value = caster(raw_value)
    except ValueError:
        print(f"Could not parse value for {name}: {raw_value}\n")
        return False

    setattr(args, name, value)
    return True


def _edit_params_interactively(args: argparse.Namespace) -> None:
    print("\nInteractive parameter editor")
    print("Press Enter to keep a value unchanged. Type 'q' to stop editing.\n")

    for name, _, description in PARAM_SPECS:
        current = _format_value(getattr(args, name))
        raw = input(f"{name} [{current}] - {description}: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print()
            return
        if raw == "":
            continue
        _set_param(args, name, raw)

    raw_show = input(f"show [{args.show}] - Show intermediate figures: ").strip()
    if raw_show and raw_show.lower() not in {"q", "quit", "exit"}:
        _set_param(args, "show", raw_show)
    print()


def _resolve_meta_path(mask_path: Path) -> Path:
    return mask_path.with_name(mask_path.stem.replace("_mask", "") + "_meta.json")


def _resolve_overlay_path(mask_path: Path) -> Path:
    return mask_path.with_name(mask_path.stem.replace("_mask", "") + "_overlay.png")


def _load_saved_seed_count(meta_path: Path):
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        return None

    num_seeds = metadata.get("num_seeds")
    try:
        return int(num_seeds)
    except (TypeError, ValueError):
        return None


def _load_saved_crop_box(meta_path: Path):
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        return None

    crop_box = metadata.get("crop_box")
    if not isinstance(crop_box, list) or len(crop_box) != 4:
        return None

    try:
        return [int(value) for value in crop_box]
    except (TypeError, ValueError):
        return None


def _choose_seed_mode(args: argparse.Namespace, meta_path: Path) -> None:
    args.reuse_meta_path = ""
    saved_seed_count = _load_saved_seed_count(meta_path)
    current_crop_box = [args.crop_x_min, args.crop_y_min, args.crop_x_max, args.crop_y_max]
    saved_crop_box = _load_saved_crop_box(meta_path)

    if saved_seed_count is None:
        if meta_path.exists():
            print("Saved metadata exists, but no reusable seed set was found. Please enter new seeds.\n")
        else:
            print("No saved seed set found for this image. Please enter new seeds.\n")
        return

    if saved_seed_count != args.max_seeds:
        print(
            "Saved seed set found, but its size "
            f"({saved_seed_count}) does not match current max_seeds ({args.max_seeds}). "
            "Please enter new seeds.\n"
        )
        return

    if saved_crop_box is not None and saved_crop_box != current_crop_box:
        print(
            "Saved seed set found, but its crop box "
            f"({saved_crop_box}) does not match the current crop ({current_crop_box}). "
            "Please enter new seeds.\n"
        )
        return

    while True:
        choice = input(
            f"Saved seed set found ({saved_seed_count} seeds). "
            "Use it and skip clicking? [Y/n]: "
        ).strip().lower()
        if choice in {"", "y", "yes"}:
            args.reuse_meta_path = meta_path
            print(f"Reusing saved seeds from: {meta_path}\n")
            return
        if choice in {"n", "no"}:
            print("Using new manual seeds for this run.\n")
            return
        print("Please answer y or n.\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Patient1 segmentation by simple image index (no filename copy/paste)."
    )
    parser.add_argument(
        "--index_json",
        type=str,
        default=str(INDEX_PATH_DEFAULT),
        help="Path to the index JSON file (default: ./patient1_image_index.json).",
    )
    parser.add_argument(
        "--refresh_index",
        action="store_true",
        help="Rebuild index from Dataset/Patient1/IMG_frames and overwrite JSON.",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default="",
        help="Python executable to run the segmentation CLI with. "
        "If omitted, uses the active conda env python when available, otherwise the current python.",
    )

    # Pass-through defaults (same as your CLI defaults, but user can override here)
    parser.add_argument("--max_seeds", type=int, default=8)
    parser.add_argument("--tolerance", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--y_band", type=int, default=35)
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--open_r", type=int, default=1)
    parser.add_argument("--close_w", type=int, default=25)
    parser.add_argument("--snap_window", type=int, default=7)
    parser.add_argument("--snake_dilate", type=int, default=3)
    parser.add_argument("--crop_y_min", type=int, default=100)
    parser.add_argument("--crop_y_max", type=int, default=700)
    parser.add_argument("--crop_x_min", type=int, default=200)
    parser.add_argument("--crop_x_max", type=int, default=800)
    parser.add_argument("--clahe_clip_limit", type=float, default=0.01)
    parser.add_argument("--gaussian_kernel_y", type=int, default=7)
    parser.add_argument("--gaussian_kernel_x", type=int, default=7)
    parser.add_argument("--snake_sigma", type=float, default=1.0)
    parser.add_argument("--snake_alpha", type=float, default=0.0015)
    parser.add_argument("--snake_beta", type=float, default=0.3)
    parser.add_argument("--snake_gamma", type=float, default=0.01)
    parser.add_argument("--snake_w_line", type=float, default=0.0)
    parser.add_argument("--snake_w_edge", type=float, default=1.0)
    parser.add_argument("--final_mask_mode", type=str, default="union")
    parser.add_argument("--final_open_r", type=int, default=0)
    parser.add_argument("--boundary_smooth", action="store_true")
    parser.add_argument("--boundary_smooth_sigma", type=float, default=1.0)

    args = parser.parse_args()
    args.reuse_meta_path = ""
    args.output_overlay_path = ""

    index_path = Path(args.index_json).expanduser()
    index = _ensure_index(index_path, refresh=args.refresh_index)

    img_root = REPO_ROOT / Path(index["dataset_root"])
    mask_root = REPO_ROOT / Path(index["mask_root"])
    images = index["images"]

    if not images:
        print(f"No PNGs found in: {img_root}")
        return 1

    print(f"Loaded {len(images)} images from: {img_root}")
    print("Enter an image index to run, 'params' to view settings, 'set <name> <value>' to change one, 'edit' to walk through all settings, or 'q' to quit.\n")

    while True:
        raw = input(f"Index (1..{len(images)} / q): ").strip()
        s = raw.lower()
        if s in {"q", "quit", "exit"}:
            return 0
        if s in {"params", "p", "settings"}:
            _show_params(args)
            continue
        if s in {"edit", "e"}:
            _edit_params_interactively(args)
            _show_params(args)
            continue
        if s.startswith("set "):
            parts = raw.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: set <name> <value>\n")
                continue
            if _set_param(args, parts[1], parts[2]):
                print(f"Updated {parts[1]} = {_format_value(getattr(args, parts[1]))}\n")
            continue

        try:
            idx = int(s)
        except ValueError:
            print("Please enter a number, or 'q'.\n")
            continue

        if not (1 <= idx <= len(images)):
            print(f"Index out of range: {idx}\n")
            continue

        name = images[idx - 1]
        image_path = img_root / name
        output_mask_path = mask_root / (Path(name).stem + "_mask.png")
        overlay_path = _resolve_overlay_path(output_mask_path)
        meta_path = _resolve_meta_path(output_mask_path)
        args.output_overlay_path = overlay_path

        print(f"\nSelected index: {idx}/{len(images)}")
        print(f"Image filename: {name}")
        print(f"Image path:     {image_path}")
        print(f"Output mask:    {output_mask_path}")
        print(f"Overlay path:   {overlay_path}")
        print(f"Metadata path:  {meta_path}")
        _choose_seed_mode(args, meta_path)
        rc = _run_cli(CLI_PATH, image_path, output_mask_path, args)
        print(f"\nFinished with exit code: {rc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
