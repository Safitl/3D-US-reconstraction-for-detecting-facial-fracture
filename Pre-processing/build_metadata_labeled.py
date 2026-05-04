import argparse
import csv
import re
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # Optional dependency for --check_sizes
    Image = None


IMAGE_PATTERN = re.compile(r"^(image_(\d+)(?:_f(\d+))?)\.png$", re.IGNORECASE)
MASK_PATTERN = re.compile(r"^(image_(\d+)(?:_f(\d+))?)_mask\.png$", re.IGNORECASE)
CSV_COLUMNS = [
    "patient_id",
    "scan_id",
    "frame_id",
    "image_path",
    "mask_path",
    "meta_path",
    "notes",
    "quality",
    "use_for_training",
    "split",
    "review_comment",
]


def extract_numeric_suffix(name: str) -> int:
    """Extract the trailing number from names like patient01 or frame_013."""
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else float("inf")


def to_posix_relative_path(path: Path, base_dir: Path) -> str:
    """Return a stable forward-slash relative path for CSV output."""
    return path.relative_to(base_dir).as_posix()


def get_image_size(path: Path):
    """Return image size as (width, height) when PIL is available."""
    with Image.open(path) as img:
        return img.size


def parse_image_key(file_name: str):
    """Parse image or mask naming into a stable key and metadata parts."""
    image_match = IMAGE_PATTERN.match(file_name)
    if image_match:
        key = image_match.group(1)
        scan_id = f"image_{image_match.group(2)}"
        frame_id = int(image_match.group(3)) if image_match.group(3) is not None else 0
        return key, scan_id, frame_id

    mask_match = MASK_PATTERN.match(file_name)
    if mask_match:
        key = mask_match.group(1)
        scan_id = f"image_{mask_match.group(2)}"
        frame_id = int(mask_match.group(3)) if mask_match.group(3) is not None else 0
        return key, scan_id, frame_id

    return None


def scan_and_collect_pairs(dataset_root: Path, check_sizes: bool):
    """Scan patient folders and collect valid image-mask pairs plus warnings."""
    rows = []
    warnings = []
    size_mismatches = []
    dataset_parent = dataset_root.parent

    for patient_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        patient_id = patient_dir.name
        image_dir = patient_dir / "IMG_frames"
        mask_dir = patient_dir / "Masks"

        if not image_dir.is_dir() and not mask_dir.is_dir():
            continue

        image_files = {}
        mask_files = {}

        if image_dir.is_dir():
            for path in image_dir.iterdir():
                if not path.is_file():
                    continue
                parsed = parse_image_key(path.name)
                if parsed and IMAGE_PATTERN.match(path.name):
                    key, scan_id, frame_id = parsed
                    image_files[key] = {
                        "path": path,
                        "scan_id": scan_id,
                        "frame_id": frame_id,
                    }

        if mask_dir.is_dir():
            for path in mask_dir.iterdir():
                if not path.is_file():
                    continue
                parsed = parse_image_key(path.name)
                if parsed and MASK_PATTERN.match(path.name):
                    key, scan_id, frame_id = parsed
                    mask_files[key] = {
                        "path": path,
                        "scan_id": scan_id,
                        "frame_id": frame_id,
                    }

        image_keys = set(image_files)
        mask_keys = set(mask_files)
        valid_keys = sorted(
            image_keys & mask_keys,
            key=lambda key: (
                extract_numeric_suffix(image_files[key]["scan_id"]),
                image_files[key]["frame_id"],
            ),
        )

        for frame_key in sorted(
            image_keys - mask_keys,
            key=lambda key: (
                extract_numeric_suffix(image_files[key]["scan_id"]),
                image_files[key]["frame_id"],
            ),
        ):
            warnings.append(
                f"Image without mask: {to_posix_relative_path(image_files[frame_key]['path'], dataset_parent)}"
            )

        for frame_key in sorted(
            mask_keys - image_keys,
            key=lambda key: (
                extract_numeric_suffix(mask_files[key]["scan_id"]),
                mask_files[key]["frame_id"],
            ),
        ):
            warnings.append(
                f"Mask without image: {to_posix_relative_path(mask_files[frame_key]['path'], dataset_parent)}"
            )

        for frame_key in valid_keys:
            image_info = image_files[frame_key]
            mask_info = mask_files[frame_key]
            image_path = image_info["path"]
            mask_path = mask_info["path"]

            if check_sizes:
                image_size = get_image_size(image_path)
                mask_size = get_image_size(mask_path)
                if image_size != mask_size:
                    mismatch_message = (
                        "Size mismatch: "
                        f"{to_posix_relative_path(image_path, dataset_parent)} {image_size} vs "
                        f"{to_posix_relative_path(mask_path, dataset_parent)} {mask_size}"
                    )
                    warnings.append(mismatch_message)
                    size_mismatches.append(mismatch_message)

            rows.append(
                {
                    "patient_id": patient_id,
                    "scan_id": image_info["scan_id"],
                    "frame_id": image_info["frame_id"],
                    "image_path": to_posix_relative_path(image_path, dataset_parent),
                    "mask_path": to_posix_relative_path(mask_path, dataset_parent),
                    "meta_path": to_posix_relative_path(
                        mask_path.with_name(mask_path.name.replace("_mask.png", "_meta.json")),
                        dataset_parent,
                    ),
                }
            )

    rows.sort(
        key=lambda row: (
            extract_numeric_suffix(row["patient_id"]),
            extract_numeric_suffix(row["scan_id"]),
            row["frame_id"],
        )
    )

    return rows, warnings, size_mismatches


def build_csv_rows(rows, notes_default: str):
    """Add the required metadata columns with default values."""
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "patient_id": row["patient_id"],
                "scan_id": row["scan_id"],
                "frame_id": row["frame_id"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "meta_path": row["meta_path"],
                "notes": notes_default,
                "quality": "",
                "use_for_training": "",
                "split": "",
                "review_comment": "",
            }
        )
    return csv_rows


def write_csv(output_csv: Path, rows):
    """Write metadata rows to CSV with the required column order."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows, warnings, size_mismatches):
    """Build summary lines for console output and optional report."""
    image_without_masks = sum(1 for warning in warnings if warning.startswith("Image without mask:"))
    masks_without_images = sum(1 for warning in warnings if warning.startswith("Mask without image:"))

    return [
        f"Labeled pairs found: {len(rows)}",
        f"Images without masks: {image_without_masks}",
        f"Masks without images: {masks_without_images}",
        f"Size mismatches: {len(size_mismatches)}",
    ]


def write_report(report_path: Path, warnings, summary_lines):
    """Write warnings and summary details to a plain-text report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    if warnings:
        lines.append("Warnings:")
        lines.extend(warnings)
    else:
        lines.append("Warnings:")
        lines.append("None")

    lines.append("")
    lines.append("Summary:")
    lines.extend(summary_lines)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build metadata CSV for a labeled ultrasound bone-segmentation dataset."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to the dataset root folder (for example, Dataset).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <dataset_root>/metadata_labeled.csv.",
    )
    parser.add_argument(
        "--notes_default",
        type=str,
        default="",
        help="Default text to populate the notes column.",
    )
    parser.add_argument(
        "--check_sizes",
        action="store_true",
        help="Check that each image and mask have the same dimensions.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Optional path to save warnings and summary as a text report.",
    )
    return parser.parse_args()


def main():
    """Run metadata generation from the command line."""
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")
    if args.check_sizes and Image is None:
        raise ImportError("PIL is required for --check_sizes. Install Pillow or omit this flag.")

    output_csv = args.output_csv.resolve() if args.output_csv else dataset_root / "metadata_labeled.csv"
    rows, warnings, size_mismatches = scan_and_collect_pairs(dataset_root, args.check_sizes)
    csv_rows = build_csv_rows(rows, args.notes_default)
    write_csv(output_csv, csv_rows)

    for warning in warnings:
        print(f"Warning: {warning}")

    summary_lines = build_summary(csv_rows, warnings, size_mismatches)
    for line in summary_lines:
        print(line)
    print(f"Metadata CSV written to: {output_csv}")

    if args.report_path:
        write_report(args.report_path.resolve(), warnings, summary_lines + [f"Output CSV: {output_csv}"])
        print(f"Report written to: {args.report_path.resolve()}")


if __name__ == "__main__":
    main()
