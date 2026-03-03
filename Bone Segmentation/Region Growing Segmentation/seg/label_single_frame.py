import argparse
import os
import csv
import sys

from ultrasound_bone_segmentation_cli import main as seg_main  # adjust name if different


def append_to_metadata(csv_path, row_dict):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "scan_id",
                "frame_id",
                "image_path",
                "mask_path",
                "notes",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Run classical segmentation on one frame and update metadata_labeled.csv"
    )
    parser.add_argument("--dataset_root", type=str, default="../../../../Dataset")
    parser.add_argument("--patient_id", type=str, required=True)
    parser.add_argument("--scan_id", type=str, required=True)
    parser.add_argument("--frame_id", type=str, required=True,
                        help="Frame id string, e.g. '000'")
    parser.add_argument("--notes", type=str, default="", help="Optional notes")
    parser.add_argument("--show", action="store_true",
                        help="Show intermediate figures from the segmenter")

    args = parser.parse_args()

    # Build paths relative to Dataset root
    img_rel = os.path.join(args.patient_id, args.scan_id, f"frame_{args.frame_id}.png")
    mask_rel = os.path.join(args.patient_id, args.scan_id, f"frame_{args.frame_id}_mask.png")

    img_path = os.path.join(args.dataset_root, img_rel)
    mask_path = os.path.join(args.dataset_root, mask_rel)

    # Prepare argv for the CLI segmenter
    seg_argv = [
        "ultrasound_bone_segmentation_cli.py",
        "--image_path", img_path,
        "--output_mask_path", mask_path,
    ]
    if args.show:
        seg_argv.append("--show")

    # Temporarily replace sys.argv for the imported main()
    old_argv = sys.argv
    sys.argv = seg_argv
    try:
        seg_main()
    finally:
        sys.argv = old_argv

    # Update metadata CSV
    csv_path = os.path.join(args.dataset_root, "metadata_labeled.csv")
    append_to_metadata(csv_path, {
        "patient_id": args.patient_id,
        "scan_id": args.scan_id,
        "frame_id": args.frame_id,
        "image_path": img_rel.replace("\\", "/"),
        "mask_path": mask_rel.replace("\\", "/"),
        "notes": args.notes,
    })

    print(f"Appended labeled frame to {csv_path}")


if __name__ == "__main__":
    main()
