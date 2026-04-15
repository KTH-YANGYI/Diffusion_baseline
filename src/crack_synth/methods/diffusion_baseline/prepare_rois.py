from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

from .config import DEFAULT_CONFIG_PATH, dump_resolved_config, load_config
from .io_utils import ensure_dir, read_csv_records, write_json, write_jsonl
from .paths import audit_and_resolve_records, to_repo_relative
from .progress import tqdm
from .roi import (
    compute_union_bbox,
    crop_pair,
    dilate_binary_mask,
    fixed_square_crop_box_from_bbox,
    load_binary_mask,
    load_image,
    resize_with_padding,
    threshold_mask,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare 512x512 donor ROI assets for one defect fold.")
    parser.add_argument("--fold", type=int, required=False, help="Fold index override.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, fold_override=args.fold)

    defect_manifest = config.manifests_root / f"defect_fold{config.fold_index}_train.csv"
    if not defect_manifest.exists():
        raise FileNotFoundError(f"Missing defect manifest: {defect_manifest}. Run build_manifests first.")

    records = read_csv_records(defect_manifest)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    resolved_records, audit = audit_and_resolve_records(
        records,
        repo_root=config.repo_root,
        dataset_root=config.dataset_root,
        manifests_root=config.manifests_root,
        output_root=config.output_root,
        require_mask=True,
    )

    fold_dir = ensure_dir(config.output_root / f"fold_{config.fold_index}")
    roi_root = ensure_dir(fold_dir / "roi_assets")
    donor_root = ensure_dir(roi_root / "donors")
    dump_resolved_config(config, roi_root / "resolved_config.json")
    write_json(roi_root / "path_audit.json", audit)

    manifest_records: list[dict] = []
    for record in tqdm(resolved_records, desc="prepare_rois", unit="sample"):
        image_path = Path(record["resolved_image_path"])
        mask_path = Path(record["resolved_mask_path"])
        sample_id = record["sample_id"]

        image = load_image(image_path)
        mask_raw = load_binary_mask(mask_path)
        bbox = compute_union_bbox(mask_raw)
        crop_box = fixed_square_crop_box_from_bbox(
            bbox,
            image.size,
            config.roi_out_size,
        )
        image_crop, mask_crop = crop_pair(image, mask_raw, crop_box)

        if image_crop.size == (config.roi_out_size, config.roi_out_size):
            image_roi = image_crop
            mask_raw_roi = threshold_mask(mask_crop)
            resize_meta = None
        else:
            image_roi, resize_meta = resize_with_padding(
                image_crop,
                out_size=config.roi_out_size,
                resample=Image.Resampling.BICUBIC,
                fill=(0, 0, 0),
            )
            mask_raw_roi, _ = resize_with_padding(
                mask_crop,
                out_size=config.roi_out_size,
                resample=Image.Resampling.NEAREST,
                fill=0,
            )
            mask_raw_roi = threshold_mask(mask_raw_roi)
        mask_edit_roi = dilate_binary_mask(mask_raw_roi, config.mask_edit_dilate_px)

        sample_dir = ensure_dir(donor_root / sample_id)
        image_roi_path = sample_dir / "image_roi.png"
        mask_raw_roi_path = sample_dir / "mask_raw_roi.png"
        mask_edit_roi_path = sample_dir / "mask_edit_roi.png"

        image_roi.save(image_roi_path)
        mask_raw_roi.save(mask_raw_roi_path)
        mask_edit_roi.save(mask_edit_roi_path)

        manifest_records.append(
            {
                "sample_id": sample_id,
                "image_name": record["image_name"],
                "video_id": record.get("video_id", ""),
                "video_name": record.get("video_name", ""),
                "frame_id": record.get("frame_id", ""),
                "resolved_image_path": to_repo_relative(image_path, config.repo_root),
                "resolved_mask_path": to_repo_relative(mask_path, config.repo_root),
                "resolved_json_path": to_repo_relative(record["resolved_json_path"], config.repo_root)
                if record.get("resolved_json_path")
                else "",
                "image_roi_path": to_repo_relative(image_roi_path, config.repo_root),
                "mask_raw_roi_path": to_repo_relative(mask_raw_roi_path, config.repo_root),
                "mask_edit_roi_path": to_repo_relative(mask_edit_roi_path, config.repo_root),
                "original_bbox_xyxy": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "crop_box_xyxy": [crop_box[0], crop_box[1], crop_box[2], crop_box[3]],
                "roi_margin_px": config.roi_margin_px,
                "roi_out_size": config.roi_out_size,
                "mask_edit_dilate_px": config.mask_edit_dilate_px,
                "crop_mode": "fixed_square_centered_on_bbox",
                "resize_scale": 1.0 if resize_meta is None else resize_meta.scale,
                "resized_width": image_roi.size[0] if resize_meta is None else resize_meta.resized_width,
                "resized_height": image_roi.size[1] if resize_meta is None else resize_meta.resized_height,
                "pad_left": 0 if resize_meta is None else resize_meta.pad_left,
                "pad_top": 0 if resize_meta is None else resize_meta.pad_top,
                "pad_right": 0 if resize_meta is None else resize_meta.pad_right,
                "pad_bottom": 0 if resize_meta is None else resize_meta.pad_bottom,
            }
        )

    write_jsonl(roi_root / "donor_manifest.jsonl", manifest_records)
    write_json(
        roi_root / "summary.json",
        {
            "fold_index": config.fold_index,
            "donor_count": len(manifest_records),
            "roi_root": str(roi_root),
        },
    )


if __name__ == "__main__":
    main()
