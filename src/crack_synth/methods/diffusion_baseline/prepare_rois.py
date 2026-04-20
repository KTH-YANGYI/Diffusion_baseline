from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

from .config import DEFAULT_CONFIG_PATH, dump_resolved_config, load_config
from .io_utils import ensure_dir, write_json, write_jsonl
from .paired_dataset import load_paired_samples
from .paths import to_repo_relative
from .progress import tqdm
from .roi import (
    compute_union_bbox,
    crop_pair,
    dilate_binary_mask,
    fixed_square_crop_box_from_bbox,
    load_image,
    load_labelme_mask,
    resize_with_padding,
    threshold_mask,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare paired ROI assets directly from crack/normal dataset pairs.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    records = load_paired_samples(config.dataset_root, config.repo_root)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    roi_root = ensure_dir(config.output_root / "roi_assets")
    pair_root = ensure_dir(roi_root / "pairs")
    dump_resolved_config(config, roi_root / "resolved_config.json")

    pair_records: list[dict] = []
    for record in tqdm(records, desc="prepare_rois", unit="pair"):
        defect_image_path = Path(record["defect_image_path"])
        normal_image_path = Path(record["normal_image_path"])
        defect_json_path = Path(record["defect_json_path"])
        pair_id = record["pair_id"]

        image = load_image(defect_image_path)
        normal = load_image(normal_image_path)
        if normal.size != image.size:
            raise ValueError(
                f"Paired images must have the same size for {pair_id}: crack={image.size}, normal={normal.size}"
            )

        mask_raw = load_labelme_mask(defect_json_path, image.size)
        bbox = compute_union_bbox(mask_raw)
        crop_box = fixed_square_crop_box_from_bbox(
            bbox,
            image.size,
            config.roi_out_size,
        )
        image_crop, mask_crop = crop_pair(image, mask_raw, crop_box)
        normal_crop = normal.crop(crop_box)

        if image_crop.size == (config.roi_out_size, config.roi_out_size):
            image_roi = image_crop
            background_roi = normal_crop
            mask_raw_roi = threshold_mask(mask_crop)
            resize_meta = None
        else:
            image_roi, resize_meta = resize_with_padding(
                image_crop,
                out_size=config.roi_out_size,
                resample=Image.Resampling.BICUBIC,
                fill=(0, 0, 0),
            )
            background_roi, _ = resize_with_padding(
                normal_crop,
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

        sample_dir = ensure_dir(pair_root / pair_id)
        image_roi_path = sample_dir / "defect_roi.png"
        background_roi_path = sample_dir / "background_roi.png"
        mask_raw_roi_path = sample_dir / "mask_raw_roi.png"
        mask_edit_roi_path = sample_dir / "mask_edit_roi.png"

        image_roi.save(image_roi_path)
        background_roi.save(background_roi_path)
        mask_raw_roi.save(mask_raw_roi_path)
        mask_edit_roi.save(mask_edit_roi_path)

        pair_records.append(
            {
                **record,
                "defect_image_path": to_repo_relative(defect_image_path, config.repo_root),
                "normal_image_path": to_repo_relative(normal_image_path, config.repo_root),
                "defect_json_path": to_repo_relative(defect_json_path, config.repo_root),
                "image_roi_path": to_repo_relative(image_roi_path, config.repo_root),
                "background_roi_path": to_repo_relative(background_roi_path, config.repo_root),
                "mask_raw_roi_path": to_repo_relative(mask_raw_roi_path, config.repo_root),
                "mask_edit_roi_path": to_repo_relative(mask_edit_roi_path, config.repo_root),
                "original_bbox_xyxy": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "crop_box_xyxy": [crop_box[0], crop_box[1], crop_box[2], crop_box[3]],
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

    write_jsonl(roi_root / "roi_pairs.jsonl", pair_records)
    write_json(
        roi_root / "summary.json",
        {
            "pair_count": len(pair_records),
            "dataset_root": str(config.dataset_root),
            "roi_root": str(roi_root),
            "pair_file": str(roi_root / "roi_pairs.jsonl"),
            "pairing_sources": sorted({str(record.get("pairing_source", "")) for record in pair_records}),
        },
    )


if __name__ == "__main__":
    main()
