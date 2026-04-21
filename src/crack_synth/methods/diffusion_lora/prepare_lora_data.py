from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

from PIL import Image

from crack_synth.methods.diffusion_baseline.config import DEFAULT_CONFIG_PATH, load_config
from crack_synth.methods.diffusion_baseline.guide import make_crack_guide, make_paired_residual_crack_guide
from crack_synth.methods.diffusion_baseline.io_utils import (
    ensure_dir,
    read_jsonl,
    write_csv_records,
    write_json,
    write_jsonl,
)
from crack_synth.methods.diffusion_baseline.paths import resolve_project_path
from crack_synth.methods.diffusion_baseline.progress import tqdm
from crack_synth.methods.diffusion_baseline.roi import (
    compute_union_bbox,
    dilate_binary_mask,
    fixed_square_crop_box_from_bbox,
    load_image,
    threshold_mask,
)


DEFAULT_CAPTION = (
    "ctwirecrack, close-up grayscale railway contact wire surface, one thin dark longitudinal crack"
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Stable Diffusion inpainting LoRA samples from ROI pairs.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output lora_data directory.")
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[96, 128, 160, 192, 256])
    parser.add_argument("--resolution", type=int, default=512, help="Training image resolution.")
    parser.add_argument("--mask-train-dilate-px", type=int, default=3)
    parser.add_argument("--pair-offset", type=int, default=0, help="Skip this many pairs before selecting train pairs.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Number of pairs to export for training.")
    parser.add_argument("--holdout-pairs", type=int, default=0, help="Save this many following pairs for inference.")
    parser.add_argument("--caption", type=str, default=DEFAULT_CAPTION)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pair_offset < 0:
        raise ValueError("--pair-offset cannot be negative.")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive when provided.")
    if args.holdout_pairs < 0:
        raise ValueError("--holdout-pairs cannot be negative.")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.mask_train_dilate_px < 0:
        raise ValueError("--mask-train-dilate-px cannot be negative.")
    if any(size <= 0 for size in args.crop_sizes):
        raise ValueError("--crop-sizes must be positive.")

    config = load_config(args.config)
    default_lora_dir = "diffusion_lora_v2" if config.crack_prior_mode == "paired_residual" else "diffusion_lora"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        config.output_root.parent / default_lora_dir / "lora_data"
    )
    train_dir = ensure_dir(output_dir / "train")

    pair_file = config.output_root / "roi_assets" / "roi_pairs.jsonl"
    if not pair_file.exists():
        raise FileNotFoundError(f"Missing prepared pair file: {pair_file}. Run prepare_rois first.")

    all_pairs = [_resolve_pair_record_paths(record, config) for record in read_jsonl(pair_file)]
    selected = all_pairs[args.pair_offset :]
    train_pairs = selected[: args.max_pairs] if args.max_pairs is not None else selected
    holdout_start = len(train_pairs)
    holdout_pairs = selected[holdout_start : holdout_start + args.holdout_pairs]
    if not train_pairs:
        raise ValueError("No train pairs selected.")

    sample_records: list[dict] = []
    for pair_index, pair in enumerate(tqdm(train_pairs, desc="prepare_lora_data", unit="pair")):
        sample_records.extend(
            export_pair_samples(
                pair=pair,
                pair_index=args.pair_offset + pair_index,
                train_dir=train_dir,
                config=config,
                crop_sizes=args.crop_sizes,
                resolution=args.resolution,
                mask_train_dilate_px=args.mask_train_dilate_px,
                caption=args.caption,
            )
        )

    write_jsonl(output_dir / "train_samples.jsonl", sample_records)
    write_csv_records(output_dir / "train_samples.csv", sample_records)
    write_jsonl(output_dir / "train_pairs.jsonl", train_pairs)
    write_csv_records(output_dir / "train_pairs.csv", train_pairs)
    write_jsonl(output_dir / "holdout_pairs.jsonl", holdout_pairs)
    write_csv_records(output_dir / "holdout_pairs.csv", holdout_pairs)
    write_json(
        output_dir / "summary.json",
        {
            "generator": "crack_synth.methods.diffusion_lora.prepare_lora_data",
            "source_pair_file": str(pair_file),
            "source_pair_count": len(all_pairs),
            "pair_offset": int(args.pair_offset),
            "train_pair_count": len(train_pairs),
            "holdout_pair_count": len(holdout_pairs),
            "output_sample_count": len(sample_records),
            "crop_sizes": [int(size) for size in args.crop_sizes],
            "resolution": int(args.resolution),
            "mask_train_dilate_px": int(args.mask_train_dilate_px),
            "guide_crack": bool(config.guide_crack),
            "guide_darken": float(config.guide_darken),
            "guide_blur": float(config.guide_blur),
            "guide_label_dilate_px": int(config.guide_label_dilate_px),
            "crack_prior_mode": config.crack_prior_mode,
            "crack_prior_alpha": float(config.crack_prior_alpha),
            "crack_prior_clip_percentile": float(config.crack_prior_clip_percentile),
            "crack_prior_mask_dilate_px": int(config.crack_prior_mask_dilate_px),
            "crack_prior_blur_px": float(config.crack_prior_blur_px),
            "caption": args.caption,
            "train_dir": str(train_dir),
        },
    )


def export_pair_samples(
    *,
    pair: dict,
    pair_index: int,
    train_dir: Path,
    config,
    crop_sizes: list[int],
    resolution: int,
    mask_train_dilate_px: int,
    caption: str,
) -> list[dict]:
    target_roi = load_image(pair["image_roi_path"])
    background_roi = load_image(pair["background_roi_path"])
    with Image.open(pair["mask_raw_roi_path"]) as mask_image:
        mask_raw = threshold_mask(mask_image.convert("L"))

    if target_roi.size != background_roi.size or target_roi.size != mask_raw.size:
        raise ValueError(f"ROI size mismatch for {pair['pair_id']}")

    bbox = compute_union_bbox(mask_raw)
    records: list[dict] = []
    for crop_size in crop_sizes:
        crop_box = fixed_square_crop_box_from_bbox(bbox, target_roi.size, crop_size)
        sample_id = f"pair_{pair_index:03d}_{pair['pair_id']}__crop_{crop_size}"
        sample_dir = ensure_dir(train_dir / sample_id)

        target = _crop_resize(target_roi, crop_box, resolution, Image.Resampling.BICUBIC)
        background = _crop_resize(background_roi, crop_box, resolution, Image.Resampling.BICUBIC)
        mask_label = _crop_resize(mask_raw, crop_box, resolution, Image.Resampling.NEAREST)
        mask_label = threshold_mask(mask_label)
        mask_train = dilate_binary_mask(mask_label, mask_train_dilate_px)
        if config.crack_prior_mode == "paired_residual":
            condition = make_paired_residual_crack_guide(
                background=background,
                defect=target,
                mask_label=mask_label,
                alpha=config.crack_prior_alpha,
                clip_percentile=config.crack_prior_clip_percentile,
                mask_dilate_px=config.crack_prior_mask_dilate_px,
                blur_radius=config.crack_prior_blur_px,
            )
        elif config.crack_prior_mode == "mask_dark" or config.guide_crack:
            condition = make_crack_guide(
                background=background,
                mask_label=mask_label,
                darken=config.guide_darken,
                blur_radius=config.guide_blur,
                label_dilate_px=config.guide_label_dilate_px,
            )
        else:
            condition = background

        condition_path = sample_dir / "condition.png"
        background_path = sample_dir / "background.png"
        target_path = sample_dir / "target.png"
        mask_label_path = sample_dir / "mask_label.png"
        mask_train_path = sample_dir / "mask_train.png"
        caption_path = sample_dir / "caption.txt"
        metadata_path = sample_dir / "metadata.json"

        condition.save(condition_path)
        background.save(background_path)
        target.save(target_path)
        mask_label.save(mask_label_path)
        mask_train.save(mask_train_path)
        caption_path.write_text(caption, encoding="utf-8")

        metadata = {
            "sample_id": sample_id,
            "pair_index": int(pair_index),
            "pair_id": pair["pair_id"],
            "defect_id": pair["defect_id"],
            "normal_id": pair["normal_id"],
            "crop_size": int(crop_size),
            "resolution": int(resolution),
            "crop_box_xyxy": [int(value) for value in crop_box],
            "mask_train_dilate_px": int(mask_train_dilate_px),
            "condition_path": str(condition_path),
            "background_path": str(background_path),
            "target_path": str(target_path),
            "mask_label_path": str(mask_label_path),
            "mask_train_path": str(mask_train_path),
            "caption_path": str(caption_path),
            "caption": caption,
            "guide_crack": bool(config.guide_crack),
            "crack_prior_mode": config.crack_prior_mode,
            "crack_prior_alpha": float(config.crack_prior_alpha),
        }
        write_json(metadata_path, metadata)
        records.append({**metadata, "metadata_path": str(metadata_path)})
    return records


def _crop_resize(image: Image.Image, crop_box: tuple[int, int, int, int], resolution: int, resample: int) -> Image.Image:
    return image.crop(crop_box).resize((resolution, resolution), resample=resample)


def _resolve_pair_record_paths(record: dict, config) -> dict:
    resolved = dict(record)
    for key in [
        "defect_image_path",
        "normal_image_path",
        "defect_json_path",
        "image_roi_path",
        "background_roi_path",
        "mask_raw_roi_path",
        "mask_edit_roi_path",
        "mask_paste_roi_path",
    ]:
        value = record.get(key, "")
        if value:
            resolved[key] = str(
                resolve_project_path(
                    value,
                    repo_root=config.repo_root,
                    dataset_root=config.dataset_root,
                    output_root=config.output_root,
                )
            )
    return resolved


if __name__ == "__main__":
    main()
