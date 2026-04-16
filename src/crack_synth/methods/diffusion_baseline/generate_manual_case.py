from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import DEFAULT_CONFIG_PATH, load_config, dump_resolved_config
from .generate_baseline import build_pipeline
from .io_utils import ensure_dir, read_csv_records, read_jsonl, write_csv_records, write_json, write_jsonl
from .paths import audit_and_resolve_records, resolve_project_path
from .progress import tqdm
from .roi import blur_mask, load_image


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run manual diffusion inference for one donor with explicitly selected backgrounds."
    )
    parser.add_argument("--fold", type=int, required=False, help="Fold index override.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--donor",
        type=str,
        required=True,
        help="Donor selector. Accepts sample_id like defect_000826, image name like 000826.jpg, or numeric stem 826.",
    )
    parser.add_argument(
        "--backgrounds",
        type=str,
        nargs="+",
        required=True,
        help="Background selectors. Accept sample_id, image name, or numeric stems such as 693 694 695 696.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[20260417],
        help="One or more random seeds. Defaults to 20260417.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional inference batch size override.")
    parser.add_argument("--plan-only", action="store_true", help="Write plan only, skip model loading and inference.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, fold_override=args.fold)

    fold_dir = ensure_dir(config.output_root / f"fold_{config.fold_index}")
    roi_root = ensure_dir(fold_dir / "roi_assets")
    donor_manifest_path = roi_root / "donor_manifest.jsonl"
    if not donor_manifest_path.exists():
        raise FileNotFoundError(f"Missing donor manifest: {donor_manifest_path}. Run prepare_rois first.")

    donor_records = read_jsonl(donor_manifest_path)
    donor = _select_donor_record(donor_records, args.donor, config)

    normal_manifest = config.manifests_root / "normal_pool.csv"
    if not normal_manifest.exists():
        raise FileNotFoundError(f"Missing normal manifest: {normal_manifest}")
    normal_records = read_csv_records(normal_manifest)
    normal_records, normal_audit = audit_and_resolve_records(
        normal_records,
        repo_root=config.repo_root,
        dataset_root=config.dataset_root,
        manifests_root=config.manifests_root,
        output_root=config.output_root,
        require_mask=False,
    )
    selected_backgrounds = [_select_background_record(normal_records, selector) for selector in args.backgrounds]

    batch_size = args.batch_size if args.batch_size is not None else config.inference_batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not args.seed:
        raise ValueError("At least one seed is required.")

    run_dir = ensure_dir(
        fold_dir / f"manual_{donor['sample_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    sample_root = ensure_dir(run_dir / "samples")
    dump_resolved_config(config, run_dir / "resolved_config.json")
    write_json(run_dir / "normal_path_audit.json", normal_audit)

    planned_records = _build_manual_records(
        donor=donor,
        backgrounds=selected_backgrounds,
        seeds=args.seed,
        config=config,
        run_dir=run_dir,
    )
    write_jsonl(run_dir / "planned_pairs.jsonl", planned_records)
    write_csv_records(run_dir / "planned_pairs.csv", planned_records)
    write_json(
        run_dir / "selection_summary.json",
        {
            "donor_id": donor["sample_id"],
            "donor_image_name": donor["image_name"],
            "background_ids": [row["sample_id"] for row in selected_backgrounds],
            "background_image_names": [row["image_name"] for row in selected_backgrounds],
            "seeds": args.seed,
            "run_dir": str(run_dir),
            "mode": "plan_only" if args.plan_only else "inference",
        },
    )

    if args.plan_only:
        write_json(
            run_dir / "summary.json",
            {
                "mode": "plan_only",
                "planned_pair_count": len(planned_records),
                "run_dir": str(run_dir),
            },
        )
        return

    pipe, torch, device_name, torch_dtype = build_pipeline(config)
    outputs: list[dict] = []
    batch_count = (len(planned_records) + batch_size - 1) // batch_size

    for batch_records in tqdm(
        _iter_batches(planned_records, batch_size),
        desc="generate_manual_case",
        unit="batch",
        total=batch_count,
    ):
        batch_inputs = [_prepare_manual_inference_input(record, config, torch, device_name) for record in batch_records]
        result = pipe(
            prompt=[item["prompt"] for item in batch_inputs],
            negative_prompt=[item["negative_prompt"] for item in batch_inputs],
            image=[item["background"] for item in batch_inputs],
            mask_image=[item["mask_for_inference"] for item in batch_inputs],
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            strength=config.strength,
            generator=[item["generator"] for item in batch_inputs],
        )

        for item, image_syn in zip(batch_inputs, result.images, strict=True):
            record = item["record"]
            sample_dir = ensure_dir(sample_root / record["record_id"])
            image_syn_path = sample_dir / "image_syn.png"
            mask_raw_path = sample_dir / "mask_raw_roi.png"
            mask_edit_path = sample_dir / "mask_edit_roi.png"
            background_crop_path = sample_dir / "background_crop.png"
            metadata_path = sample_dir / "metadata.json"

            image_syn.save(image_syn_path)
            item["mask_raw"].save(mask_raw_path)
            item["mask_edit"].save(mask_edit_path)
            item["background"].save(background_crop_path)

            metadata = dict(record)
            metadata.update(
                {
                    "image_syn_path": str(image_syn_path),
                    "mask_raw_output_path": str(mask_raw_path),
                    "mask_edit_output_path": str(mask_edit_path),
                    "background_crop_path": str(background_crop_path),
                    "device": device_name,
                    "torch_dtype": str(torch_dtype).replace("torch.", ""),
                    "inference_batch_size": batch_size,
                }
            )
            write_json(metadata_path, metadata)
            outputs.append(metadata)

    write_jsonl(run_dir / "outputs.jsonl", outputs)
    write_csv_records(run_dir / "outputs.csv", outputs)
    write_json(
        run_dir / "summary.json",
        {
            "mode": "inference",
            "output_count": len(outputs),
            "run_dir": str(run_dir),
            "donor_id": donor["sample_id"],
            "background_ids": [row["sample_id"] for row in selected_backgrounds],
            "seeds": args.seed,
            "inference_batch_size": batch_size,
        },
    )


def _select_donor_record(records: list[dict], selector: str, config) -> dict:
    normalized = _normalize_donor_selector(selector)
    for record in records:
        if record.get("sample_id", "") == normalized:
            return _resolve_donor_record_paths(record, config)
    raise KeyError(f"Donor not found in donor manifest: {selector}")


def _select_background_record(records: list[dict], selector: str) -> dict:
    image_name, sample_id = _normalize_background_selector(selector)
    for record in records:
        if record.get("sample_id", "") == sample_id or record.get("image_name", "") == image_name:
            return record
    raise KeyError(f"Background not found in normal manifest: {selector}")


def _normalize_donor_selector(selector: str) -> str:
    text = selector.strip()
    if text.startswith("defect_"):
        return text
    stem = Path(text).stem if "." in text else text
    return f"defect_{int(stem):06d}"


def _normalize_background_selector(selector: str) -> tuple[str, str]:
    text = selector.strip()
    if text.startswith("normal_"):
        stem = text.removeprefix("normal_")
        return f"{int(stem):06d}.jpg", f"normal_{int(stem):06d}"
    stem = Path(text).stem if "." in text else text
    number = int(stem)
    return f"{number:06d}.jpg", f"normal_{number:06d}"


def _build_manual_records(
    *,
    donor: dict,
    backgrounds: list[dict],
    seeds: list[int],
    config,
    run_dir: Path,
) -> list[dict]:
    records: list[dict] = []
    for background in backgrounds:
        for seed in seeds:
            record_id = f"{donor['sample_id']}__{background['sample_id']}__seed_{seed}"
            records.append(
                {
                    "record_id": record_id,
                    "fold_index": config.fold_index,
                    "run_dir": str(run_dir),
                    "sample_dir": str(run_dir / "samples" / record_id),
                    "selection_mode": "manual_background_list",
                    "model_id_or_path": config.model_id_or_path,
                    "prompt": config.prompt,
                    "negative_prompt": config.negative_prompt,
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "strength": config.strength,
                    "mask_blur": config.mask_blur,
                    "seed": int(seed),
                    "donor_id": donor["sample_id"],
                    "donor_image_name": donor["image_name"],
                    "donor_video_id": donor.get("video_id", ""),
                    "donor_frame_id": donor.get("frame_id", ""),
                    "image_roi_path": donor["image_roi_path"],
                    "mask_raw_roi_path": donor["mask_raw_roi_path"],
                    "mask_edit_roi_path": donor["mask_edit_roi_path"],
                    "crop_box_xyxy": donor["crop_box_xyxy"],
                    "background_id": background["sample_id"],
                    "background_image_name": background["image_name"],
                    "background_video_id": background.get("video_id", ""),
                    "background_frame_id": background.get("frame_id", ""),
                    "background_image_path": background["resolved_image_path"],
                }
            )
    return records


def _prepare_manual_inference_input(record: dict, config, torch, device_name: str) -> dict:
    background_full = load_image(record["background_image_path"])
    crop_box = tuple(int(v) for v in record["crop_box_xyxy"])
    background = background_full.crop(crop_box)
    if background.size != (config.roi_out_size, config.roi_out_size):
        raise ValueError(
            f"Background crop size {background.size} does not match roi_out_size={config.roi_out_size}."
        )
    with Image.open(record["mask_edit_roi_path"]) as mask_image:
        mask_edit = mask_image.convert("L")
    with Image.open(record["mask_raw_roi_path"]) as raw_mask_image:
        mask_raw = raw_mask_image.convert("L")
    return {
        "record": record,
        "prompt": record["prompt"],
        "negative_prompt": record["negative_prompt"],
        "background": background,
        "mask_edit": mask_edit,
        "mask_raw": mask_raw,
        "mask_for_inference": blur_mask(mask_edit, config.mask_blur),
        "generator": _make_generator(torch, device_name, int(record["seed"])),
    }


def _make_generator(torch, device_name: str, seed: int):
    if device_name.startswith("cuda"):
        return torch.Generator(device=device_name).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _resolve_donor_record_paths(record: dict, config) -> dict:
    resolved = dict(record)
    for key in [
        "resolved_image_path",
        "resolved_mask_path",
        "resolved_json_path",
        "image_roi_path",
        "mask_raw_roi_path",
        "mask_edit_roi_path",
    ]:
        value = record.get(key, "")
        if value:
            resolved[key] = str(
                resolve_project_path(
                    value,
                    repo_root=config.repo_root,
                    dataset_root=config.dataset_root,
                    manifests_root=config.manifests_root,
                    output_root=config.output_root,
                )
            )
    return resolved


def _iter_batches(records: list[dict], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


if __name__ == "__main__":
    main()
