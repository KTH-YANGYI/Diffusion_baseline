from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import random

from PIL import Image

from .config import DEFAULT_CONFIG_PATH, BaselineConfig, dump_resolved_config, load_config
from .guide import make_crack_guide
from .io_utils import ensure_dir, read_jsonl, write_csv_records, write_json, write_jsonl
from .paths import resolve_project_path
from .postprocess import apply_hard_overlay, compute_generation_metrics, make_abs_diff_image
from .progress import tqdm
from .roi import blur_mask, compute_union_bbox, load_image


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate diffusion inpainting outputs from prepared crack/normal pairs.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--plan-only", action="store_true", help="Write planned_pairs only, skip model loading and inference.")
    parser.add_argument("--pair-offset", type=int, default=0, help="Skip this many prepared pairs before planning.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional pair limit for smoke tests.")
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional seed-per-pair override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional inference batch size override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    roi_root = ensure_dir(config.output_root / "roi_assets")
    pair_file = roi_root / "roi_pairs.jsonl"
    if not pair_file.exists():
        raise FileNotFoundError(
            f"Missing prepared pair file: {pair_file}. Run prepare_rois first."
        )

    pairs = read_jsonl(pair_file)
    if args.pair_offset < 0:
        raise ValueError("--pair-offset cannot be negative.")
    if args.pair_offset:
        pairs = pairs[args.pair_offset :]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError("No prepared ROI pair records found.")
    pairs = [_resolve_pair_record_paths(record, config) for record in pairs]

    seeds_per_pair = args.max_seeds if args.max_seeds is not None else config.seeds_per_pair
    if seeds_per_pair <= 0:
        raise ValueError("seeds_per_pair must be positive.")
    inference_batch_size = args.batch_size if args.batch_size is not None else config.inference_batch_size
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive.")

    run_dir = ensure_dir(config.output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    sample_root = ensure_dir(run_dir / "samples")
    dump_resolved_config(config, run_dir / "resolved_config.json")

    planned_pairs = plan_pairs(
        pairs=pairs,
        config=config,
        seeds_per_pair=seeds_per_pair,
        run_dir=run_dir,
    )
    write_jsonl(run_dir / "planned_pairs.jsonl", planned_pairs)
    write_csv_records(run_dir / "planned_pairs.csv", planned_pairs)

    if args.plan_only:
        write_json(
            run_dir / "summary.json",
            {
                "mode": "plan_only",
                "prepared_pair_count": len(pairs),
                "planned_output_count": len(planned_pairs),
                "pair_offset": int(args.pair_offset),
                "pair_file": str(pair_file),
                "run_dir": str(run_dir),
            },
        )
        return

    pipe, torch, device_name, torch_dtype = build_pipeline(config)
    outputs: list[dict] = []

    batch_count = (len(planned_pairs) + inference_batch_size - 1) // inference_batch_size
    for batch_records in tqdm(
        _iter_batches(planned_pairs, inference_batch_size),
        desc="generate_baseline",
        unit="batch",
        total=batch_count,
    ):
        batch_inputs = [_prepare_inference_input(record, config, torch, device_name) for record in batch_records]
        use_single_item_call = config.padding_mask_crop > 0 and len(batch_inputs) == 1
        if use_single_item_call:
            only_item = batch_inputs[0]
            call_kwargs = dict(
                prompt=only_item["prompt"],
                negative_prompt=only_item["negative_prompt"],
                image=only_item["background"],
                mask_image=only_item["mask_for_inference"],
                generator=only_item["generator"],
            )
        else:
            call_kwargs = dict(
                prompt=[item["prompt"] for item in batch_inputs],
                negative_prompt=[item["negative_prompt"] for item in batch_inputs],
                image=[item["background"] for item in batch_inputs],
                mask_image=[item["mask_for_inference"] for item in batch_inputs],
                generator=[item["generator"] for item in batch_inputs],
            )
        call_kwargs.update(
            {
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "strength": config.strength,
            }
        )
        if config.padding_mask_crop > 0:
            call_kwargs["padding_mask_crop"] = config.padding_mask_crop
        try:
            result = pipe(**call_kwargs)
        except TypeError:
            if "padding_mask_crop" not in call_kwargs:
                raise
            call_kwargs.pop("padding_mask_crop")
            result = pipe(**call_kwargs)

        for item, model_output in zip(batch_inputs, result.images, strict=True):
            record = item["record"]
            sample_dir = ensure_dir(sample_root / record["record_id"])
            image_syn_path = sample_dir / "image_syn.png"
            mask_label_path = sample_dir / "mask_label.png"
            mask_raw_path = sample_dir / "mask_raw_roi.png"
            mask_edit_path = sample_dir / "mask_edit_roi.png"
            background_roi_path = sample_dir / "background_roi.png"
            metadata_path = sample_dir / "metadata.json"

            image_syn = _postprocess_model_output(model_output, item, config)
            if config.apply_overlay:
                image_syn = apply_hard_overlay(
                    background=item["full_background"],
                    generated=image_syn,
                    mask_edit=item["mask_edit"],
                )
            image_syn.save(image_syn_path)
            item["mask_raw"].save(mask_label_path)
            item["mask_raw"].save(mask_raw_path)
            item["mask_edit"].save(mask_edit_path)
            item["full_background"].save(background_roi_path)
            if config.save_guide_image:
                item["guided_full_background"].save(sample_dir / "guide_roi.png")
            if config.save_diff_abs:
                make_abs_diff_image(item["full_background"], image_syn).save(sample_dir / "diff_abs.png")
            if item["local_inpaint"]:
                item["background"].save(sample_dir / "inpaint_background_input.png")
                item["mask_for_inference"].save(sample_dir / "inpaint_mask_input.png")
                model_output.save(sample_dir / "inpaint_model_output.png")

            metrics = {}
            if config.save_quality_metrics:
                metrics = compute_generation_metrics(
                    background=item["full_background"],
                    generated=image_syn,
                    mask_raw=item["mask_raw"],
                    mask_edit=item["mask_edit"],
                )

            metadata = dict(record)
            metadata.update(
                {
                    "image_syn_path": str(image_syn_path),
                    "mask_label_path": str(mask_label_path),
                    "mask_raw_output_path": str(mask_raw_path),
                    "mask_edit_output_path": str(mask_edit_path),
                    "background_roi_output_path": str(background_roi_path),
                    "local_inpaint": item["local_inpaint"],
                    "local_crop_box_xyxy": item["local_crop_box_xyxy"],
                    "local_crop_size": config.local_crop_size,
                    "background_overlay": config.background_overlay,
                    "overlay_blur_px": config.overlay_blur_px,
                    "padding_mask_crop": config.padding_mask_crop,
                    "apply_overlay": config.apply_overlay,
                    "guide_crack": config.guide_crack,
                    "guide_darken": config.guide_darken,
                    "guide_blur": config.guide_blur,
                    "guide_label_dilate_px": config.guide_label_dilate_px,
                    "lora_path": config.lora_path,
                    "lora_scale": config.lora_scale,
                    "lora_adapter_name": config.lora_adapter_name,
                    "device": device_name,
                    "torch_dtype": str(torch_dtype).replace("torch.", ""),
                    "inference_batch_size": inference_batch_size,
                    "disable_safety_checker": config.disable_safety_checker,
                    **metrics,
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
            "prepared_pair_count": len(pairs),
            "planned_output_count": len(planned_pairs),
            "output_count": len(outputs),
            "inference_batch_size": inference_batch_size,
            "pair_offset": int(args.pair_offset),
            "pair_file": str(pair_file),
            "run_dir": str(run_dir),
        },
    )


def plan_pairs(
    *,
    pairs: list[dict],
    config: BaselineConfig,
    seeds_per_pair: int,
    run_dir: Path,
) -> list[dict]:
    planned: list[dict] = []
    for pair in pairs:
        pair_rng = random.Random(_stable_int_seed(f"{config.planning_seed}:{pair['pair_id']}"))
        seeds = [pair_rng.randint(0, 2_147_483_647) for _ in range(seeds_per_pair)]
        for seed in seeds:
            record_id = f"{pair['pair_id']}__seed_{seed}"
            planned.append(
                {
                    "record_id": record_id,
                    "run_dir": str(run_dir),
                    "sample_dir": str(run_dir / "samples" / record_id),
                    "model_id_or_path": config.model_id_or_path,
                    "prompt": config.prompt,
                    "negative_prompt": config.negative_prompt,
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "strength": config.strength,
                    "mask_blur": config.mask_blur,
                    "local_inpaint": config.local_inpaint,
                    "local_crop_size": config.local_crop_size,
                    "background_overlay": config.background_overlay,
                    "overlay_blur_px": config.overlay_blur_px,
                    "padding_mask_crop": config.padding_mask_crop,
                    "apply_overlay": config.apply_overlay,
                    "guide_crack": config.guide_crack,
                    "guide_darken": config.guide_darken,
                    "guide_blur": config.guide_blur,
                    "guide_label_dilate_px": config.guide_label_dilate_px,
                    "lora_path": config.lora_path,
                    "lora_scale": config.lora_scale,
                    "seed": seed,
                    "pair_id": pair["pair_id"],
                    "defect_id": pair["defect_id"],
                    "normal_id": pair["normal_id"],
                    "defect_image_name": pair["defect_image_name"],
                    "normal_image_name": pair["normal_image_name"],
                    "defect_frame_id": pair.get("defect_frame_id", ""),
                    "normal_frame_id": pair.get("normal_frame_id", ""),
                    "pairing_source": pair.get("pairing_source", ""),
                    "defect_image_path": pair["defect_image_path"],
                    "normal_image_path": pair["normal_image_path"],
                    "defect_json_path": pair["defect_json_path"],
                    "image_roi_path": pair["image_roi_path"],
                    "background_roi_path": pair["background_roi_path"],
                    "mask_raw_roi_path": pair["mask_raw_roi_path"],
                    "mask_edit_roi_path": pair["mask_edit_roi_path"],
                    "crop_box_xyxy": pair["crop_box_xyxy"],
                    "pairing_mode": "dataset_one_to_one",
                }
            )
    return planned


def build_pipeline(config: BaselineConfig):
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as exc:
        raise ImportError(
            "diffusers/torch are required for inference. Install the runtime dependencies or use --plan-only."
        ) from exc

    device_name = _resolve_device(config.device, torch)
    torch_dtype = _resolve_dtype(config.dtype, device_name, torch)
    model_path = Path(config.model_id_or_path)
    pretrained_ref = str(model_path.resolve()) if model_path.exists() else config.model_id_or_path

    pipe = AutoPipelineForInpainting.from_pretrained(
        pretrained_ref,
        torch_dtype=torch_dtype,
        local_files_only=bool(config.local_files_only),
    )
    pipe.set_progress_bar_config(disable=True)
    if config.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if config.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    if config.disable_safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
    if config.lora_path:
        lora_path = Path(config.lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"Configured LoRA path does not exist: {lora_path}")
        pipe.load_lora_weights(str(lora_path), adapter_name=config.lora_adapter_name)
        if hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters([config.lora_adapter_name], adapter_weights=[config.lora_scale])
            except TypeError:
                pipe.set_adapters([config.lora_adapter_name], [config.lora_scale])
        elif hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=config.lora_scale)
    pipe = pipe.to(device_name)
    return pipe, torch, device_name, torch_dtype


def _resolve_device(raw_device: str, torch) -> str:
    if raw_device != "auto":
        return raw_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(raw_dtype: str, device_name: str, torch):
    normalized = raw_dtype.lower()
    if device_name == "cpu" and normalized in {"float16", "fp16", "half"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _iter_batches(records: list[dict], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _prepare_inference_input(record: dict, config: BaselineConfig, torch, device_name: str) -> dict:
    full_background = load_image(record["background_roi_path"])
    if full_background.size != (config.roi_out_size, config.roi_out_size):
        raise ValueError(
            f"Background ROI size {full_background.size} does not match roi_out_size={config.roi_out_size}."
        )
    with Image.open(record["mask_edit_roi_path"]) as mask_image:
        mask_edit = mask_image.convert("L")
    with Image.open(record["mask_raw_roi_path"]) as raw_mask_image:
        mask_raw = raw_mask_image.convert("L")

    if config.guide_crack:
        guided_full_background = make_crack_guide(
            background=full_background,
            mask_label=mask_raw,
            darken=config.guide_darken,
            blur_radius=config.guide_blur,
            label_dilate_px=config.guide_label_dilate_px,
        )
    else:
        guided_full_background = full_background

    local_crop_box_xyxy: list[int] | None = None
    background_for_model = guided_full_background
    mask_for_model = mask_edit
    if config.local_inpaint:
        crop_box = _fixed_square_crop_box_from_mask(
            mask_raw,
            full_background.size,
            config.local_crop_size,
        )
        local_crop_box_xyxy = [crop_box[0], crop_box[1], crop_box[2], crop_box[3]]
        background_for_model = guided_full_background.crop(crop_box).resize(
            (config.roi_out_size, config.roi_out_size),
            resample=Image.Resampling.BICUBIC,
        )
        mask_for_model = mask_edit.crop(crop_box).resize(
            (config.roi_out_size, config.roi_out_size),
            resample=Image.Resampling.NEAREST,
        )

    return {
        "record": record,
        "prompt": record["prompt"],
        "negative_prompt": record["negative_prompt"],
        "background": background_for_model,
        "full_background": full_background,
        "guided_full_background": guided_full_background,
        "mask_edit": mask_edit,
        "mask_raw": mask_raw,
        "mask_for_inference": blur_mask(mask_for_model, config.mask_blur),
        "local_inpaint": config.local_inpaint,
        "local_crop_box_xyxy": local_crop_box_xyxy,
        "generator": _make_generator(torch, device_name, int(record["seed"])),
    }


def _postprocess_model_output(
    model_output: Image.Image,
    item: dict,
    config: BaselineConfig,
) -> Image.Image:
    if item["local_inpaint"]:
        crop_box = tuple(int(value) for value in item["local_crop_box_xyxy"])
        x0, y0, x1, y1 = crop_box
        crop_size = (x1 - x0, y1 - y0)
        generated_crop = model_output.resize(crop_size, resample=Image.Resampling.BICUBIC)
        local_mask = item["mask_edit"].crop(crop_box)
        return _paste_with_soft_mask(
            background=item["full_background"],
            generated_crop=generated_crop,
            mask_crop=local_mask,
            crop_box=crop_box,
            blur_px=config.overlay_blur_px if config.background_overlay else 0.0,
            use_mask=config.background_overlay,
        )
    if config.background_overlay:
        return Image.composite(
            model_output,
            item["full_background"],
            blur_mask(item["mask_edit"], config.overlay_blur_px),
        )
    return model_output


def _paste_with_soft_mask(
    *,
    background: Image.Image,
    generated_crop: Image.Image,
    mask_crop: Image.Image,
    crop_box: tuple[int, int, int, int],
    blur_px: float,
    use_mask: bool,
) -> Image.Image:
    output = background.copy()
    x0, y0, x1, y1 = crop_box
    background_crop = background.crop(crop_box)
    if use_mask:
        soft_mask = blur_mask(mask_crop.convert("L"), blur_px)
        pasted_crop = Image.composite(generated_crop, background_crop, soft_mask)
    else:
        pasted_crop = generated_crop
    output.paste(pasted_crop, (x0, y0, x1, y1))
    return output


def _fixed_square_crop_box_from_mask(
    mask: Image.Image,
    image_size: tuple[int, int],
    crop_size: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = compute_union_bbox(mask)
    width, height = image_size
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half = crop_size / 2.0
    crop_x0 = int(round(cx - half))
    crop_y0 = int(round(cy - half))
    crop_x0 = max(0, min(crop_x0, width - crop_size))
    crop_y0 = max(0, min(crop_y0, height - crop_size))
    return crop_x0, crop_y0, crop_x0 + crop_size, crop_y0 + crop_size


def _make_generator(torch, device_name: str, seed: int):
    if device_name.startswith("cuda"):
        return torch.Generator(device=device_name).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _stable_int_seed(text: str) -> int:
    digest = sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _parse_optional_int(value: object) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _resolve_pair_record_paths(record: dict, config: BaselineConfig) -> dict:
    resolved = dict(record)
    for key in [
        "defect_image_path",
        "normal_image_path",
        "defect_json_path",
        "image_roi_path",
        "background_roi_path",
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
                    output_root=config.output_root,
                )
            )
    return resolved


if __name__ == "__main__":
    main()
