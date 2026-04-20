from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import random

from PIL import Image

from .config import DEFAULT_CONFIG_PATH, BaselineConfig, dump_resolved_config, load_config
from .io_utils import ensure_dir, read_jsonl, write_csv_records, write_json, write_jsonl
from .paths import resolve_project_path
from .progress import tqdm
from .roi import blur_mask, load_image


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate diffusion inpainting outputs from prepared crack/normal pairs.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--plan-only", action="store_true", help="Write planned_pairs only, skip model loading and inference.")
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
            background_roi_path = sample_dir / "background_roi.png"
            metadata_path = sample_dir / "metadata.json"

            image_syn.save(image_syn_path)
            item["mask_raw"].save(mask_raw_path)
            item["mask_edit"].save(mask_edit_path)
            item["background"].save(background_roi_path)

            metadata = dict(record)
            metadata.update(
                {
                    "image_syn_path": str(image_syn_path),
                    "mask_raw_output_path": str(mask_raw_path),
                    "mask_edit_output_path": str(mask_edit_path),
                    "background_roi_output_path": str(background_roi_path),
                    "device": device_name,
                    "torch_dtype": str(torch_dtype).replace("torch.", ""),
                    "inference_batch_size": inference_batch_size,
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
    background = load_image(record["background_roi_path"])
    if background.size != (config.roi_out_size, config.roi_out_size):
        raise ValueError(
            f"Background ROI size {background.size} does not match roi_out_size={config.roi_out_size}."
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
