from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import random

from PIL import Image

from .config import DEFAULT_CONFIG_PATH, BaselineConfig, dump_resolved_config, load_config
from .io_utils import ensure_dir, read_csv_records, read_jsonl, write_csv_records, write_json, write_jsonl
from .paths import audit_and_resolve_records, resolve_project_path
from .progress import tqdm
from .roi import blur_mask, load_image


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate patch-level diffusion baseline outputs for one fold.")
    parser.add_argument("--fold", type=int, required=False, help="Fold index override.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--plan-only", action="store_true", help="Write planned_pairs only, skip model loading and inference.")
    parser.add_argument("--max-donors", type=int, default=None, help="Optional donor limit for smoke tests.")
    parser.add_argument("--max-backgrounds", type=int, default=None, help="Optional background-per-donor override.")
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional seed-per-pair override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, fold_override=args.fold)

    fold_dir = ensure_dir(config.output_root / f"fold_{config.fold_index}")
    roi_root = ensure_dir(fold_dir / "roi_assets")
    donor_manifest_path = roi_root / "donor_manifest.jsonl"
    if not donor_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing donor manifest: {donor_manifest_path}. Run prepare_rois first."
        )

    donors = read_jsonl(donor_manifest_path)
    if args.max_donors is not None:
        donors = donors[: args.max_donors]
    if not donors:
        raise ValueError("No donor ROI records found for the requested fold.")
    donors = [_resolve_donor_record_paths(record, config) for record in donors]

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

    normals_by_video = group_normals_by_video(normal_records)
    backgrounds_per_donor = args.max_backgrounds if args.max_backgrounds is not None else config.backgrounds_per_donor
    if backgrounds_per_donor <= 0:
        raise ValueError("backgrounds_per_donor must be positive.")
    seeds_per_pair = args.max_seeds if args.max_seeds is not None else config.seeds_per_pair
    if seeds_per_pair <= 0:
        raise ValueError("seeds_per_pair must be positive.")

    run_dir = ensure_dir(fold_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    sample_root = ensure_dir(run_dir / "samples")
    dump_resolved_config(config, run_dir / "resolved_config.json")
    write_json(run_dir / "normal_path_audit.json", normal_audit)

    planned_pairs, skipped_donors = plan_pairs(
        donors=donors,
        normals_by_video=normals_by_video,
        config=config,
        backgrounds_per_donor=backgrounds_per_donor,
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
                "planned_pair_count": len(planned_pairs),
                "skipped_donor_count": len(skipped_donors),
                "skipped_donors": skipped_donors[:20],
                "background_manifest": str(normal_manifest),
                "run_dir": str(run_dir),
            },
        )
        return

    pipe, torch, device_name, torch_dtype = build_pipeline(config)
    outputs: list[dict] = []

    for record in tqdm(planned_pairs, desc="generate_baseline", unit="sample"):
        sample_dir = ensure_dir(sample_root / record["record_id"])
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

        mask_for_inference = blur_mask(mask_edit, config.mask_blur)
        generator = _make_generator(torch, device_name, int(record["seed"]))
        result = pipe(
            prompt=record["prompt"],
            negative_prompt=record["negative_prompt"],
            image=background,
            mask_image=mask_for_inference,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            strength=config.strength,
            generator=generator,
        )
        image_syn = result.images[0]

        image_syn_path = sample_dir / "image_syn.png"
        mask_raw_path = sample_dir / "mask_raw_roi.png"
        mask_edit_path = sample_dir / "mask_edit_roi.png"
        metadata_path = sample_dir / "metadata.json"

        image_syn.save(image_syn_path)
        mask_raw.save(mask_raw_path)
        mask_edit.save(mask_edit_path)

        metadata = dict(record)
        metadata.update(
            {
                "image_syn_path": str(image_syn_path),
                "mask_raw_output_path": str(mask_raw_path),
                "mask_edit_output_path": str(mask_edit_path),
                "device": device_name,
                "torch_dtype": str(torch_dtype).replace("torch.", ""),
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
            "planned_pair_count": len(planned_pairs),
            "output_count": len(outputs),
            "skipped_donor_count": len(skipped_donors),
            "skipped_donors": skipped_donors[:20],
            "background_manifest": str(normal_manifest),
            "run_dir": str(run_dir),
        },
    )


def plan_pairs(
    *,
    donors: list[dict],
    normals_by_video: dict[str, list[dict]],
    config: BaselineConfig,
    backgrounds_per_donor: int,
    seeds_per_pair: int,
    run_dir: Path,
) -> tuple[list[dict], list[dict]]:
    planned: list[dict] = []
    skipped_donors: list[dict] = []
    for donor in donors:
        donor_video_id = str(donor.get("video_id", "")).strip()
        candidate_normals = normals_by_video.get(donor_video_id, [])
        if not candidate_normals:
            skipped_donors.append(
                {
                    "donor_id": donor["sample_id"],
                    "donor_video_id": donor_video_id,
                    "reason": "no_normal_background_for_same_video_id",
                }
            )
            continue

        ranked_normals = rank_backgrounds_for_donor(donor, candidate_normals)
        background_count = min(backgrounds_per_donor, len(ranked_normals))
        for bg_order, background in enumerate(ranked_normals[:background_count]):
            pair_rng = random.Random(
                _stable_int_seed(
                    f"{config.planning_seed}:{donor['sample_id']}:{background['sample_id']}:{bg_order}"
                )
            )
            seeds = [pair_rng.randint(0, 2_147_483_647) for _ in range(seeds_per_pair)]
            for seed in seeds:
                record_id = f"{donor['sample_id']}__{background['sample_id']}__seed_{seed}"
                planned.append(
                    {
                        "record_id": record_id,
                        "fold_index": config.fold_index,
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
                        "donor_id": donor["sample_id"],
                        "donor_video_id": donor.get("video_id", ""),
                        "donor_frame_id": donor.get("frame_id", ""),
                        "background_selection_mode": config.background_selection_mode,
                        "image_roi_path": donor["image_roi_path"],
                        "mask_raw_roi_path": donor["mask_raw_roi_path"],
                        "mask_edit_roi_path": donor["mask_edit_roi_path"],
                        "crop_box_xyxy": donor["crop_box_xyxy"],
                        "background_id": background["sample_id"],
                        "background_video_id": background.get("video_id", ""),
                        "background_frame_id": background.get("frame_id", ""),
                        "background_image_path": background["resolved_image_path"],
                    }
                )
    return planned, skipped_donors


def group_normals_by_video(normal_records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in normal_records:
        grouped[str(row.get("video_id", "")).strip()].append(row)
    return dict(grouped)


def rank_backgrounds_for_donor(donor: dict, candidate_normals: list[dict]) -> list[dict]:
    donor_frame_id = _parse_optional_int(donor.get("frame_id", ""))
    if donor_frame_id is None:
        return sorted(candidate_normals, key=lambda row: row.get("sample_id", ""))

    def sort_key(row: dict) -> tuple[int, int, str]:
        normal_frame_id = _parse_optional_int(row.get("frame_id", ""))
        if normal_frame_id is None:
            return (10**9, 10**9, row.get("sample_id", ""))
        return (
            abs(normal_frame_id - donor_frame_id),
            normal_frame_id,
            row.get("sample_id", ""),
        )

    return sorted(candidate_normals, key=sort_key)


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


def _resolve_donor_record_paths(record: dict, config: BaselineConfig) -> dict:
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


if __name__ == "__main__":
    main()
