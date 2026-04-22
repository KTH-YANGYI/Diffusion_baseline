from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIG_PATH, load_config
from .io_utils import ensure_dir, read_jsonl, write_csv_records, write_json, write_jsonl
from .paths import resolve_project_path
from .roi import dilate_binary_mask, load_binary_mask, load_image


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate generation quality for one diffusion run.")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory to evaluate.")
    parser.add_argument("--latest", action="store_true", help="Evaluate the latest run under the configured output root.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--ring-width-px", type=int, default=5, help="Width of the local ring used for contrast comparison.")
    parser.add_argument("--min-edit-energy-ratio", type=float, default=0.60, help="Minimum share of total change energy expected inside the edit mask.")
    parser.add_argument("--max-outside-energy-ratio", type=float, default=0.40, help="Maximum allowed share of total change energy outside the edit mask.")
    parser.add_argument("--min-raw-diff-mean", type=float, default=3.0, help="Minimum mean absolute change expected inside the raw crack mask.")
    parser.add_argument("--min-contrast-gain", type=float, default=1.0, help="Minimum expected crack-vs-ring contrast gain over the background.")
    parser.add_argument("--preview-count", type=int, default=20, help="Number of low-score samples to keep in the summary preview.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_dir = _resolve_run_dir(config, args.run_dir, args.latest)
    records = _load_run_records(run_dir)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError(f"No generated records found in {run_dir}")

    evaluation_dir = ensure_dir(run_dir / "evaluation")
    evaluated_records: list[dict] = []
    for record in records:
        evaluated_records.append(
            evaluate_record(
                record=record,
                run_dir=run_dir,
                config=config,
                ring_width_px=args.ring_width_px,
                min_edit_energy_ratio=args.min_edit_energy_ratio,
                max_outside_energy_ratio=args.max_outside_energy_ratio,
                min_raw_diff_mean=args.min_raw_diff_mean,
                min_contrast_gain=args.min_contrast_gain,
            )
        )

    summary = build_summary(
        run_dir=run_dir,
        records=evaluated_records,
        preview_count=args.preview_count,
        thresholds={
            "min_edit_energy_ratio": float(args.min_edit_energy_ratio),
            "max_outside_energy_ratio": float(args.max_outside_energy_ratio),
            "min_raw_diff_mean": float(args.min_raw_diff_mean),
            "min_contrast_gain": float(args.min_contrast_gain),
            "ring_width_px": int(args.ring_width_px),
        },
    )

    write_jsonl(evaluation_dir / "generation_eval.jsonl", evaluated_records)
    write_csv_records(evaluation_dir / "generation_eval.csv", evaluated_records)
    write_json(evaluation_dir / "generation_eval_summary.json", summary)


def _resolve_run_dir(config, raw_run_dir: str | None, use_latest: bool) -> Path:
    if raw_run_dir:
        run_dir = Path(raw_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir

    run_dirs = sorted(path for path in config.output_root.glob("run_*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {config.output_root}")
    if use_latest or len(run_dirs) == 1:
        return run_dirs[-1]
    raise ValueError("Multiple run directories found. Pass --latest or provide --run-dir explicitly.")


def _load_run_records(run_dir: Path) -> list[dict]:
    outputs_jsonl = run_dir / "outputs.jsonl"
    if outputs_jsonl.exists():
        return read_jsonl(outputs_jsonl)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("mode") == "plan_only":
            raise ValueError(f"Run is plan_only and has no generated outputs: {run_dir}")

    records: list[dict] = []
    for metadata_path in sorted((run_dir / "samples").glob("*/metadata.json")):
        records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    return records


def evaluate_record(
    *,
    record: dict,
    run_dir: Path,
    config,
    ring_width_px: int,
    min_edit_energy_ratio: float,
    max_outside_energy_ratio: float,
    min_raw_diff_mean: float,
    min_contrast_gain: float,
) -> dict:
    sample_dir = run_dir / "samples" / record["record_id"]
    image_syn_path = _pick_existing_path(
        [
            sample_dir / "image_syn.png",
            _resolve_record_path(record.get("image_syn_path", ""), config),
        ]
    )
    mask_raw_path = _pick_existing_path(
        [
            sample_dir / "mask_raw_roi.png",
            _resolve_record_path(record.get("mask_raw_output_path", ""), config),
            _resolve_record_path(record.get("mask_raw_roi_path", ""), config),
        ]
    )
    mask_edit_path = _pick_existing_path(
        [
            sample_dir / "mask_edit_roi.png",
            _resolve_record_path(record.get("mask_edit_output_path", ""), config),
            _resolve_record_path(record.get("mask_edit_roi_path", ""), config),
        ]
    )
    synth = _load_gray(image_syn_path)
    background_roi_path = _first_existing_path(
        [
            sample_dir / "background_roi.png",
            _resolve_record_path(record.get("background_roi_output_path", ""), config),
            _resolve_record_path(record.get("background_roi_path", ""), config),
        ]
    )
    if background_roi_path is not None:
        background = _load_gray(background_roi_path)
    else:
        background_image_path = _resolve_record_path(record.get("background_image_path", ""), config)
        background_full = _load_gray(background_image_path)
        crop_box = tuple(int(v) for v in record["crop_box_xyxy"])
        x0, y0, x1, y1 = crop_box
        background = background_full[y0:y1, x0:x1]
    if background.shape != synth.shape:
        raise ValueError(
            f"Background ROI shape {background.shape} does not match generated image shape {synth.shape} for {record['record_id']}"
        )

    raw_mask = _mask_to_bool(mask_raw_path)
    edit_mask = _mask_to_bool(mask_edit_path)
    ring_mask = _build_ring_mask(mask_raw_path, raw_mask, edit_mask, ring_width_px)
    outside_edit_mask = ~edit_mask

    diff_abs = np.abs(synth - background)
    grad_synth = _gradient_magnitude(synth)
    grad_background = _gradient_magnitude(background)

    total_diff_sum = float(diff_abs.sum())
    edit_energy_ratio = _safe_sum(diff_abs, edit_mask) / total_diff_sum if total_diff_sum > 0 else 0.0
    outside_energy_ratio = _safe_sum(diff_abs, outside_edit_mask) / total_diff_sum if total_diff_sum > 0 else 0.0

    synth_raw_contrast = abs(_safe_mean(synth, raw_mask) - _safe_mean(synth, ring_mask))
    background_raw_contrast = abs(_safe_mean(background, raw_mask) - _safe_mean(background, ring_mask))
    raw_ring_contrast_gain = synth_raw_contrast - background_raw_contrast
    raw_edge_gain = _safe_mean(grad_synth, raw_mask) - _safe_mean(grad_background, raw_mask)

    raw_diff_mean = _safe_mean(diff_abs, raw_mask)
    outside_edit_diff_mean = _safe_mean(diff_abs, outside_edit_mask)

    heuristic_quality_score = 100.0 * np.mean(
        [
            _clip01(edit_energy_ratio),
            _clip01(1.0 - outside_energy_ratio),
            _clip01(raw_diff_mean / 12.0),
            _clip01(max(raw_ring_contrast_gain, 0.0) / 8.0),
            _clip01(max(raw_edge_gain, 0.0) / 10.0),
        ]
    )

    pass_edit_focus = edit_energy_ratio >= min_edit_energy_ratio
    pass_low_leakage = outside_energy_ratio <= max_outside_energy_ratio
    pass_visible_change = raw_diff_mean >= min_raw_diff_mean
    pass_contrast_gain = raw_ring_contrast_gain >= min_contrast_gain
    pass_overall = bool(pass_edit_focus and pass_low_leakage and pass_visible_change and pass_contrast_gain)

    evaluated = dict(record)
    evaluated.update(
        {
            "eval_run_dir": str(run_dir),
            "eval_image_syn_path": str(image_syn_path),
            "eval_background_patch_shape": [int(synth.shape[1]), int(synth.shape[0])],
            "eval_raw_pixel_count": int(raw_mask.sum()),
            "eval_edit_pixel_count": int(edit_mask.sum()),
            "eval_ring_pixel_count": int(ring_mask.sum()),
            "eval_total_diff_mean": round(float(diff_abs.mean()), 6),
            "eval_raw_diff_mean": round(float(raw_diff_mean), 6),
            "eval_edit_diff_mean": round(float(_safe_mean(diff_abs, edit_mask)), 6),
            "eval_outside_edit_diff_mean": round(float(outside_edit_diff_mean), 6),
            "eval_edit_energy_ratio": round(float(edit_energy_ratio), 6),
            "eval_outside_energy_ratio": round(float(outside_energy_ratio), 6),
            "eval_synth_raw_ring_contrast": round(float(synth_raw_contrast), 6),
            "eval_background_raw_ring_contrast": round(float(background_raw_contrast), 6),
            "eval_raw_ring_contrast_gain": round(float(raw_ring_contrast_gain), 6),
            "eval_raw_edge_gain": round(float(raw_edge_gain), 6),
            "eval_heuristic_quality_score": round(float(heuristic_quality_score), 6),
            "eval_pass_edit_focus": pass_edit_focus,
            "eval_pass_low_leakage": pass_low_leakage,
            "eval_pass_visible_change": pass_visible_change,
            "eval_pass_contrast_gain": pass_contrast_gain,
            "eval_pass_overall": pass_overall,
        }
    )
    return evaluated


def build_summary(*, run_dir: Path, records: list[dict], preview_count: int, thresholds: dict) -> dict:
    numeric_keys = [
        "eval_total_diff_mean",
        "eval_raw_diff_mean",
        "eval_edit_diff_mean",
        "eval_outside_edit_diff_mean",
        "eval_edit_energy_ratio",
        "eval_outside_energy_ratio",
        "eval_synth_raw_ring_contrast",
        "eval_background_raw_ring_contrast",
        "eval_raw_ring_contrast_gain",
        "eval_raw_edge_gain",
        "eval_heuristic_quality_score",
    ]
    mean_metrics = {
        key: round(float(np.mean([float(record[key]) for record in records])), 6)
        for key in numeric_keys
    }
    median_metrics = {
        key: round(float(np.median([float(record[key]) for record in records])), 6)
        for key in numeric_keys
    }
    sorted_records = sorted(records, key=lambda record: float(record["eval_heuristic_quality_score"]))
    pass_count = sum(1 for record in records if bool(record["eval_pass_overall"]))

    return {
        "generator": "crack_synth.methods.diffusion_v2.evaluate_generation",
        "run_dir": str(run_dir),
        "sample_count": len(records),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / len(records), 6) if records else 0.0,
        "thresholds": thresholds,
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "worst_samples": [
            {
                "record_id": record["record_id"],
                "heuristic_quality_score": record["eval_heuristic_quality_score"],
                "pass_overall": record["eval_pass_overall"],
                "outside_energy_ratio": record["eval_outside_energy_ratio"],
                "raw_ring_contrast_gain": record["eval_raw_ring_contrast_gain"],
            }
            for record in sorted_records[:preview_count]
        ],
    }


def _resolve_record_path(path_value: str | Path, config) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        return resolve_project_path(
            path,
            repo_root=config.repo_root,
            dataset_root=config.dataset_root,
            output_root=config.output_root,
        )

    parts = path.parts
    for index in range(len(parts)):
        candidate_parts = parts[index:]
        if not candidate_parts:
            continue
        candidate = Path(*candidate_parts)
        try:
            mapped = resolve_project_path(
                candidate,
                repo_root=config.repo_root,
                dataset_root=config.dataset_root,
                output_root=config.output_root,
            )
        except Exception:
            continue
        if mapped.exists():
            return mapped
    return path


def _pick_existing_path(candidates: list[Path | None]) -> Path:
    existing = _first_existing_path(candidates)
    if existing is not None:
        return existing
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def _first_existing_path(candidates: list[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def _load_gray(path: Path) -> np.ndarray:
    image = load_image(path)
    return np.asarray(image.convert("L"), dtype=np.float32)


def _mask_to_bool(path: Path) -> np.ndarray:
    mask = load_binary_mask(path)
    return np.asarray(mask, dtype=np.uint8) > 0


def _build_ring_mask(mask_raw_path: Path, raw_mask: np.ndarray, edit_mask: np.ndarray, ring_width_px: int) -> np.ndarray:
    if ring_width_px <= 0:
        return edit_mask & ~raw_mask
    mask_raw_image = load_binary_mask(mask_raw_path)
    dilated = dilate_binary_mask(mask_raw_image, ring_width_px)
    ring_mask = (np.asarray(dilated, dtype=np.uint8) > 0) & ~raw_mask
    if ring_mask.any():
        return ring_mask
    return edit_mask & ~raw_mask


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(image)
    return np.sqrt((grad_x ** 2) + (grad_y ** 2))


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(values[mask].mean())


def _safe_sum(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(values[mask].sum())


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


if __name__ == "__main__":
    main()
