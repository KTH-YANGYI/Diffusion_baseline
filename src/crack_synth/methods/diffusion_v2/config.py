from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG_PATH = "configs/methods/diffusion_v2/contact_wire_v2.yaml"


@dataclass(frozen=True)
class DiffusionV2Config:
    repo_root: Path
    dataset_root: Path
    model_id_or_path: str
    output_root: Path
    roi_out_size: int = 512
    mask_edit_dilate_px: int = 3
    mask_paste_dilate_px: int = 2
    mask_paste_blur_px: float = 1.0
    seeds_per_pair: int = 4
    inference_batch_size: int = 1
    planning_seed: int = 20260413
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 35
    guidance_scale: float = 5.5
    strength: float = 0.88
    mask_blur: float = 1.0
    local_inpaint: bool = False
    local_crop_size: int = 192
    background_overlay: bool = True
    overlay_blur_px: float = 2.0
    padding_mask_crop: int = 64
    apply_overlay: bool = True
    save_guide_image: bool = True
    save_diff_abs: bool = True
    save_quality_metrics: bool = True
    guide_crack: bool = False
    guide_darken: float = 45.0
    guide_blur: float = 0.6
    guide_label_dilate_px: int = 1
    crack_prior_mode: str = "none"
    crack_prior_alpha: float = 0.65
    crack_prior_clip_percentile: float = 95.0
    crack_prior_mask_dilate_px: int = 2
    crack_prior_blur_px: float = 0.4
    candidate_local_crop_sizes: tuple[int, ...] = ()
    candidate_strengths: tuple[float, ...] = ()
    candidate_residual_alphas: tuple[float, ...] = ()
    candidate_seed_count: int = 0
    candidate_guidance_scale: float | None = None
    candidate_lora_scale: float | None = None
    top_candidates_per_pair: int = 3
    debug_output_dir_name: str = "debug"
    lora_path: str = ""
    lora_scale: float = 0.75
    lora_adapter_name: str = "ctwirecrack"
    device: str = "auto"
    dtype: str = "float16"
    local_files_only: bool = False
    enable_attention_slicing: bool = True
    enable_xformers: bool = False
    disable_safety_checker: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repo_root"] = str(self.repo_root)
        data["dataset_root"] = str(self.dataset_root)
        data["output_root"] = str(self.output_root)
        return data


def _resolve_path(base_dir: Path, raw_value: str | None, *, default: Path | None = None) -> Path:
    if raw_value is None:
        if default is None:
            raise ValueError("Missing required path value.")
        return default.resolve()
    path = Path(raw_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _discover_repo_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate.resolve()
    return config_path.parent.resolve()


def load_config(config_path: str | Path) -> DiffusionV2Config:
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent
    raw = _load_mapping(config_path) or {}

    repo_root = _resolve_path(base_dir, raw.get("repo_root"), default=_discover_repo_root(config_path))
    dataset_root = _resolve_path(base_dir, raw.get("dataset_root"), default=repo_root.parent / "dataset_real")
    output_root = _resolve_path(
        base_dir,
        raw.get("output_root"),
        default=repo_root / "artifacts" / "dataset_real" / "methods" / "diffusion_v2",
    )
    config = DiffusionV2Config(
        repo_root=repo_root,
        dataset_root=dataset_root,
        model_id_or_path=str(raw.get("model_id_or_path", "stable-diffusion-v1-5/stable-diffusion-inpainting")),
        output_root=output_root,
        roi_out_size=int(raw.get("roi_out_size", 512)),
        mask_edit_dilate_px=int(raw.get("mask_edit_dilate_px", 3)),
        mask_paste_dilate_px=int(raw.get("mask_paste_dilate_px", 2)),
        mask_paste_blur_px=float(raw.get("mask_paste_blur_px", 1.0)),
        seeds_per_pair=int(raw.get("seeds_per_pair", 4)),
        inference_batch_size=int(raw.get("inference_batch_size", 1)),
        planning_seed=int(raw.get("planning_seed", 20260413)),
        prompt=str(raw.get("prompt", "")),
        negative_prompt=str(raw.get("negative_prompt", "")),
        num_inference_steps=int(raw.get("num_inference_steps", 35)),
        guidance_scale=float(raw.get("guidance_scale", 5.5)),
        strength=float(raw.get("strength", 0.88)),
        mask_blur=float(raw.get("mask_blur", 1.0)),
        local_inpaint=bool(raw.get("local_inpaint", False)),
        local_crop_size=int(raw.get("local_crop_size", 192)),
        background_overlay=bool(raw.get("background_overlay", True)),
        overlay_blur_px=float(raw.get("overlay_blur_px", 2.0)),
        padding_mask_crop=int(raw.get("padding_mask_crop", 64)),
        apply_overlay=bool(raw.get("apply_overlay", raw.get("background_overlay", True))),
        save_guide_image=bool(raw.get("save_guide_image", True)),
        save_diff_abs=bool(raw.get("save_diff_abs", True)),
        save_quality_metrics=bool(raw.get("save_quality_metrics", True)),
        guide_crack=bool(raw.get("guide_crack", False)),
        guide_darken=float(raw.get("guide_darken", 45.0)),
        guide_blur=float(raw.get("guide_blur", 0.6)),
        guide_label_dilate_px=int(raw.get("guide_label_dilate_px", 1)),
        crack_prior_mode=str(raw.get("crack_prior_mode", "mask_dark" if raw.get("guide_crack", False) else "none")),
        crack_prior_alpha=float(raw.get("crack_prior_alpha", 0.65)),
        crack_prior_clip_percentile=float(raw.get("crack_prior_clip_percentile", 95.0)),
        crack_prior_mask_dilate_px=int(raw.get("crack_prior_mask_dilate_px", 2)),
        crack_prior_blur_px=float(raw.get("crack_prior_blur_px", 0.4)),
        candidate_local_crop_sizes=_coerce_int_tuple(raw.get("candidate_local_crop_sizes", ())),
        candidate_strengths=_coerce_float_tuple(raw.get("candidate_strengths", ())),
        candidate_residual_alphas=_coerce_float_tuple(raw.get("candidate_residual_alphas", ())),
        candidate_seed_count=int(raw.get("candidate_seed_count", 0)),
        candidate_guidance_scale=_coerce_optional_float(raw.get("candidate_guidance_scale")),
        candidate_lora_scale=_coerce_optional_float(raw.get("candidate_lora_scale")),
        top_candidates_per_pair=int(raw.get("top_candidates_per_pair", 3)),
        debug_output_dir_name=str(raw.get("debug_output_dir_name", "debug")),
        lora_path=_resolve_optional_path_string(base_dir, raw.get("lora_path")),
        lora_scale=float(raw.get("lora_scale", 0.75)),
        lora_adapter_name=str(raw.get("lora_adapter_name", "ctwirecrack")),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float16")),
        local_files_only=bool(raw.get("local_files_only", False)),
        enable_attention_slicing=bool(raw.get("enable_attention_slicing", True)),
        enable_xformers=bool(raw.get("enable_xformers", False)),
        disable_safety_checker=bool(raw.get("disable_safety_checker", False)),
    )
    _validate_config(config)
    return config


def _validate_config(config: DiffusionV2Config) -> None:
    if config.roi_out_size <= 0:
        raise ValueError("roi_out_size must be positive.")
    if config.mask_edit_dilate_px < 0:
        raise ValueError("mask_edit_dilate_px cannot be negative.")
    if config.mask_paste_dilate_px < 0:
        raise ValueError("mask_paste_dilate_px cannot be negative.")
    if config.mask_paste_blur_px < 0:
        raise ValueError("mask_paste_blur_px cannot be negative.")
    if config.seeds_per_pair <= 0:
        raise ValueError("seeds_per_pair must be positive.")
    if config.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive.")
    if config.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    if not (0.0 <= config.strength <= 1.0):
        raise ValueError("strength must be in [0, 1].")
    if config.local_crop_size <= 0:
        raise ValueError("local_crop_size must be positive.")
    if config.local_crop_size > config.roi_out_size:
        raise ValueError("local_crop_size cannot exceed roi_out_size.")
    if config.overlay_blur_px < 0:
        raise ValueError("overlay_blur_px cannot be negative.")
    if config.padding_mask_crop < 0:
        raise ValueError("padding_mask_crop cannot be negative.")
    if config.guide_darken < 0:
        raise ValueError("guide_darken cannot be negative.")
    if config.guide_blur < 0:
        raise ValueError("guide_blur cannot be negative.")
    if config.guide_label_dilate_px < 0:
        raise ValueError("guide_label_dilate_px cannot be negative.")
    if config.crack_prior_mode not in {"none", "mask_dark", "paired_residual"}:
        raise ValueError("crack_prior_mode must be one of: none, mask_dark, paired_residual.")
    if config.crack_prior_alpha < 0:
        raise ValueError("crack_prior_alpha cannot be negative.")
    if not (0 < config.crack_prior_clip_percentile <= 100):
        raise ValueError("crack_prior_clip_percentile must be in (0, 100].")
    if config.crack_prior_mask_dilate_px < 0:
        raise ValueError("crack_prior_mask_dilate_px cannot be negative.")
    if config.crack_prior_blur_px < 0:
        raise ValueError("crack_prior_blur_px cannot be negative.")
    for crop_size in config.candidate_local_crop_sizes:
        if crop_size <= 0:
            raise ValueError("candidate_local_crop_sizes values must be positive.")
        if crop_size > config.roi_out_size:
            raise ValueError("candidate_local_crop_sizes cannot exceed roi_out_size.")
    for strength in config.candidate_strengths:
        if not (0.0 <= strength <= 1.0):
            raise ValueError("candidate_strengths values must be in [0, 1].")
    for alpha in config.candidate_residual_alphas:
        if alpha < 0:
            raise ValueError("candidate_residual_alphas values cannot be negative.")
    if config.candidate_seed_count < 0:
        raise ValueError("candidate_seed_count cannot be negative.")
    if config.candidate_guidance_scale is not None and config.candidate_guidance_scale <= 0:
        raise ValueError("candidate_guidance_scale must be positive when provided.")
    if config.candidate_lora_scale is not None and not (0.0 <= config.candidate_lora_scale <= 2.0):
        raise ValueError("candidate_lora_scale should normally be in [0, 2].")
    if config.top_candidates_per_pair <= 0:
        raise ValueError("top_candidates_per_pair must be positive.")
    if not (0.0 <= config.lora_scale <= 2.0):
        raise ValueError("lora_scale should normally be in [0, 2].")


def dump_resolved_config(config: DiffusionV2Config, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _minimal_yaml_load(text)


def _resolve_optional_path_string(base_dir: Path, raw_value: Any) -> str:
    if raw_value is None:
        return ""
    value = str(raw_value).strip()
    if not value:
        return ""
    return str(_resolve_path(base_dir, value))


def _coerce_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return ()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _coerce_float_tuple(value: Any) -> tuple[float, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return ()
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(value)


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        data[key.strip()] = _coerce_scalar(value.strip())
    return data


def _coerce_scalar(value: str) -> Any:
    if value == "":
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
