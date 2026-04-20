from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG_PATH = "configs/methods/diffusion_baseline/contact_wire_v1.yaml"


@dataclass(frozen=True)
class BaselineConfig:
    repo_root: Path
    dataset_root: Path
    model_id_or_path: str
    output_root: Path
    roi_out_size: int = 512
    mask_edit_dilate_px: int = 3
    seeds_per_pair: int = 4
    inference_batch_size: int = 1
    planning_seed: int = 20260413
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 35
    guidance_scale: float = 5.5
    strength: float = 0.88
    mask_blur: float = 1.0
    device: str = "auto"
    dtype: str = "float16"
    local_files_only: bool = False
    enable_attention_slicing: bool = True
    enable_xformers: bool = False

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


def load_config(config_path: str | Path) -> BaselineConfig:
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent
    raw = _load_mapping(config_path) or {}

    repo_root = _resolve_path(base_dir, raw.get("repo_root"), default=_discover_repo_root(config_path))
    dataset_root = _resolve_path(base_dir, raw.get("dataset_root"), default=repo_root.parent / "dataset_real")
    output_root = _resolve_path(
        base_dir,
        raw.get("output_root"),
        default=repo_root / "artifacts" / "dataset_real" / "methods" / "diffusion_baseline",
    )
    config = BaselineConfig(
        repo_root=repo_root,
        dataset_root=dataset_root,
        model_id_or_path=str(raw.get("model_id_or_path", "stable-diffusion-v1-5/stable-diffusion-inpainting")),
        output_root=output_root,
        roi_out_size=int(raw.get("roi_out_size", 512)),
        mask_edit_dilate_px=int(raw.get("mask_edit_dilate_px", 3)),
        seeds_per_pair=int(raw.get("seeds_per_pair", 4)),
        inference_batch_size=int(raw.get("inference_batch_size", 1)),
        planning_seed=int(raw.get("planning_seed", 20260413)),
        prompt=str(raw.get("prompt", "")),
        negative_prompt=str(raw.get("negative_prompt", "")),
        num_inference_steps=int(raw.get("num_inference_steps", 35)),
        guidance_scale=float(raw.get("guidance_scale", 5.5)),
        strength=float(raw.get("strength", 0.88)),
        mask_blur=float(raw.get("mask_blur", 1.0)),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float16")),
        local_files_only=bool(raw.get("local_files_only", False)),
        enable_attention_slicing=bool(raw.get("enable_attention_slicing", True)),
        enable_xformers=bool(raw.get("enable_xformers", False)),
    )
    _validate_config(config)
    return config


def _validate_config(config: BaselineConfig) -> None:
    if config.roi_out_size <= 0:
        raise ValueError("roi_out_size must be positive.")
    if config.mask_edit_dilate_px < 0:
        raise ValueError("mask_edit_dilate_px cannot be negative.")
    if config.seeds_per_pair <= 0:
        raise ValueError("seeds_per_pair must be positive.")
    if config.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive.")
    if config.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    if not (0.0 <= config.strength <= 1.0):
        raise ValueError("strength must be in [0, 1].")


def dump_resolved_config(config: BaselineConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _minimal_yaml_load(text)


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
