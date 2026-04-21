from __future__ import annotations

import numpy as np
from PIL import Image


def apply_hard_overlay(
    background: Image.Image,
    generated: Image.Image,
    mask_edit: Image.Image,
) -> Image.Image:
    """Keep generated pixels only inside mask_edit and preserve background elsewhere."""
    bg = np.asarray(background.convert("RGB"), dtype=np.uint8)
    gen = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask_edit.convert("L")) > 0

    if bg.shape != gen.shape:
        raise ValueError(f"Shape mismatch: background={bg.shape}, generated={gen.shape}")

    out = bg.copy()
    out[mask] = gen[mask]
    return Image.fromarray(out, mode="RGB")


def make_abs_diff_image(
    before: Image.Image,
    after: Image.Image,
    gain: float = 4.0,
) -> Image.Image:
    before_arr = np.asarray(before.convert("RGB"), dtype=np.float32)
    after_arr = np.asarray(after.convert("RGB"), dtype=np.float32)
    diff = np.abs(after_arr - before_arr)
    diff = np.clip(diff * float(gain), 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def compute_generation_metrics(
    background: Image.Image,
    generated: Image.Image,
    mask_raw: Image.Image,
    mask_edit: Image.Image,
) -> dict[str, float]:
    bg = np.asarray(background.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated.convert("RGB"), dtype=np.float32)
    abs_diff = np.abs(gen - bg).mean(axis=2)

    raw = np.asarray(mask_raw.convert("L")) > 0
    edit = np.asarray(mask_edit.convert("L")) > 0
    outside = ~edit
    total = float(raw.shape[0] * raw.shape[1])

    return {
        "mask_raw_area_ratio": float(raw.sum() / total),
        "mask_edit_area_ratio": float(edit.sum() / total),
        "inside_edit_mae": _safe_mean(abs_diff[edit]),
        "outside_edit_mae": _safe_mean(abs_diff[outside]),
        "inside_raw_mae": _safe_mean(abs_diff[raw]),
        "outside_edit_p95": _safe_p95(abs_diff[outside]),
    }


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _safe_p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95))
