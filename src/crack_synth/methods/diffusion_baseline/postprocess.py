from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


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


def apply_soft_overlay(
    background: Image.Image,
    generated: Image.Image,
    mask_paste: Image.Image,
) -> Image.Image:
    bg = np.asarray(background.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask_paste.convert("L"), dtype=np.float32)[..., None] / 255.0

    if bg.shape != gen.shape:
        raise ValueError(f"Shape mismatch: background={bg.shape}, generated={gen.shape}")
    if alpha.shape[:2] != bg.shape[:2]:
        raise ValueError(f"Mask shape mismatch: mask={alpha.shape}, image={bg.shape}")

    out = bg * (1.0 - alpha) + gen * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


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
    bg_gray = np.asarray(background.convert("L"), dtype=np.float32)
    gen_gray = np.asarray(generated.convert("L"), dtype=np.float32)
    abs_diff = np.abs(gen - bg).mean(axis=2)

    raw = np.asarray(mask_raw.convert("L")) > 0
    edit = np.asarray(mask_edit.convert("L")) > 0
    edit_ring = edit & ~raw
    outside = ~edit
    total = float(raw.shape[0] * raw.shape[1])
    total_diff_sum = float(abs_diff.sum())
    edit_energy_ratio = float(abs_diff[edit].sum() / total_diff_sum) if total_diff_sum > 0 else 0.0
    outside_energy_ratio = float(abs_diff[outside].sum() / total_diff_sum) if total_diff_sum > 0 else 0.0

    ring = _build_ring(raw, width_px=5)
    synth_raw_contrast = abs(_safe_mean(gen_gray[raw]) - _safe_mean(gen_gray[ring]))
    background_raw_contrast = abs(_safe_mean(bg_gray[raw]) - _safe_mean(bg_gray[ring]))
    raw_ring_contrast_gain = synth_raw_contrast - background_raw_contrast

    grad_gen = _gradient_magnitude(gen_gray)
    grad_bg = _gradient_magnitude(bg_gray)
    raw_edge_gain = _safe_mean(grad_gen[raw]) - _safe_mean(grad_bg[raw])

    raw_diff_mean = _safe_mean(abs_diff[raw])
    edit_diff_mean = _safe_mean(abs_diff[edit])
    outside_edit_diff_mean = _safe_mean(abs_diff[outside])
    raw_dark_mean = _safe_mean(np.maximum(bg_gray - gen_gray, 0.0)[raw])
    edit_ring_dark_mean = _safe_mean(np.maximum(bg_gray - gen_gray, 0.0)[edit_ring])
    overfill_penalty = _clip01(max(edit_ring_dark_mean - raw_dark_mean * 0.65, 0.0) / 20.0)

    raw_diff_strength = _clip01(raw_diff_mean / 18.0)
    positive_contrast_gain = _clip01(max(raw_ring_contrast_gain, 0.0) / 8.0)
    positive_edge_gain = _clip01(max(raw_edge_gain, 0.0) / 10.0)
    background_preservation = _clip01(1.0 - outside_energy_ratio)
    visual_score = 100.0 * (
        0.30 * raw_diff_strength
        + 0.25 * positive_contrast_gain
        + 0.20 * positive_edge_gain
        + 0.15 * _clip01(edit_energy_ratio)
        + 0.10 * background_preservation
        - 0.20 * overfill_penalty
    )

    return {
        "mask_raw_area_ratio": float(raw.sum() / total),
        "mask_edit_area_ratio": float(edit.sum() / total),
        "inside_edit_mae": edit_diff_mean,
        "outside_edit_mae": outside_edit_diff_mean,
        "inside_raw_mae": raw_diff_mean,
        "outside_edit_p95": _safe_p95(abs_diff[outside]),
        "edit_energy_ratio": edit_energy_ratio,
        "outside_energy_ratio": outside_energy_ratio,
        "raw_ring_contrast_gain": float(raw_ring_contrast_gain),
        "raw_edge_gain": float(raw_edge_gain),
        "raw_diff_strength": raw_diff_strength,
        "positive_contrast_gain": positive_contrast_gain,
        "positive_edge_gain": positive_edge_gain,
        "background_preservation": background_preservation,
        "overfill_penalty": overfill_penalty,
        "visual_score": float(max(0.0, visual_score)),
    }


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _safe_p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95))


def _build_ring(mask: np.ndarray, width_px: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(size=width_px * 2 + 1))
    ring = (np.asarray(dilated) > 0) & ~mask
    if ring.any():
        return ring
    return ~mask


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
