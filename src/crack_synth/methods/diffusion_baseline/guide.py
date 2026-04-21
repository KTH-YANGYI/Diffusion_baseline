from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def make_crack_guide(
    background: Image.Image,
    mask_label: Image.Image,
    *,
    darken: float = 45.0,
    blur_radius: float = 0.6,
    label_dilate_px: int = 1,
) -> Image.Image:
    """Draw a weak dark spatial hint along the raw crack label mask."""
    bg = np.asarray(background.convert("RGB"), dtype=np.float32)

    mask = mask_label.convert("L")
    if label_dilate_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=label_dilate_px * 2 + 1))
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    guided = np.clip(bg - float(darken) * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(guided, mode="RGB")


def make_paired_residual_crack_guide(
    background: Image.Image,
    defect: Image.Image,
    mask_label: Image.Image,
    *,
    alpha: float = 0.65,
    clip_percentile: float = 95.0,
    mask_dilate_px: int = 2,
    blur_radius: float = 0.4,
) -> Image.Image:
    """Transfer the dark crack residual from a paired defect ROI onto its normal ROI."""
    bg_rgb = np.asarray(background.convert("RGB"), dtype=np.float32)
    bg_gray = np.asarray(background.convert("L"), dtype=np.float32)
    defect_gray = np.asarray(defect.convert("L"), dtype=np.float32)

    residual = np.maximum(bg_gray - defect_gray, 0.0)
    mask = mask_label.convert("L")
    if mask_dilate_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=mask_dilate_px * 2 + 1))
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_alpha = np.asarray(mask, dtype=np.float32) / 255.0

    active = residual[mask_alpha > 0.01]
    if active.size:
        cap = float(np.percentile(active, float(clip_percentile)))
        if cap > 0:
            residual = np.clip(residual, 0.0, cap)

    guided = bg_rgb - float(alpha) * residual[..., None] * mask_alpha[..., None]
    guided = np.clip(guided, 0, 255).astype(np.uint8)
    return Image.fromarray(guided, mode="RGB")
