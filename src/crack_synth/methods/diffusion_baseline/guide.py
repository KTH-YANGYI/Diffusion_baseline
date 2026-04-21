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
