from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class ResizeMetadata:
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode == "RGB":
            return image.copy()
        gray = image.convert("L")
        return Image.merge("RGB", (gray, gray, gray))


def load_binary_mask(path: str | Path) -> Image.Image:
    with Image.open(path) as mask:
        return threshold_mask(mask.convert("L"))


def threshold_mask(mask: Image.Image) -> Image.Image:
    array = (np.array(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(array, mode="L")


def compute_union_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    array = np.array(mask)
    ys, xs = np.where(array > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask has no positive pixels.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width, height = image_size
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def fixed_square_crop_box_from_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    crop_size: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if crop_size > width or crop_size > height:
        raise ValueError("crop_size cannot exceed source image size.")

    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half = crop_size / 2.0

    crop_x0 = int(round(cx - half))
    crop_y0 = int(round(cy - half))

    crop_x0 = max(0, min(crop_x0, width - crop_size))
    crop_y0 = max(0, min(crop_y0, height - crop_size))
    crop_x1 = crop_x0 + crop_size
    crop_y1 = crop_y0 + crop_size
    return crop_x0, crop_y0, crop_x1, crop_y1


def crop_pair(
    image: Image.Image,
    mask: Image.Image,
    bbox: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image]:
    return image.crop(bbox), mask.crop(bbox)


def resize_with_padding(
    image: Image.Image,
    *,
    out_size: int,
    resample: int,
    fill: int | tuple[int, int, int] = 0,
) -> tuple[Image.Image, ResizeMetadata]:
    src_w, src_h = image.size
    scale = min(out_size / src_w, out_size / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = image.resize((dst_w, dst_h), resample=resample)
    canvas = Image.new(image.mode, (out_size, out_size), color=fill)
    pad_left = (out_size - dst_w) // 2
    pad_top = (out_size - dst_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    meta = ResizeMetadata(
        scale=float(scale),
        resized_width=dst_w,
        resized_height=dst_h,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=out_size - dst_w - pad_left,
        pad_bottom=out_size - dst_h - pad_top,
    )
    return canvas, meta


def dilate_binary_mask(mask: Image.Image, radius_px: int) -> Image.Image:
    if radius_px <= 0:
        return threshold_mask(mask)
    kernel = radius_px * 2 + 1
    dilated = mask.filter(ImageFilter.MaxFilter(size=kernel))
    return threshold_mask(dilated)


def blur_mask(mask: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return mask.copy()
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))
