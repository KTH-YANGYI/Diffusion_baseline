from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


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


def load_labelme_mask(path: str | Path, image_size: tuple[int, int] | None = None) -> Image.Image:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    width = int(data.get("imageWidth") or (image_size[0] if image_size else 0))
    height = int(data.get("imageHeight") or (image_size[1] if image_size else 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot infer mask size from annotation: {path}")

    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    for shape in data.get("shapes", []):
        points = [
            (float(point[0]), float(point[1]))
            for point in shape.get("points", [])
            if len(point) >= 2
        ]
        if not points:
            continue

        shape_type = str(shape.get("shape_type") or "polygon").lower()
        if shape_type == "rectangle" and len(points) >= 2:
            x_values = [point[0] for point in points[:2]]
            y_values = [point[1] for point in points[:2]]
            draw.rectangle((min(x_values), min(y_values), max(x_values), max(y_values)), fill=255)
        elif shape_type == "circle" and len(points) >= 2:
            cx, cy = points[0]
            px, py = points[1]
            radius = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
        elif shape_type in {"line", "linestrip"} and len(points) >= 2:
            draw.line(points, fill=255, width=3)
        elif len(points) >= 3:
            draw.polygon(points, fill=255)
    return threshold_mask(mask)


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
    try:
        import cv2

        array = (np.asarray(mask) > 0).astype(np.uint8) * 255
        kernel = np.ones((radius_px * 2 + 1, radius_px * 2 + 1), dtype=np.uint8)
        dilated = cv2.dilate(array, kernel, iterations=1)
        return threshold_mask(Image.fromarray(dilated, mode="L"))
    except ImportError:
        pass
    kernel = radius_px * 2 + 1
    dilated = mask.filter(ImageFilter.MaxFilter(size=kernel))
    return threshold_mask(dilated)


def blur_mask(mask: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return mask.copy()
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))
