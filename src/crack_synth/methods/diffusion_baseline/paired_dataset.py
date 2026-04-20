from __future__ import annotations

from pathlib import Path
import csv
import json
import re

from .paths import to_repo_relative


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_paired_samples(dataset_root: Path, repo_root: Path) -> list[dict]:
    dataset_root = Path(dataset_root).resolve()
    validate_dataset_layout(dataset_root)
    mapping_path = find_pair_mapping_file(dataset_root)
    if mapping_path is not None:
        pairs = load_explicit_pairs(mapping_path, dataset_root, repo_root)
    else:
        pairs = load_sorted_pairs(dataset_root, repo_root)
    if not pairs:
        raise ValueError(f"No crack/normal pairs found under {dataset_root}")
    validate_unique_pair_ids(pairs)
    return pairs


def validate_dataset_layout(dataset_root: Path) -> None:
    missing_dirs = [name for name in ["crack", "normal"] if not (dataset_root / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Dataset root must contain crack/ and normal/ directories. Missing: {', '.join(missing_dirs)}"
        )


def find_pair_mapping_file(dataset_root: Path) -> Path | None:
    names = [
        "pairs.csv",
        "pair_mapping.csv",
        "crack_normal_pairs.csv",
        "crack_normal_mapping.csv",
        "mapping.csv",
        "pairs.json",
        "pair_mapping.json",
        "crack_normal_pairs.json",
        "crack_normal_mapping.json",
        "mapping.json",
    ]
    for name in names:
        candidate = dataset_root / name
        if candidate.exists():
            return candidate
    mapping_dir = dataset_root / "mapping"
    if mapping_dir.is_dir():
        for name in names:
            candidate = mapping_dir / name
            if candidate.exists():
                return candidate
    for candidate in sorted(dataset_root.glob("*mapping*.csv")) + sorted(dataset_root.glob("*pairs*.csv")):
        if candidate.is_file():
            return candidate
    for candidate in sorted(dataset_root.glob("*mapping*.json")) + sorted(dataset_root.glob("*pairs*.json")):
        if candidate.is_file():
            return candidate
    return None


def load_explicit_pairs(mapping_path: Path, dataset_root: Path, repo_root: Path) -> list[dict]:
    if mapping_path.suffix.lower() == ".json":
        raw_pairs = _load_json_pairs(mapping_path)
    else:
        raw_pairs = _load_csv_pairs(mapping_path)
    return [
        build_pair_record(
            defect_path=resolve_dataset_file(row["defect"], dataset_root, "crack"),
            normal_path=resolve_dataset_file(row["normal"], dataset_root, "normal"),
            repo_root=repo_root,
            pairing_source=f"mapping:{mapping_path.name}",
        )
        for row in raw_pairs
    ]


def _load_csv_pairs(mapping_path: Path) -> list[dict[str, str]]:
    defect_keys = ["defect_image", "crack_image", "defect", "crack", "image_name", "crack_name"]
    normal_keys = ["normal_image", "background_image", "normal", "background", "normal_name"]
    rows: list[dict[str, str]] = []
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            defect_value = first_nonempty(raw_row, defect_keys)
            normal_value = first_nonempty(raw_row, normal_keys)
            if defect_value and normal_value:
                rows.append({"defect": defect_value, "normal": normal_value})
    return rows


def _load_json_pairs(mapping_path: Path) -> list[dict[str, str]]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and all(isinstance(value, str) for value in data.values()):
        return [{"defect": str(key), "normal": str(value)} for key, value in data.items()]
    if isinstance(data, dict):
        data = data.get("pairs", [])
    if not isinstance(data, list):
        raise ValueError(f"Unsupported mapping JSON structure: {mapping_path}")

    rows: list[dict[str, str]] = []
    defect_keys = ["defect_image", "crack_image", "defect", "crack", "image_name", "crack_name"]
    normal_keys = ["normal_image", "background_image", "normal", "background", "normal_name"]
    for raw_row in data:
        if not isinstance(raw_row, dict):
            continue
        defect_value = first_nonempty(raw_row, defect_keys)
        normal_value = first_nonempty(raw_row, normal_keys)
        if defect_value and normal_value:
            rows.append({"defect": defect_value, "normal": normal_value})
    return rows


def load_sorted_pairs(dataset_root: Path, repo_root: Path) -> list[dict]:
    defect_paths = sorted_image_paths(dataset_root / "crack")
    normal_paths = sorted_image_paths(dataset_root / "normal")
    if len(defect_paths) != len(normal_paths):
        raise ValueError(
            f"Cannot infer one-to-one pairs: {len(defect_paths)} crack images and {len(normal_paths)} normal images."
        )
    return [
        build_pair_record(
            defect_path=defect_path,
            normal_path=normal_path,
            repo_root=repo_root,
            pairing_source="sorted_filename_order",
        )
        for defect_path, normal_path in zip(defect_paths, normal_paths, strict=True)
    ]


def sorted_image_paths(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: (extract_frame_id(path.name) if extract_frame_id(path.name) is not None else 10**12, path.name),
    )


def build_pair_record(*, defect_path: Path, normal_path: Path, repo_root: Path, pairing_source: str) -> dict:
    defect_json_path = defect_path.with_suffix(".json")
    if not defect_json_path.exists():
        raise FileNotFoundError(f"Missing crack annotation JSON: {defect_json_path}")
    defect_frame_id = extract_frame_id(defect_path.name)
    normal_frame_id = extract_frame_id(normal_path.name)
    defect_id = build_sample_id("defect", defect_path.name)
    normal_id = build_sample_id("normal", normal_path.name)
    return {
        "pair_id": f"{defect_id}__{normal_id}",
        "defect_id": defect_id,
        "normal_id": normal_id,
        "defect_image_name": defect_path.name,
        "normal_image_name": normal_path.name,
        "defect_image_path": to_repo_relative(defect_path, repo_root),
        "normal_image_path": to_repo_relative(normal_path, repo_root),
        "defect_json_path": to_repo_relative(defect_json_path, repo_root),
        "defect_frame_id": "" if defect_frame_id is None else str(defect_frame_id),
        "normal_frame_id": "" if normal_frame_id is None else str(normal_frame_id),
        "pairing_source": pairing_source,
    }


def resolve_dataset_file(value: str, dataset_root: Path, subdir: str) -> Path:
    raw_path = Path(value)
    candidates: list[Path]
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [
            dataset_root / raw_path,
            dataset_root / subdir / raw_path,
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve mapped {subdir} file: {value}")


def first_nonempty(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def validate_unique_pair_ids(pairs: list[dict]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        if pair_id in seen:
            duplicates.append(pair_id)
        seen.add(pair_id)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate pair_id values found: {preview}")


def build_sample_id(prefix: str, image_name: str) -> str:
    frame_id = extract_frame_id(image_name)
    if frame_id is not None:
        return f"{prefix}_{frame_id:06d}"
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(image_name).stem).strip("_")
    return f"{prefix}_{stem}"


def extract_frame_id(image_name: str) -> int | None:
    match = re.search(r"\d+", Path(image_name).stem)
    if match is None:
        return None
    return int(match.group(0))
