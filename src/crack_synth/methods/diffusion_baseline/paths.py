from __future__ import annotations

from pathlib import Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _rewrite_legacy_repo_relative(
    path: Path,
    *,
    repo_root: Path,
    dataset_root: Path,
    manifests_root: Path,
    output_root: Path,
) -> Path:
    parts = path.parts
    if not parts:
        return (repo_root / path).resolve()
    if parts[0] == "dataset_new":
        return (dataset_root / Path(*parts[1:])).resolve()
    if parts[0] == "manifests":
        return (manifests_root / Path(*parts[1:])).resolve()
    if parts[:2] == ("outputs", "baseline"):
        return (output_root / Path(*parts[2:])).resolve()
    return (repo_root / path).resolve()


def resolve_project_path(
    path_value: str | Path,
    *,
    repo_root: Path,
    dataset_root: Path,
    manifests_root: Path,
    output_root: Path,
) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return _rewrite_legacy_repo_relative(
        path,
        repo_root=repo_root,
        dataset_root=dataset_root,
        manifests_root=manifests_root,
        output_root=output_root,
    )


def resolve_sample_paths(
    record: dict[str, str],
    *,
    repo_root: Path,
    dataset_root: Path,
    manifests_root: Path,
    output_root: Path,
) -> dict[str, Path | None]:
    sample_type = record.get("sample_type", "")
    image_name = record.get("image_name", "")
    image_stem = Path(image_name).stem

    raw_image_path = record.get("image_path") or ""
    raw_mask_path = record.get("mask_path") or ""
    raw_json_path = record.get("json_path") or ""
    raw_image = Path(raw_image_path) if raw_image_path else None
    raw_mask = Path(raw_mask_path) if raw_mask_path else None
    raw_json = Path(raw_json_path) if raw_json_path else None

    if sample_type == "defect":
        image_candidates = [
            dataset_root / "train" / "images" / image_name,
        ]
        mask_name = Path(raw_mask_path).name if raw_mask_path else f"{image_stem}.png"
        json_name = Path(raw_json_path).name if raw_json_path else f"{image_stem}.json"
        mask_candidates = [
            dataset_root / "train" / "masks" / mask_name,
        ]
        json_candidates = [
            dataset_root / "train" / "images" / json_name,
        ]
    elif sample_type == "normal":
        image_candidates = [
            dataset_root / "normal_crops_selected" / image_name,
        ]
        mask_candidates = []
        json_candidates = []
    else:
        image_candidates = [
            dataset_root / "val" / image_name,
        ]
        mask_candidates = []
        json_candidates = []

    if raw_image is not None:
        if raw_image.is_absolute() and _is_within(raw_image, repo_root):
            image_candidates.append(raw_image)
        elif not raw_image.is_absolute():
            image_candidates.append(
                resolve_project_path(
                    raw_image,
                    repo_root=repo_root,
                    dataset_root=dataset_root,
                    manifests_root=manifests_root,
                    output_root=output_root,
                )
            )
    if raw_mask is not None:
        if raw_mask.is_absolute() and _is_within(raw_mask, repo_root):
            mask_candidates.append(raw_mask)
        elif not raw_mask.is_absolute():
            mask_candidates.append(
                resolve_project_path(
                    raw_mask,
                    repo_root=repo_root,
                    dataset_root=dataset_root,
                    manifests_root=manifests_root,
                    output_root=output_root,
                )
            )
    if raw_json is not None:
        if raw_json.is_absolute() and _is_within(raw_json, repo_root):
            json_candidates.append(raw_json)
        elif not raw_json.is_absolute():
            json_candidates.append(
                resolve_project_path(
                    raw_json,
                    repo_root=repo_root,
                    dataset_root=dataset_root,
                    manifests_root=manifests_root,
                    output_root=output_root,
                )
            )

    return {
        "image_path": _first_existing(image_candidates),
        "mask_path": _first_existing(mask_candidates) if mask_candidates else None,
        "json_path": _first_existing(json_candidates) if json_candidates else None,
    }


def to_repo_relative(path: str | Path, repo_root: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def audit_and_resolve_records(
    records: list[dict[str, str]],
    *,
    repo_root: Path,
    dataset_root: Path,
    manifests_root: Path,
    output_root: Path,
    require_mask: bool,
) -> tuple[list[dict], dict]:
    resolved_records: list[dict] = []
    missing: list[dict] = []

    for record in records:
        resolved = resolve_sample_paths(
            record,
            repo_root=repo_root,
            dataset_root=dataset_root,
            manifests_root=manifests_root,
            output_root=output_root,
        )
        missing_fields: list[str] = []
        if resolved["image_path"] is None:
            missing_fields.append("image_path")
        if require_mask and resolved["mask_path"] is None:
            missing_fields.append("mask_path")
        if missing_fields:
            missing.append(
                {
                    "sample_id": record.get("sample_id", ""),
                    "image_name": record.get("image_name", ""),
                    "missing_fields": missing_fields,
                }
            )
            continue

        merged = dict(record)
        merged["resolved_image_path"] = str(resolved["image_path"])
        merged["resolved_mask_path"] = str(resolved["mask_path"]) if resolved["mask_path"] is not None else ""
        merged["resolved_json_path"] = str(resolved["json_path"]) if resolved["json_path"] is not None else ""
        resolved_records.append(merged)

    audit = {
        "total_records": len(records),
        "resolved_records": len(resolved_records),
        "missing_records": len(missing),
        "missing_preview": missing[:20],
    }
    if missing:
        preview = ", ".join(
            f"{item['sample_id']}({';'.join(item['missing_fields'])})"
            for item in missing[:10]
        )
        raise FileNotFoundError(
            f"Path audit failed for {len(missing)} record(s). Preview: {preview}"
        )
    return resolved_records, audit
