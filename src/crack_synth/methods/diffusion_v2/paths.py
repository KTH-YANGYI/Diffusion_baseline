from __future__ import annotations

from pathlib import Path


def resolve_project_path(
    path_value: str | Path,
    *,
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    parts = path.parts
    if parts and parts[0] == "dataset_new":
        return (dataset_root / Path(*parts[1:])).resolve()
    return (repo_root / path).resolve()


def to_repo_relative(path: str | Path, repo_root: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)
