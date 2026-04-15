from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import random

from .config import DEFAULT_CONFIG_PATH, load_config
from .io_utils import ensure_dir, read_csv_records, write_csv_records, write_json
from .paths import to_repo_relative


MANIFEST_FIELDNAMES = [
    "sample_id",
    "image_name",
    "image_path",
    "mask_path",
    "json_path",
    "sample_type",
    "is_labeled",
    "source_split",
    "video_id",
    "video_name",
    "frame_id",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Rebuild manifests from the configured dataset root.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--n-folds", type=int, default=4, help="Number of video-group folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold construction.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    repo_root = config.repo_root
    dataset_root = config.dataset_root
    manifest_dir = ensure_dir(config.manifests_root)

    defect_rows_raw = scan_labeled_defects(dataset_root, repo_root)
    holdout_rows_raw = scan_unlabeled_holdout(dataset_root, repo_root)
    normal_rows_raw = scan_normals(dataset_root, repo_root)

    defect_train_mapping = load_mapping_by_image_name(dataset_root / "图片来源视频映射" / "train_image_to_video.csv")
    defect_holdout_mapping = load_mapping_by_image_name(dataset_root / "图片来源视频映射" / "val_image_to_video.csv")
    normal_mapping = load_mapping_by_image_name(dataset_root / "normal_crops_selected" / "video_mapping.csv")

    defect_rows, defect_missing = attach_video_info(defect_rows_raw, defect_train_mapping)
    holdout_rows, holdout_missing = attach_video_info(holdout_rows_raw, defect_holdout_mapping)
    normal_rows, normal_missing = attach_video_info(normal_rows_raw, normal_mapping)

    raise_if_missing_mapping("defect_train_mapping", defect_missing)
    raise_if_missing_mapping("defect_holdout_mapping", holdout_missing)
    raise_if_missing_mapping("normal_mapping", normal_missing)

    write_json(
        manifest_dir / "data_audit.json",
        build_data_audit(
            defect_rows_raw=defect_rows_raw,
            holdout_rows_raw=holdout_rows_raw,
            normal_rows_raw=normal_rows_raw,
            defect_rows=defect_rows,
            holdout_rows=holdout_rows,
            normal_rows=normal_rows,
            defect_missing=defect_missing,
            holdout_missing=holdout_missing,
            normal_missing=normal_missing,
        ),
    )

    master_manifest = build_master_manifest(defect_rows, holdout_rows, normal_rows)
    write_manifest_csv(manifest_dir / "master_manifest.csv", sort_rows_for_manifest(master_manifest))
    write_manifest_csv(manifest_dir / "defect_labeled.csv", sort_rows_for_manifest(defect_rows))
    write_manifest_csv(manifest_dir / "defect_holdout_unlabeled.csv", sort_rows_for_manifest(holdout_rows))
    write_manifest_csv(manifest_dir / "normal_pool.csv", sort_rows_for_manifest(normal_rows))

    future_holdout_video_ids = collect_unique_video_ids(holdout_rows)
    future_holdout_video_id_set = set(future_holdout_video_ids)
    normal_future_holdout_rows = sort_rows_for_manifest(
        [row for row in normal_rows if str(row["video_id"]).strip() in future_holdout_video_id_set]
    )
    normal_trainable_rows = sort_rows_for_manifest(
        [row for row in normal_rows if str(row["video_id"]).strip() not in future_holdout_video_id_set]
    )
    write_manifest_csv(manifest_dir / "normal_future_holdout.csv", normal_future_holdout_rows)

    defect_folds = build_rows_by_video_folds(defect_rows, n_folds=args.n_folds, seed=args.seed)
    normal_folds = build_rows_by_video_folds(normal_trainable_rows, n_folds=args.n_folds, seed=args.seed + 1000)

    summary = {
        "generator": "crack_synth.methods.diffusion_baseline.build_manifests",
        "dataset_root": str(dataset_root),
        "manifests_root": str(manifest_dir),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "master_manifest_count": len(master_manifest),
        "defect_labeled_count": len(defect_rows),
        "defect_holdout_unlabeled_count": len(holdout_rows),
        "normal_pool_count": len(normal_rows),
        "normal_future_holdout_count": len(normal_future_holdout_rows),
        "future_holdout_video_ids": future_holdout_video_ids,
        "normal_fold_seed": args.seed + 1000,
        "folds": [],
    }

    for fold_index in range(args.n_folds):
        defect_train_rows, defect_val_rows, defect_train_video_ids, defect_val_video_ids = build_train_val_rows_for_fold(
            folds=defect_folds,
            val_fold_index=fold_index,
        )
        normal_train_rows, normal_val_rows, normal_train_video_ids, normal_val_video_ids = build_train_val_rows_for_fold(
            folds=normal_folds,
            val_fold_index=fold_index,
        )

        write_manifest_csv(manifest_dir / f"defect_fold{fold_index}_train.csv", defect_train_rows)
        write_manifest_csv(manifest_dir / f"defect_fold{fold_index}_val.csv", defect_val_rows)
        write_manifest_csv(manifest_dir / f"normal_fold{fold_index}_train.csv", normal_train_rows)
        write_manifest_csv(manifest_dir / f"normal_fold{fold_index}_val.csv", normal_val_rows)

        summary["folds"].append(
            {
                "fold_index": fold_index,
                "defect_train_count": len(defect_train_rows),
                "defect_val_count": len(defect_val_rows),
                "normal_train_count": len(normal_train_rows),
                "normal_val_count": len(normal_val_rows),
                "defect_train_video_ids": defect_train_video_ids,
                "defect_val_video_ids": defect_val_video_ids,
                "normal_train_video_ids": normal_train_video_ids,
                "normal_val_video_ids": normal_val_video_ids,
            }
        )

    write_json(manifest_dir / "split_summary.json", summary)


def build_sample_id(prefix: str, image_name: str) -> str:
    return f"{prefix}_{Path(image_name).stem}"


def scan_labeled_defects(dataset_root: Path, repo_root: Path) -> list[dict]:
    image_dir = dataset_root / "train" / "images"
    mask_dir = dataset_root / "train" / "masks"
    rows: list[dict] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        stem = image_path.stem
        json_path = image_dir / f"{stem}.json"
        mask_path = mask_dir / f"{stem}.png"
        if not json_path.exists() or not mask_path.exists():
            continue
        rows.append(
            {
                "sample_id": build_sample_id("defect", image_path.name),
                "image_name": image_path.name,
                "image_path": to_repo_relative(image_path, repo_root),
                "mask_path": to_repo_relative(mask_path, repo_root),
                "json_path": to_repo_relative(json_path, repo_root),
                "sample_type": "defect",
                "is_labeled": True,
                "source_split": "train",
            }
        )
    return rows


def scan_unlabeled_holdout(dataset_root: Path, repo_root: Path) -> list[dict]:
    rows: list[dict] = []
    for image_path in sorted((dataset_root / "val").glob("*.jpg")):
        rows.append(
            {
                "sample_id": build_sample_id("holdout", image_path.name),
                "image_name": image_path.name,
                "image_path": to_repo_relative(image_path, repo_root),
                "mask_path": "",
                "json_path": "",
                "sample_type": "defect_holdout_unlabeled",
                "is_labeled": False,
                "source_split": "val",
            }
        )
    return rows


def scan_normals(dataset_root: Path, repo_root: Path) -> list[dict]:
    rows: list[dict] = []
    for image_path in sorted((dataset_root / "normal_crops_selected").glob("*.jpg")):
        rows.append(
            {
                "sample_id": build_sample_id("normal", image_path.name),
                "image_name": image_path.name,
                "image_path": to_repo_relative(image_path, repo_root),
                "mask_path": "",
                "json_path": "",
                "sample_type": "normal",
                "is_labeled": False,
                "source_split": "normal_pool",
            }
        )
    return rows


def load_mapping_by_image_name(path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for row in read_csv_records(path):
        image_name = str(row.get("image_name", "")).strip()
        if image_name:
            mapping[image_name] = row
    return mapping


def attach_video_info(rows: list[dict], mapping_dict: dict[str, dict[str, str]]) -> tuple[list[dict], list[str]]:
    attached_rows: list[dict] = []
    missing_image_names: list[str] = []
    for row in rows:
        image_name = row["image_name"]
        mapping_row = mapping_dict.get(image_name)
        if mapping_row is None:
            missing_image_names.append(image_name)
            continue
        frame_id = mapping_row.get("frame_id", "")
        if str(frame_id).strip() == "":
            frame_id = mapping_row.get("source_frame_idx", "")
        new_row = dict(row)
        new_row["video_id"] = str(mapping_row.get("video_id", "")).strip()
        new_row["video_name"] = str(mapping_row.get("video_name", "")).strip()
        new_row["frame_id"] = str(frame_id).strip()
        attached_rows.append(new_row)
    return attached_rows, missing_image_names


def video_id_to_int(video_id: str) -> int:
    return int(str(video_id).strip())


def sort_video_id_list(video_id_list: list[str]) -> list[str]:
    return sorted(video_id_list, key=video_id_to_int)


def build_master_manifest(defect_rows: list[dict], holdout_rows: list[dict], normal_rows: list[dict]) -> list[dict]:
    return [*defect_rows, *holdout_rows, *normal_rows]


def group_rows_by_video_id(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        video_id = str(row["video_id"]).strip()
        grouped.setdefault(video_id, []).append(row)
    return grouped


def group_sample_count(group_info: dict) -> int:
    return int(group_info["sample_count"])


def choose_fold_with_smallest_sample_count(folds: list[dict]) -> dict:
    best_fold = None
    for fold in folds:
        if best_fold is None:
            best_fold = fold
            continue
        if fold["sample_count"] < best_fold["sample_count"]:
            best_fold = fold
            continue
        if fold["sample_count"] == best_fold["sample_count"]:
            if len(fold["video_ids"]) < len(best_fold["video_ids"]):
                best_fold = fold
                continue
            if len(fold["video_ids"]) == len(best_fold["video_ids"]) and fold["fold_index"] < best_fold["fold_index"]:
                best_fold = fold
    if best_fold is None:
        raise ValueError("No folds available.")
    return best_fold


def build_rows_by_video_folds(rows: list[dict], *, n_folds: int, seed: int) -> list[dict]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    grouped = group_rows_by_video_id(rows)
    video_ids = list(grouped.keys())
    if len(video_ids) < n_folds:
        raise ValueError("Not enough video groups to build folds")

    video_groups = [
        {
            "video_id": video_id,
            "rows": grouped[video_id],
            "sample_count": len(grouped[video_id]),
        }
        for video_id in video_ids
    ]
    rng = random.Random(seed)
    rng.shuffle(video_groups)
    video_groups = sorted(video_groups, key=group_sample_count, reverse=True)

    folds = [
        {"fold_index": fold_index, "video_ids": [], "rows": [], "sample_count": 0}
        for fold_index in range(n_folds)
    ]
    for group_info in video_groups:
        target_fold = choose_fold_with_smallest_sample_count(folds)
        target_fold["video_ids"].append(group_info["video_id"])
        target_fold["rows"].extend(group_info["rows"])
        target_fold["sample_count"] += group_info["sample_count"]
    for fold in folds:
        fold["video_ids"] = sort_video_id_list(fold["video_ids"])
    return folds


def build_train_val_rows_for_fold(
    *,
    folds: list[dict],
    val_fold_index: int,
) -> tuple[list[dict], list[dict], list[str], list[str]]:
    if val_fold_index < 0 or val_fold_index >= len(folds):
        raise ValueError("val_fold_index out of range")
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    train_video_ids: list[str] = []
    val_video_ids: list[str] = []

    for fold in folds:
        if fold["fold_index"] == val_fold_index:
            val_rows.extend(fold["rows"])
            val_video_ids.extend(fold["video_ids"])
        else:
            train_rows.extend(fold["rows"])
            train_video_ids.extend(fold["video_ids"])

    train_video_ids = sort_video_id_list(list(set(train_video_ids)))
    val_video_ids = sort_video_id_list(list(set(val_video_ids)))
    if not set(train_video_ids).isdisjoint(set(val_video_ids)):
        raise ValueError("Train/val video overlap detected")
    return (
        sort_rows_for_manifest(train_rows),
        sort_rows_for_manifest(val_rows),
        train_video_ids,
        val_video_ids,
    )


def sort_rows_for_manifest(rows: list[dict]) -> list[dict]:
    def sort_key(row: dict) -> tuple[int, str, str]:
        video_id = str(row.get("video_id", "")).strip()
        video_sort_value = -1 if video_id == "" else video_id_to_int(video_id)
        return (video_sort_value, row.get("image_name", ""), row.get("sample_id", ""))

    return sorted(rows, key=sort_key)


def collect_unique_video_ids(rows: list[dict]) -> list[str]:
    return sort_video_id_list(list({str(row["video_id"]).strip() for row in rows}))


def count_duplicate_values(rows: list[dict], key: str) -> int:
    counter = Counter(str(row.get(key, "")).strip() for row in rows)
    return sum(1 for value, count in counter.items() if value != "" and count > 1)


def build_video_distribution(rows: list[dict]) -> dict[str, int]:
    counter = Counter(str(row.get("video_id", "")).strip() for row in rows if str(row.get("video_id", "")).strip() != "")
    return {video_id: int(counter[video_id]) for video_id in sort_video_id_list(list(counter.keys()))}


def build_data_audit(
    *,
    defect_rows_raw: list[dict],
    holdout_rows_raw: list[dict],
    normal_rows_raw: list[dict],
    defect_rows: list[dict],
    holdout_rows: list[dict],
    normal_rows: list[dict],
    defect_missing: list[str],
    holdout_missing: list[str],
    normal_missing: list[str],
) -> dict:
    return {
        "generator": "crack_synth.methods.diffusion_baseline.build_manifests",
        "raw_scan_counts": {
            "defect": len(defect_rows_raw),
            "holdout": len(holdout_rows_raw),
            "normal": len(normal_rows_raw),
        },
        "attached_counts": {
            "defect": len(defect_rows),
            "holdout": len(holdout_rows),
            "normal": len(normal_rows),
        },
        "unique_video_id_counts": {
            "defect": len(collect_unique_video_ids(defect_rows)),
            "holdout": len(collect_unique_video_ids(holdout_rows)),
            "normal": len(collect_unique_video_ids(normal_rows)),
        },
        "unique_image_name_counts": {
            "defect": len({row["image_name"] for row in defect_rows}),
            "holdout": len({row["image_name"] for row in holdout_rows}),
            "normal": len({row["image_name"] for row in normal_rows}),
        },
        "missing_mapping_counts": {
            "defect": len(defect_missing),
            "holdout": len(holdout_missing),
            "normal": len(normal_missing),
        },
        "missing_mapping_preview": {
            "defect": defect_missing[:10],
            "holdout": holdout_missing[:10],
            "normal": normal_missing[:10],
        },
        "duplicate_image_name_counts": {
            "defect": count_duplicate_values(defect_rows, "image_name"),
            "holdout": count_duplicate_values(holdout_rows, "image_name"),
            "normal": count_duplicate_values(normal_rows, "image_name"),
        },
        "duplicate_sample_id_counts": {
            "defect": count_duplicate_values(defect_rows, "sample_id"),
            "holdout": count_duplicate_values(holdout_rows, "sample_id"),
            "normal": count_duplicate_values(normal_rows, "sample_id"),
        },
        "per_video_sample_count": {
            "defect": build_video_distribution(defect_rows),
            "holdout": build_video_distribution(holdout_rows),
            "normal": build_video_distribution(normal_rows),
        },
    }


def raise_if_missing_mapping(mapping_name: str, missing_image_names: list[str]) -> None:
    if not missing_image_names:
        return
    preview = missing_image_names[:10]
    raise ValueError(f"{mapping_name} missing {len(missing_image_names)} images, preview: {preview}")


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    write_csv_records(path, [{field: row.get(field, "") for field in MANIFEST_FIELDNAMES} for row in rows])


if __name__ == "__main__":
    main()
