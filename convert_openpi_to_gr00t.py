#!/usr/bin/env python
"""Convert an OpenPI-style LeRobot v2 dataset to GR00T LeRobot format.

The OpenPI conversion used by this repository stores low-dimensional data as
``state`` and ``actions`` and may keep camera frames in parquet image columns.
GR00T expects ``observation.state``, ``action``, annotation columns, a
``meta/modality.json`` file, and video observations under ``videos/``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

try:
    import imageio.v3 as iio
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by users' envs
    missing = exc.name or str(exc)
    raise SystemExit(
        f"Missing dependency '{missing}'. Install pyarrow and imageio first, for example:\n"
        "  pip install pyarrow imageio[ffmpeg]"
    ) from exc


DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
TASK_ANNOTATION_KEY = "annotation.human.action.task_description"
VALIDITY_ANNOTATION_KEY = "annotation.human.validity"

IMAGE_KEY_MAP = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "observation.images.image": "observation.images.image",
    "observation.images.wrist_image": "observation.images.wrist_image",
}
LOW_DIM_KEY_MAP = {
    "state": "observation.state",
    "observation.state": "observation.state",
    "actions": "action",
    "action": "action",
}
DEFAULT_STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
DEFAULT_ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OpenPI-style LeRobot dataset into GR00T LeRobot format."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the existing OpenPI-style LeRobot dataset root containing meta/ and data/.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Output dataset path. Defaults to '<input_dir>_gr00t'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir if it already exists.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video FPS. Defaults to meta/info.json fps, or 10 if absent.",
    )
    parser.add_argument(
        "--no-validity-annotation",
        action="store_true",
        help="Do not add annotation.human.validity and the 'valid' task entry.",
    )
    parser.add_argument(
        "--keep-image-columns",
        action="store_true",
        help="Keep original image columns in parquet after exporting mp4 files.",
    )
    parser.add_argument(
        "--skip-video-export",
        action="store_true",
        help="Only update metadata and parquet low-dimensional columns; do not write mp4 files.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Do not generate meta/stats.json or meta/relative_stats.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
        file.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_gr00t")


def prepare_output(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")
    if not (input_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Input dataset is missing meta/info.json: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input_dir and output_dir must be different. Use a new output directory.")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)
    legacy_episode_stats = output_dir / "meta" / "episodes_stats.jsonl"
    if legacy_episode_stats.exists():
        legacy_episode_stats.unlink()


def episode_index_from_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Unexpected parquet filename: {path}")
    return int(stem.removeprefix("episode_"))


def get_shape_dim(feature: dict[str, Any] | None, fallback: int) -> int:
    if not feature:
        return fallback
    shape = feature.get("shape") or []
    if not shape:
        return fallback
    return int(shape[0])


def make_slices(names: list[str], dim: int) -> dict[str, dict[str, int]]:
    if dim <= 0:
        return {}
    if dim <= len(names):
        selected = names[:dim]
        return {name: {"start": idx, "end": idx + 1} for idx, name in enumerate(selected)}

    slices = {name: {"start": idx, "end": idx + 1} for idx, name in enumerate(names[:-1])}
    slices[names[-1]] = {"start": len(names) - 1, "end": dim}
    return slices


def feature_with_key(features: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        if key in features:
            return features[key]
    return None


def short_video_key(feature_key: str) -> str:
    prefix = "observation.images."
    if feature_key.startswith(prefix):
        return feature_key[len(prefix) :]
    return feature_key


def build_modality(info: dict[str, Any], video_feature_keys: list[str], add_validity: bool) -> dict[str, Any]:
    features = info.get("features", {})
    state_dim = get_shape_dim(feature_with_key(features, "observation.state", "state"), 8)
    action_dim = get_shape_dim(feature_with_key(features, "action", "actions"), 7)

    annotation = {
        "human.action.task_description": {"original_key": TASK_ANNOTATION_KEY},
    }
    if add_validity:
        annotation["human.validity"] = {"original_key": VALIDITY_ANNOTATION_KEY}

    return {
        "state": make_slices(DEFAULT_STATE_NAMES, state_dim),
        "action": make_slices(DEFAULT_ACTION_NAMES, action_dim),
        "video": {
            short_video_key(key): {"original_key": key}
            for key in video_feature_keys
        },
        "annotation": annotation,
    }


def update_tasks(output_dir: Path, add_validity: bool) -> int | None:
    tasks_path = output_dir / "meta" / "tasks.jsonl"
    tasks = read_jsonl(tasks_path)
    if not add_validity:
        return None

    for task in tasks:
        if task.get("task") == "valid":
            return int(task["task_index"])

    next_index = max((int(task.get("task_index", -1)) for task in tasks), default=-1) + 1
    tasks.append({"task_index": next_index, "task": "valid"})
    tasks.sort(key=lambda item: int(item["task_index"]))
    write_jsonl(tasks_path, tasks)
    return next_index


def normalize_image(value: Any, dataset_dir: Path, parquet_path: Path) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path") is not None:
            image_path = Path(value["path"])
            candidates = [image_path]
            if not image_path.is_absolute():
                candidates = [dataset_dir / image_path, parquet_path.parent / image_path]
            for candidate in candidates:
                if candidate.exists():
                    return np.asarray(iio.imread(candidate), dtype=np.uint8)
            raise FileNotFoundError(f"Could not resolve image path '{image_path}' from {parquet_path}")
        else:
            raise ValueError(f"Unsupported image dictionary keys: {sorted(value.keys())}")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.asarray(iio.imread(io.BytesIO(bytes(value))), dtype=np.uint8)

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def export_video(
    table: pa.Table,
    source_key: str,
    video_key: str,
    output_dir: Path,
    parquet_path: Path,
    episode_index: int,
    chunks_size: int,
    fps: int,
) -> None:
    if source_key not in table.column_names:
        return
    chunk_index = episode_index // chunks_size
    video_path = output_dir / VIDEO_PATH_TEMPLATE.format(
        episode_chunk=chunk_index,
        video_key=video_key,
        episode_index=episode_index,
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [
        normalize_image(value, output_dir, parquet_path)
        for value in table[source_key].to_pylist()
    ]
    if not frames:
        raise ValueError(f"No frames found for {source_key} in {parquet_path}")
    iio.imwrite(video_path, np.stack(frames, axis=0), fps=fps, codec="libx264", macro_block_size=1)


def make_annotation_array(table: pa.Table, fallback_index: int = 0) -> pa.Array:
    if "task_index" in table.column_names:
        return table["task_index"].combine_chunks()
    return pa.array([fallback_index] * table.num_rows, type=pa.int64())


def constant_int_array(length: int, value: int) -> pa.Array:
    return pa.array([value] * length, type=pa.int64())


def transformed_table(
    table: pa.Table,
    keep_image_columns: bool,
    add_validity: bool,
    validity_task_index: int | None,
) -> pa.Table:
    arrays = []
    names = []
    for name in table.column_names:
        if name in IMAGE_KEY_MAP and not keep_image_columns:
            continue
        new_name = LOW_DIM_KEY_MAP.get(name, IMAGE_KEY_MAP.get(name, name))
        if new_name in names:
            continue
        arrays.append(table[name])
        names.append(new_name)

    if TASK_ANNOTATION_KEY not in names:
        arrays.append(make_annotation_array(table))
        names.append(TASK_ANNOTATION_KEY)
    if add_validity and VALIDITY_ANNOTATION_KEY not in names:
        if validity_task_index is None:
            raise ValueError("validity_task_index is required when adding validity annotations")
        arrays.append(constant_int_array(table.num_rows, validity_task_index))
        names.append(VALIDITY_ANNOTATION_KEY)
    return pa.table(arrays, names=names)


def convert_parquets(
    output_dir: Path,
    info: dict[str, Any],
    video_sources: dict[str, str],
    fps: int,
    add_validity: bool,
    validity_task_index: int | None,
    keep_image_columns: bool,
    skip_video_export: bool,
) -> list[Path]:
    parquet_paths = sorted((output_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {output_dir / 'data'}")

    chunks_size = int(info.get("chunks_size", 1000))
    transformed_paths = []
    for parquet_path in parquet_paths:
        episode_index = episode_index_from_path(parquet_path)
        table = pq.read_table(parquet_path)
        if not skip_video_export:
            for source_key, video_key in video_sources.items():
                export_video(
                    table,
                    source_key,
                    video_key,
                    output_dir,
                    parquet_path,
                    episode_index,
                    chunks_size,
                    fps,
                )

        converted = transformed_table(
            table,
            keep_image_columns=keep_image_columns,
            add_validity=add_validity,
            validity_task_index=validity_task_index,
        )
        pq.write_table(converted, parquet_path)
        transformed_paths.append(parquet_path)
    return transformed_paths


def float_feature_keys(info: dict[str, Any]) -> list[str]:
    result = []
    for key, feature in info.get("features", {}).items():
        if "float" in str(feature.get("dtype", "")):
            result.append(key)
    return result


def as_2d_float(values: list[Any]) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    return np.vstack(rows)


def generate_stats(output_dir: Path, info: dict[str, Any], parquet_paths: list[Path]) -> None:
    feature_keys = float_feature_keys(info)
    collected = {key: [] for key in feature_keys}
    for parquet_path in parquet_paths:
        schema_names = set(pq.read_schema(parquet_path).names)
        table = pq.read_table(parquet_path, columns=[key for key in feature_keys if key in schema_names])
        for key in table.column_names:
            collected[key].extend(table[key].to_pylist())

    stats = {}
    for key, values in collected.items():
        if not values:
            continue
        data = as_2d_float(values)
        stats[key] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    dump_json(output_dir / "meta" / "stats.json", stats)


def generate_relative_stats(output_dir: Path) -> None:
    dump_json(output_dir / "meta" / "relative_stats.json", {})


def converted_feature(key: str, feature: dict[str, Any], video_keys: set[str]) -> dict[str, Any]:
    converted = dict(feature)
    if key in video_keys:
        converted["dtype"] = "video"
        converted.setdefault("shape", feature.get("shape", [256, 256, 3]))
        converted.setdefault("names", feature.get("names", ["height", "width", "channel"]))
    return converted


def update_info(
    info: dict[str, Any],
    video_sources: dict[str, str],
    add_validity: bool,
    total_tasks: int,
    fps: int,
) -> dict[str, Any]:
    old_features = info.get("features", {})
    new_features = {}
    video_feature_keys = set(video_sources.values())
    for old_key, feature in old_features.items():
        if old_key in IMAGE_KEY_MAP:
            new_key = IMAGE_KEY_MAP[old_key]
        elif old_key in LOW_DIM_KEY_MAP:
            new_key = LOW_DIM_KEY_MAP[old_key]
        else:
            new_key = old_key
        new_features[new_key] = converted_feature(new_key, feature, video_feature_keys)

    new_features.setdefault(
        TASK_ANNOTATION_KEY,
        {"dtype": "int64", "shape": [1], "names": None},
    )
    if add_validity:
        new_features.setdefault(
            VALIDITY_ANNOTATION_KEY,
            {"dtype": "int64", "shape": [1], "names": None},
        )

    total_episodes = int(info.get("total_episodes", 0))
    chunks_size = int(info.get("chunks_size", 1000))
    updated = dict(info)
    updated["codebase_version"] = "v2.1"
    updated["fps"] = fps
    updated["features"] = new_features
    updated["data_path"] = DATA_PATH_TEMPLATE
    updated["video_path"] = VIDEO_PATH_TEMPLATE if video_sources else None
    updated["total_videos"] = total_episodes * len(video_sources)
    updated["total_tasks"] = total_tasks
    updated["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else info.get("total_chunks", 1)
    return updated


def discover_video_sources(info: dict[str, Any], skip_video_export: bool) -> dict[str, str]:
    if skip_video_export:
        return {}
    features = info.get("features", {})
    sources = {}
    for source_key, target_key in IMAGE_KEY_MAP.items():
        if source_key in features:
            sources[source_key] = target_key
    return sources


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or default_output_dir(input_dir)).resolve()

    prepare_output(input_dir, output_dir, args.overwrite)

    info_path = output_dir / "meta" / "info.json"
    info = load_json(info_path)
    fps = args.fps or int(info.get("fps", 10))
    add_validity = not args.no_validity_annotation
    validity_task_index = update_tasks(output_dir, add_validity)
    tasks = read_jsonl(output_dir / "meta" / "tasks.jsonl")
    video_sources = discover_video_sources(info, args.skip_video_export)

    parquet_paths = convert_parquets(
        output_dir=output_dir,
        info=info,
        video_sources=video_sources,
        fps=fps,
        add_validity=add_validity,
        validity_task_index=validity_task_index,
        keep_image_columns=args.keep_image_columns,
        skip_video_export=args.skip_video_export,
    )

    updated_info = update_info(
        info,
        video_sources=video_sources,
        add_validity=add_validity,
        total_tasks=len(tasks),
        fps=fps,
    )
    dump_json(info_path, updated_info)
    dump_json(
        output_dir / "meta" / "modality.json",
        build_modality(updated_info, list(video_sources.values()), add_validity),
    )
    if not args.skip_stats:
        generate_stats(output_dir, updated_info, parquet_paths)
        generate_relative_stats(output_dir)

    print(f"Converted dataset written to: {output_dir}")
    print(f"Episodes: {updated_info.get('total_episodes', len(parquet_paths))}")
    print(f"Video streams: {len(video_sources)}")
    print("Next: use this output path as the GR00T dataset path.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc