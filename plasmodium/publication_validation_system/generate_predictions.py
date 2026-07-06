"""Generate locked strict and temporal masks for a blinded validation package."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import platform
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy
import skimage

from common import (
    VALID_ROLES,
    load_json,
    package_root,
    validate_algorithm_lock,
    write_binary_mask,
    write_json,
)


def load_analysis_module(source_path: Path):
    module_name = "locked_plasmodium_analysis"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import analysis source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def read_contiguous_frames(video_path: Path, first: int, last: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frames: list[tuple[int, np.ndarray]] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first))
        for frame_idx in range(first, last + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
            frames.append((frame_idx, frame))
    finally:
        cap.release()
    return frames


def validate_entrypoints(module, config: dict) -> None:
    for label, name in config["algorithm"]["implementation"].items():
        if not hasattr(module, name):
            raise AttributeError(f"Missing {label} entry point {name!r} in locked source")


def generate(
    config_path: Path,
    role: str,
    allow_development: bool,
    overwrite: bool,
) -> Path:
    config = load_json(config_path)
    source_path, source_hash, parameter_hash, runner_hash = validate_algorithm_lock(
        config,
        config_path,
        allow_development=allow_development,
    )
    root = package_root(config, config_path, role)
    manifest_path = root / "private_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing package manifest: {manifest_path}. Run prepare_dataset.py first."
        )
    rows = read_manifest(manifest_path)
    if any(row["dataset_role"] != role for row in rows):
        raise ValueError("Manifest contains rows from a different dataset role")

    predictions_root = root / "predictions"
    metadata_path = predictions_root / "prediction_run_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Predictions already exist: {predictions_root}. Use --overwrite only "
            "to reproduce the same locked run."
        )
    if predictions_root.exists() and overwrite:
        shutil.rmtree(predictions_root)
    strict_dir = predictions_root / "strict"
    temporal_dir = predictions_root / "temporal"
    candidate_dir = predictions_root / "debug_temporal_candidate"
    added_dir = predictions_root / "debug_temporal_added"
    for directory in (strict_dir, temporal_dir, candidate_dir, added_dir):
        directory.mkdir(parents=True, exist_ok=True)

    analysis = load_analysis_module(source_path)
    validate_entrypoints(analysis, config)
    algorithm = config["algorithm"]
    hsv_params = dict(algorithm["body_hsv_params"])
    temporal = dict(algorithm["temporal_params"])
    strict_fn = getattr(analysis, algorithm["implementation"]["strict_function"])
    candidate_fn = getattr(analysis, algorithm["implementation"]["candidate_function"])
    recovery_fn = getattr(analysis, algorithm["implementation"]["temporal_function"])

    temporal_half = int(temporal["window"]) // 2
    smooth_half = int(temporal.get("final_smooth_window", 1)) // 2
    halo = temporal_half + smooth_half
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["video_id"], row["window_id"])].append(row)

    output_rows: list[dict[str, object]] = []
    for (_, window_id), group in sorted(grouped.items()):
        group.sort(key=lambda value: int(value["source_frame_idx"]))
        target_indices = [int(value["source_frame_idx"]) for value in group]
        if target_indices != list(range(target_indices[0], target_indices[-1] + 1)):
            raise ValueError(f"Window is not consecutive: {window_id}")
        video_path = Path(group[0]["video_path"])
        total_frames = int(group[0]["total_frames"])
        first = target_indices[0] - halo
        last = target_indices[-1] + halo
        if first < 0 or last >= total_frames:
            raise ValueError(
                f"Window {window_id} lacks the {halo}-frame temporal halo. "
                "Increase edge_exclusion_frames when preparing the dataset."
            )
        center = (
            int(group[0]["dish_center_x"]),
            int(group[0]["dish_center_y"]),
        )
        radius = int(group[0]["dish_radius"])
        items = []
        for frame_idx, frame in read_contiguous_frames(video_path, first, last):
            _, strict_mask, _ = strict_fn(
                frame,
                dish_center=center,
                dish_radius=radius,
                **hsv_params,
            )
            candidate_mask = candidate_fn(
                frame,
                dish_center=center,
                dish_radius=radius,
                chroma_min=int(temporal["chroma_min"]),
                min_channel=int(temporal["min_channel"]),
                max_value=int(temporal["max_value"]),
                rim_exclusion_px=int(temporal["rim_exclusion_px"]),
                min_obj_size=int(temporal["min_obj_size"]),
            )
            items.append(
                {
                    "frame_idx": frame_idx,
                    "frame": frame,
                    "mask": strict_mask.astype(bool),
                    "candidate": candidate_mask.astype(bool),
                }
            )
        recovered = recovery_fn(
            items,
            window=int(temporal["window"]),
            min_hits=int(temporal["min_hits"]),
            connect_radius=int(temporal["connect_radius"]),
            exterior_only=bool(temporal["exterior_only"]),
            block_hole_max_area=int(temporal["block_hole_max_area"]),
            final_smooth_window=int(temporal["final_smooth_window"]),
            final_smooth_min_hits=int(temporal["final_smooth_min_hits"]),
        )
        recovered_by_frame = {int(item["frame_idx"]): item for item in recovered}
        original_by_frame = {int(item["frame_idx"]): item for item in items}
        for row in group:
            frame_idx = int(row["source_frame_idx"])
            bid = row["blind_id"]
            strict_mask = original_by_frame[frame_idx]["mask"].astype(bool)
            candidate_mask = original_by_frame[frame_idx]["candidate"].astype(bool)
            temporal_mask = recovered_by_frame[frame_idx]["mask"].astype(bool)
            added_mask = temporal_mask & (~strict_mask)
            write_binary_mask(strict_dir / f"{bid}.png", strict_mask)
            write_binary_mask(temporal_dir / f"{bid}.png", temporal_mask)
            write_binary_mask(candidate_dir / f"{bid}.png", candidate_mask)
            write_binary_mask(added_dir / f"{bid}.png", added_mask)
            output_rows.append(
                {
                    "blind_id": bid,
                    "strict_area_px": int(strict_mask.sum()),
                    "temporal_area_px": int(temporal_mask.sum()),
                    "temporal_added_area_px": int(added_mask.sum()),
                    "candidate_area_px": int(candidate_mask.sum()),
                }
            )

    with (predictions_root / "prediction_areas.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = list(output_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda value: str(value["blind_id"])))

    lock_valid = (
        algorithm.get("status") == "frozen"
        and source_hash == algorithm.get("locked_source_sha256")
        and runner_hash == algorithm.get("locked_prediction_runner_sha256")
        and parameter_hash == algorithm.get("locked_parameters_sha256")
    )
    publication_eligible = role == "test" and lock_valid and not allow_development
    write_json(
        metadata_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_role": role,
            "publication_eligible": publication_eligible,
            "development_override_used": bool(allow_development),
            "algorithm_status": algorithm.get("status"),
            "source_file": str(source_path),
            "source_sha256": source_hash,
            "prediction_runner_sha256": runner_hash,
            "parameters_sha256": parameter_hash,
            "temporal_halo_frames": halo,
            "frame_count": len(output_rows),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "scipy": scipy.__version__,
                "scikit_image": skimage.__version__,
            },
            "body_hsv_params": hsv_params,
            "temporal_params": temporal,
        },
    )
    return predictions_root


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), required=True)
    parser.add_argument(
        "--allow-development",
        action="store_true",
        help="Bypass the lock only for software/development checks; output is ineligible.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = generate(
        args.config.resolve(),
        args.role,
        args.allow_development,
        args.overwrite,
    )
    print(f"Saved strict and temporal predictions: {predictions}")
    print("Keep predictions hidden until all annotations and adjudication are complete.")


if __name__ == "__main__":
    main()
