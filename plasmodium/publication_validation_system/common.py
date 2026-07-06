"""Shared utilities for the publication validation workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_ROLES = {"development", "validation", "test"}


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def resolve_from(base_file: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_file).resolve().parent / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def algorithm_parameter_payload(config: dict[str, Any]) -> dict[str, Any]:
    algorithm = config["algorithm"]
    return {
        "implementation": algorithm["implementation"],
        "body_hsv_params": algorithm["body_hsv_params"],
        "temporal_params": algorithm["temporal_params"],
    }


def validate_algorithm_lock(
    config: dict[str, Any],
    config_path: Path,
    allow_development: bool = False,
) -> tuple[Path, str, str, str]:
    algorithm = config["algorithm"]
    source_path = resolve_from(config_path, algorithm["source_file"])
    runner_path = resolve_from(config_path, algorithm["prediction_runner_file"])
    if not source_path.exists():
        raise FileNotFoundError(f"Algorithm source does not exist: {source_path}")
    if not runner_path.exists():
        raise FileNotFoundError(f"Prediction runner does not exist: {runner_path}")

    source_hash = sha256_file(source_path)
    runner_hash = sha256_file(runner_path)
    parameter_hash = canonical_sha256(algorithm_parameter_payload(config))
    is_frozen = algorithm.get("status") == "frozen"
    lock_matches = (
        source_hash == algorithm.get("locked_source_sha256")
        and runner_hash == algorithm.get("locked_prediction_runner_sha256")
        and parameter_hash == algorithm.get("locked_parameters_sha256")
    )
    if not allow_development and (not is_frozen or not lock_matches):
        raise RuntimeError(
            "Publication prediction is blocked because the algorithm is not frozen "
            "or its source/parameters changed. Run freeze_algorithm.py only after "
            "development is complete."
        )
    return source_path, source_hash, parameter_hash, runner_hash


def package_root(config: dict[str, Any], config_path: Path, role: str) -> Path:
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown dataset role: {role}")
    output_root = resolve_from(config_path, config["output_root"])
    return output_root / role


def read_binary_mask(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read mask: {path}")
    mask = image > 127
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape mismatch for {path}: got {mask.shape}, expected {expected_shape}"
        )
    return mask


def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    if not ok:
        raise OSError(f"Could not write mask: {path}")


def video_metadata(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        playback_fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")
    return {
        "frame_count": frame_count,
        "playback_fps": playback_fps,
        "width": width,
        "height": height,
    }


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return frame

