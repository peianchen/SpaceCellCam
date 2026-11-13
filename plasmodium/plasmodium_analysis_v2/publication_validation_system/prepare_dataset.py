"""Create a deterministic, blinded annotation package from independent videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import (
    VALID_ROLES,
    load_json,
    package_root,
    sha256_file,
    validate_algorithm_lock,
    video_metadata,
    write_binary_mask,
    write_json,
)


REQUIRED_VIDEO_COLUMNS = {
    "video_id",
    "video_path",
    "dataset_role",
    "biological_replicate",
    "condition",
    "acquisition_date",
    "dish_center_x",
    "dish_center_y",
    "dish_radius",
    "pixel_size",
    "frame_interval_s",
}


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def blind_id(base_seed: int, role: str, video_id: str, frame_idx: int) -> str:
    payload = f"{base_seed}|{role}|{video_id}|{frame_idx}|primary".encode("utf-8")
    return "B" + hashlib.sha256(payload).hexdigest()[:15].upper()


def repeat_blind_id(base_seed: int, primary_id: str) -> str:
    payload = f"{base_seed}|{primary_id}|repeat".encode("utf-8")
    return "R" + hashlib.sha256(payload).hexdigest()[:15].upper()


def read_video_table(path: Path, role: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_VIDEO_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing video table columns: {sorted(missing)}")
        all_rows = [dict(row) for row in reader]
    invalid_roles = sorted(
        {
            row["dataset_role"].strip()
            for row in all_rows
            if row["dataset_role"].strip() not in VALID_ROLES
        }
    )
    if invalid_roles:
        raise ValueError(f"Invalid dataset roles in video registry: {invalid_roles}")
    ids = [row["video_id"].strip() for row in all_rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("video_id values must be non-empty and unique in the registry")
    rows = [row for row in all_rows if row["dataset_role"].strip() == role]
    if not rows:
        raise ValueError(f"No rows with dataset_role={role!r} in {path}")

    for target in rows:
        target_path = str(Path(target["video_path"].strip()).expanduser().resolve()).casefold()
        target_replicate = target["biological_replicate"].strip()
        for other in all_rows:
            if other is target or other["dataset_role"].strip() == role:
                continue
            other_path = str(Path(other["video_path"].strip()).expanduser().resolve()).casefold()
            if target_path == other_path:
                raise ValueError(
                    f"Video {target['video_id']!r} is reused across dataset roles. "
                    "Development, validation, and test videos must be disjoint."
                )
            if target_replicate and target_replicate == other["biological_replicate"].strip():
                raise ValueError(
                    f"Biological replicate {target_replicate!r} is reused across roles."
                )
    return rows


def phase_bounds(first: int, last: int, phase_count: int) -> list[tuple[int, int]]:
    edges = np.linspace(first, last + 1, phase_count + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1] - 1)) for i in range(phase_count)]


def choose_window_starts(
    phase_first: int,
    phase_last: int,
    frames_per_window: int,
    windows_per_phase: int,
    minimum_gap: int,
    rng: np.random.Generator,
) -> list[int]:
    max_start = phase_last - frames_per_window + 1
    starts = list(range(phase_first, max_start + 1))
    if len(starts) < windows_per_phase:
        raise ValueError(
            f"Phase {phase_first}:{phase_last} is too short for "
            f"{windows_per_phase} windows of {frames_per_window} frames"
        )

    required_separation = frames_per_window + max(0, minimum_gap)
    randomized_starts = [int(value) for value in rng.permutation(starts)]

    def search(selected: list[int], candidates: list[int]) -> list[int] | None:
        if len(selected) == windows_per_phase:
            return sorted(selected)
        for start in candidates:
            if all(abs(start - chosen) >= required_separation for chosen in selected):
                remaining = [value for value in candidates if value != start]
                result = search([*selected, start], remaining)
                if result is not None:
                    return result
        return None

    selected = search([], randomized_starts)
    if selected is None:
        raise ValueError(
            "Could not place non-overlapping windows. Reduce the minimum gap "
            "or use a longer video."
        )
    return selected


def extract_frames(video_path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frames: dict[int, np.ndarray] = {}
    try:
        for frame_idx in sorted(set(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
            frames[frame_idx] = frame
    finally:
        cap.release()
    return frames


def parse_float(row: dict[str, str], key: str, positive: bool = False) -> float:
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key!r} for video {row.get('video_id')}") from exc
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid {key!r} for video {row.get('video_id')}: {value}")
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_package(config_path: Path, videos_path: Path, role: str, force: bool) -> Path:
    config = load_json(config_path)
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
    if role == "test":
        validate_algorithm_lock(config, config_path, allow_development=False)

    rows = read_video_table(videos_path, role)
    root = package_root(config, config_path, role)
    private_manifest_path = root / "private_manifest.csv"
    if private_manifest_path.exists() and not force:
        raise FileExistsError(
            f"Package already exists: {root}. Use --force only before annotation begins."
        )
    if root.exists() and force:
        shutil.rmtree(root)

    images_dir = root / "annotation_blinded" / "images"
    templates_dir = root / "annotation_blinded" / "mask_templates"
    repeat_images_dir = root / "annotation_blinded" / "repeat_images"
    for directory in (images_dir, templates_dir, repeat_images_dir):
        directory.mkdir(parents=True, exist_ok=True)
    annotators = list(config["annotation"]["annotators"])
    repeat_annotators = list(config["annotation"].get("repeat_annotators", []))
    for annotator in annotators:
        (root / "annotators" / annotator / "masks").mkdir(parents=True, exist_ok=True)
    for annotator in repeat_annotators:
        (root / "annotators" / f"{annotator}_repeat" / "masks").mkdir(
            parents=True, exist_ok=True
        )
    (root / "adjudicated_masks").mkdir(parents=True, exist_ok=True)

    sampling = config["sampling"]
    phases = list(sampling["phases"])
    frames_per_window = int(sampling["frames_per_window"])
    windows_per_phase = int(sampling["windows_per_phase"])
    edge = int(sampling["edge_exclusion_frames"])
    gap = int(sampling.get("minimum_gap_between_windows", 0))
    base_seed = int(config["random_seed"])

    manifest: list[dict[str, object]] = []
    public_index: list[dict[str, object]] = []
    video_summaries: list[dict[str, object]] = []
    seen_replicates: set[str] = set()

    for row in rows:
        video_id = row["video_id"].strip()
        replicate = row["biological_replicate"].strip()
        if not replicate:
            raise ValueError(f"biological_replicate is required for {video_id}")
        if replicate in seen_replicates:
            raise ValueError(
                f"biological_replicate {replicate!r} appears more than once in role {role}. "
                "Use one video row per independent replicate or assign distinct replicate IDs."
            )
        seen_replicates.add(replicate)

        video_path = Path(row["video_path"].strip()).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")
        metadata = video_metadata(video_path)
        total_frames = int(metadata["frame_count"])
        valid_first = edge
        valid_last = total_frames - edge - 1
        if valid_last <= valid_first:
            raise ValueError(f"Video is too short after edge exclusion: {video_path}")

        center_x = int(round(parse_float(row, "dish_center_x")))
        center_y = int(round(parse_float(row, "dish_center_y")))
        dish_radius = int(round(parse_float(row, "dish_radius", positive=True)))
        pixel_size = parse_float(row, "pixel_size", positive=True)
        frame_interval = parse_float(row, "frame_interval_s", positive=True)
        bounds = phase_bounds(valid_first, valid_last, len(phases))
        selected_records: list[dict[str, object]] = []
        for phase, (phase_first, phase_last) in zip(phases, bounds):
            rng = np.random.default_rng(stable_seed(base_seed, role, video_id, phase))
            starts = choose_window_starts(
                phase_first,
                phase_last,
                frames_per_window,
                windows_per_phase,
                gap,
                rng,
            )
            for window_number, start in enumerate(starts, start=1):
                window_id = f"{video_id}_{phase}_w{window_number}"
                for position, frame_idx in enumerate(
                    range(start, start + frames_per_window), start=1
                ):
                    selected_records.append(
                        {
                            "phase": phase,
                            "window_id": window_id,
                            "window_position": position,
                            "source_frame_idx": frame_idx,
                        }
                    )

        frame_map = extract_frames(
            video_path,
            [int(record["source_frame_idx"]) for record in selected_records],
        )
        for record in selected_records:
            frame_idx = int(record["source_frame_idx"])
            frame = frame_map[frame_idx]
            bid = blind_id(base_seed, role, video_id, frame_idx)
            image_name = f"{bid}.png"
            image_path = images_dir / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise OSError(f"Could not write image: {image_path}")
            write_binary_mask(templates_dir / image_name, np.zeros(frame.shape[:2], dtype=bool))

            private_row = {
                "blind_id": bid,
                "video_id": video_id,
                "video_path": str(video_path),
                "dataset_role": role,
                "biological_replicate": replicate,
                "condition": row["condition"].strip(),
                "acquisition_date": row["acquisition_date"].strip(),
                "source_frame_idx": frame_idx,
                "phase": record["phase"],
                "window_id": record["window_id"],
                "window_position": record["window_position"],
                "total_frames": total_frames,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "image_sha256": sha256_file(image_path),
                "dish_center_x": center_x,
                "dish_center_y": center_y,
                "dish_radius": dish_radius,
                "pixel_size": pixel_size,
                "frame_interval_s": frame_interval,
            }
            manifest.append(private_row)
            public_index.append(
                {
                    "blind_id": bid,
                    "image_file": image_name,
                    "image_width": int(frame.shape[1]),
                    "image_height": int(frame.shape[0]),
                }
            )

        video_summaries.append(
            {
                "video_id": video_id,
                "biological_replicate": replicate,
                "sampled_frames": len(selected_records),
                "total_frames": total_frames,
            }
        )

    manifest.sort(key=lambda value: (str(value["video_id"]), int(value["source_frame_idx"])))
    public_index.sort(key=lambda value: str(value["blind_id"]))
    manifest_fields = list(manifest[0].keys())
    write_csv(private_manifest_path, manifest, manifest_fields)
    write_csv(
        root / "annotation_blinded" / "annotation_index.csv",
        public_index,
        ["blind_id", "image_file", "image_width", "image_height"],
    )

    repeat_fraction = float(sampling.get("repeat_annotation_fraction", 0.0))
    repeat_count = int(math.ceil(len(manifest) * repeat_fraction))
    repeat_count = min(max(repeat_count, 0), len(manifest))
    repeat_manifest: list[dict[str, object]] = []
    if repeat_count:
        rng = np.random.default_rng(stable_seed(base_seed, role, "repeat_selection"))
        selected_indices = sorted(rng.choice(len(manifest), size=repeat_count, replace=False))
        for index in selected_indices:
            source = manifest[int(index)]
            primary_id = str(source["blind_id"])
            repeat_id = repeat_blind_id(base_seed, primary_id)
            source_image = images_dir / f"{primary_id}.png"
            repeat_image = repeat_images_dir / f"{repeat_id}.png"
            shutil.copy2(source_image, repeat_image)
            repeat_manifest.append(
                {
                    "repeat_blind_id": repeat_id,
                    "primary_blind_id": primary_id,
                    "video_id": source["video_id"],
                    "source_frame_idx": source["source_frame_idx"],
                }
            )
    write_csv(
        root / "private_repeat_manifest.csv",
        repeat_manifest,
        ["repeat_blind_id", "primary_blind_id", "video_id", "source_frame_idx"],
    )

    write_json(
        root / "package_metadata.json",
        {
            "schema_version": config["schema_version"],
            "random_seed": base_seed,
            "dataset_role": role,
            "publication_eligible_role": role == "test",
            "video_count": len(rows),
            "biological_replicate_count": len(seen_replicates),
            "sampled_frame_count": len(manifest),
            "repeat_frame_count": repeat_count,
            "sampling": sampling,
            "videos": video_summaries,
            "warning": (
                "Keep private_manifest.csv and predictions hidden from annotators "
                "until all primary and repeat annotations are complete."
            ),
        },
    )
    return root


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(VALID_ROLES), required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing package. Use only before annotation begins.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = build_package(
        args.config.resolve(),
        args.videos.resolve(),
        args.role,
        args.force,
    )
    print(f"Created blinded {args.role} package: {root}")
    print("Do not give private_manifest.csv or prediction outputs to annotators.")


if __name__ == "__main__":
    main()

