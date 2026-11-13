"""Prepare and validate a leakage-safe representative-frame dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .registration import RegistrationResult, estimate_registration, warp_to_reference


MANIFEST_FIELDS = [
    "sample_id", "split", "group", "video_id", "source_video", "section",
    "frame_index", "total_frames", "fps", "image_path", "verified_mask_path",
    "registration_method", "registration_applied", "registration_score",
    "registration_translation_px", "registration_rotation_deg",
    "m00", "m01", "m02", "m10", "m11", "m12",
]


def _safe_id(group, index, path):
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{group}_{index:02d}_{digest}"


def _split_for_group_index(index, count):
    if count < 3:
        raise ValueError("Each condition needs at least three videos for leakage-safe splits")
    if index == count - 1:
        return "test"
    if index == count - 2:
        return "val"
    return "train"


def _indices_for_section(total, bounds, count):
    first = int(round((total - 1) * float(bounds[0])))
    last = int(round((total - 1) * float(bounds[1])))
    return np.linspace(first, last, int(count), dtype=int).tolist()


def _read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {index}")
    return frame


def _preview(images, labels, width=420, columns=5):
    thumbs = []
    for image, label in zip(images, labels):
        thumb = cv2.resize(image, (width, int(round(image.shape[0] * width / image.shape[1]))))
        cv2.putText(
            thumb, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        thumbs.append(thumb)
    blank = np.zeros_like(thumbs[0])
    rows = []
    for start in range(0, len(thumbs), columns):
        row = thumbs[start : start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def prepare_dataset(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent
    dataset_root = (project_root / config["dataset_root"]).resolve()
    images_dir = dataset_root / "images"
    verified_dir = dataset_root / "masks_verified"
    draft_dir = dataset_root / "masks_draft"
    previews_dir = dataset_root / "previews"
    for folder in (images_dir, verified_dir, draft_dir, previews_dir):
        folder.mkdir(parents=True, exist_ok=True)

    center = tuple(config["dish_center"])
    radius = int(config["dish_radius"])
    registration = dict(config.get("registration", {}))
    registration.pop("enabled", None)
    rows = []
    for group, folder_value in config["video_groups"].items():
        paths = sorted(Path(folder_value).glob("*.mp4"))
        for video_index, path in enumerate(paths):
            split = _split_for_group_index(video_index, len(paths))
            video_id = _safe_id(group, video_index, path)
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {path}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            all_targets = []
            for section, bounds in config["sections"].items():
                for index in _indices_for_section(total, bounds, config["frames_per_section"]):
                    all_targets.append((section, index))
            reference = _read_frame(cap, all_targets[0][1])
            preview_images = []
            preview_labels = []
            for section, frame_index in all_targets:
                frame = _read_frame(cap, frame_index)
                result = estimate_registration(reference, frame, center, radius, **registration)
                stabilized = warp_to_reference(frame, result)
                sample_id = f"{video_id}_{section}_f{frame_index:06d}"
                image_path = images_dir / f"{sample_id}.png"
                if not cv2.imwrite(str(image_path), stabilized, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"Could not write {image_path}")
                matrix = result.matrix.ravel().tolist()
                rows.append({
                    "sample_id": sample_id, "split": split, "group": group,
                    "video_id": video_id, "source_video": str(path), "section": section,
                    "frame_index": frame_index, "total_frames": total, "fps": fps,
                    "image_path": str(image_path.relative_to(dataset_root)),
                    "verified_mask_path": str((verified_dir / f"{sample_id}.png").relative_to(dataset_root)),
                    "registration_method": result.method,
                    "registration_applied": result.applied,
                    "registration_score": result.score,
                    "registration_translation_px": result.translation_px,
                    "registration_rotation_deg": result.rotation_deg,
                    "m00": matrix[0], "m01": matrix[1], "m02": matrix[2],
                    "m10": matrix[3], "m11": matrix[4], "m12": matrix[5],
                })
                preview_images.append(stabilized)
                preview_labels.append(f"{section} f={frame_index} {result.method}")
            cap.release()
            sheet = _preview(preview_images, preview_labels)
            cv2.imwrite(str(previews_dir / f"{video_id}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Prepared {video_id}: {len(all_targets)} frames -> {split}", flush=True)

    manifest = dataset_root / "dataset_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sample_count": len(rows),
        "video_count": len({row["video_id"] for row in rows}),
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "verified_mask_count": 0,
    }
    (dataset_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dataset_root


def validate_dataset(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = (config_path.parent / config["dataset_root"]).resolve()
    manifest = dataset_root / "dataset_manifest.csv"
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Dataset manifest is empty")
    video_splits = {}
    verified_count = 0
    for row in rows:
        video_splits.setdefault(row["video_id"], set()).add(row["split"])
        image_path = dataset_root / row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        mask_path = dataset_root / row["verified_mask_path"]
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f"Mask shape mismatch: {mask_path}")
            values = set(np.unique(mask).tolist())
            if not values.issubset({0, 255}):
                raise ValueError(f"Verified mask must be binary 0/255: {mask_path}")
            verified_count += 1
    leaked = {video: splits for video, splits in video_splits.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"Video-level split leakage: {leaked}")
    return {
        "samples": len(rows), "videos": len(video_splits),
        "verified_masks": verified_count,
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
    }

