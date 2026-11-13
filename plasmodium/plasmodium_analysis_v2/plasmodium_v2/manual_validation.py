"""Manual-mask export and scoring for the same temporal mask used formally."""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .segmentation import segment_hsv_circle
from .temporal import (
    apply_temporal_body_recovery_to_sequence,
    required_temporal_halo,
    temporal_pale_branch_candidate,
)


def _read_frame(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise ValueError(f"Could not read video frame {index}")
    return frame


def _temporal_mask_at(
    cap,
    target_index,
    total_frames,
    dish_center,
    dish_radius,
    hsv_params,
    temporal,
):
    halo = required_temporal_halo(
        temporal["window"], temporal.get("final_smooth_window", 1)
    )
    first = target_index - halo
    last = target_index + halo
    if first < 0 or last >= total_frames:
        raise ValueError(f"Frame {target_index} lacks the required {halo}-frame halo")
    items = []
    for frame_index in range(first, last + 1):
        frame = _read_frame(cap, frame_index)
        _, strict_mask, _ = segment_hsv_circle(
            frame, dish_center=dish_center, dish_radius=dish_radius, **hsv_params
        )
        candidate = temporal_pale_branch_candidate(
            frame,
            dish_center,
            dish_radius,
            chroma_min=temporal["chroma_min"],
            min_channel=temporal["min_channel"],
            max_value=temporal["max_value"],
            rim_exclusion_px=temporal["rim_exclusion_px"],
            min_obj_size=temporal["min_obj_size"],
        )
        items.append(
            {
                "frame_idx": frame_index,
                "frame": frame,
                "mask": strict_mask,
                "candidate": candidate,
                "dish_center": dish_center,
                "dish_radius": dish_radius,
            }
        )
    recovered = apply_temporal_body_recovery_to_sequence(
        items,
        window=temporal["window"],
        min_hits=temporal["min_hits"],
        connect_radius=temporal["connect_radius"],
        exterior_only=temporal["exterior_only"],
        block_hole_max_area=temporal["block_hole_max_area"],
        final_smooth_window=temporal.get("final_smooth_window", 1),
        final_smooth_min_hits=temporal.get("final_smooth_min_hits", 1),
    )
    selected = next(item for item in recovered if int(item["frame_idx"]) == int(target_index))
    return selected["frame"], selected["mask"].astype(bool), selected["strict_mask"].astype(bool)


def export_manual_validation_package(
    video_path,
    out_dir,
    dish_center,
    dish_radius,
    hsv_params,
    samples_per_section=10,
    section_frames=300,
    *,
    edge_skip=10,
    use_temporal_recovery=True,
    temporal_kwargs=None,
):
    """Export non-edge samples; auto masks match the formal temporal method."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temporal = dict(temporal_kwargs or {})
    halo = required_temporal_halo(
        temporal.get("window", 5), temporal.get("final_smooth_window", 1)
    ) if use_temporal_recovery else 0
    safe_edge = max(int(edge_skip), halo)
    usable_start = safe_edge
    usable_end = total_frames - safe_edge - 1
    if usable_end < usable_start:
        cap.release()
        raise ValueError("Video is too short after temporal edge exclusion")
    usable_count = usable_end - usable_start + 1
    section_frames = max(1, min(int(section_frames), usable_count))
    starts = {
        "front": usable_start,
        "middle": usable_start + max(0, (usable_count - section_frames) // 2),
        "rear": usable_end - section_frames + 1,
    }

    package_dir = Path(out_dir) / "manual_validation_package"
    source_dir = package_dir / "source_images"
    auto_dir = package_dir / "auto_masks"
    strict_dir = package_dir / "spatial_masks"
    overlay_dir = package_dir / "overlays"
    manual_dir = package_dir / "manual_masks"
    for directory in (source_dir, auto_dir, strict_dir, overlay_dir, manual_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_frames = set()
    try:
        for section, start in starts.items():
            end = min(usable_end, start + section_frames - 1)
            sample_indices = np.linspace(
                start, end, min(int(samples_per_section), end - start + 1), dtype=int
            )
            for frame_index in sample_indices:
                frame_index = int(frame_index)
                if frame_index in seen_frames:
                    continue
                seen_frames.add(frame_index)
                if use_temporal_recovery:
                    frame, auto_mask, strict_mask = _temporal_mask_at(
                        cap, frame_index, total_frames, dish_center, dish_radius,
                        hsv_params, temporal,
                    )
                    method = "temporal_recovered_spatial_mask"
                else:
                    frame = _read_frame(cap, frame_index)
                    _, auto_mask, _ = segment_hsv_circle(
                        frame, dish_center=dish_center, dish_radius=dish_radius, **hsv_params
                    )
                    strict_mask = auto_mask.copy()
                    method = "spatial_non_temporal_mask"

                overlay = frame.astype(np.float32)
                overlay[auto_mask] = 0.55 * overlay[auto_mask] + 0.45 * np.array([0, 255, 0])
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    auto_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
                stem = f"frame_{frame_index:06d}"
                source_path = source_dir / f"{stem}_original.png"
                auto_path = auto_dir / f"{stem}_mask.png"
                strict_path = strict_dir / f"{stem}_mask.png"
                overlay_path = overlay_dir / f"{stem}_overlay.jpg"
                manual_path = manual_dir / f"{stem}_mask.png"
                if not cv2.imwrite(str(source_path), frame):
                    raise OSError(f"Could not write {source_path}")
                cv2.imwrite(str(auto_path), auto_mask.astype(np.uint8) * 255)
                cv2.imwrite(str(strict_path), strict_mask.astype(np.uint8) * 255)
                cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(
                    {
                        "section": section,
                        "frame": frame_index,
                        "video_total_frames": total_frames,
                        "auto_mask_method": method,
                        "source_image": str(source_path.resolve()),
                        "auto_mask": str(auto_path.resolve()),
                        "spatial_mask": str(strict_path.resolve()),
                        "overlay": str(overlay_path.resolve()),
                        "expected_manual_mask": str(manual_path.resolve()),
                        "manual_mask_ready": manual_path.exists(),
                    }
                )
    finally:
        cap.release()

    manifest = pd.DataFrame(rows).sort_values(["section", "frame"])
    manifest.to_csv(
        package_dir / "manual_validation_manifest.csv", index=False, encoding="utf-8-sig"
    )
    return manifest


def validate_manual_validation_package(package_dir):
    """Score the exact auto mask recorded in the manifest."""
    package_dir = Path(package_dir)
    manifest_path = package_dir / "manual_validation_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    rows = []
    for item in manifest.itertuples(index=False):
        manual_path = Path(item.expected_manual_mask)
        if not manual_path.exists():
            continue
        auto = cv2.imread(str(item.auto_mask), cv2.IMREAD_GRAYSCALE)
        manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
        if auto is None or manual is None:
            warnings.warn(f"Could not read mask pair for frame {item.frame}")
            continue
        if manual.shape != auto.shape:
            manual = cv2.resize(manual, auto.shape[::-1], interpolation=cv2.INTER_NEAREST)
        auto, manual = auto > 0, manual > 0
        intersection = int((auto & manual).sum())
        union = int((auto | manual).sum())
        auto_area, manual_area = int(auto.sum()), int(manual.sum())
        signed_error = (auto_area - manual_area) / manual_area if manual_area else np.nan
        rows.append(
            {
                "section": item.section, "frame": int(item.frame),
                "auto_mask_method": getattr(item, "auto_mask_method", "unknown"),
                "auto_area_px": auto_area, "manual_area_px": manual_area,
                "iou": intersection / union if union else np.nan,
                "dice": 2 * intersection / (auto_area + manual_area) if auto_area + manual_area else np.nan,
                "precision": intersection / auto_area if auto_area else np.nan,
                "recall": intersection / manual_area if manual_area else np.nan,
                "signed_area_error": signed_error,
                "absolute_area_error": abs(signed_error),
            }
        )
    details = pd.DataFrame(rows)
    details.to_csv(package_dir / "manual_validation_metrics.csv", index=False, encoding="utf-8-sig")
    if details.empty:
        return details, pd.DataFrame()
    summary = details.groupby(["section", "auto_mask_method"], as_index=False).agg(
        n_masks=("frame", "count"), iou_mean=("iou", "mean"), dice_mean=("dice", "mean"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        absolute_area_error_mean=("absolute_area_error", "mean"),
    )
    summary.to_csv(package_dir / "manual_validation_summary.csv", index=False, encoding="utf-8-sig")
    return details, summary
