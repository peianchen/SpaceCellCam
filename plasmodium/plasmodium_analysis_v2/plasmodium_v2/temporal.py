"""Temporal pale-branch recovery with strict-mask preservation."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

from .mask_utils import (
    as_gray_uint8,
    circular_roi_mask,
    keep_components_touching_mask,
    keep_largest_component,
)


def _motion_feature(frame, dish_center, dish_radius, max_dimension=640):
    """Build a lighting-resistant dish-rim feature for translation matching."""
    frame = np.asarray(frame)
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    if scale < 1.0:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = frame
    gray = as_gray_uint8(small)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    feature = cv2.magnitude(grad_x, grad_y)

    yy, xx = np.ogrid[: small.shape[0], : small.shape[1]]
    cx = float(dish_center[0]) * scale
    cy = float(dish_center[1]) * scale
    radius = float(dish_radius) * scale
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # The organism is normally inside this annulus.  Dish rims and fixed marks
    # remain useful even when the organism changes substantially over time.
    annulus = ((distance >= 0.70 * radius) & (distance <= 1.18 * radius)).astype(np.float32)
    window = cv2.createHanningWindow(
        (small.shape[1], small.shape[0]), cv2.CV_32F
    ) * annulus
    if np.any(annulus):
        feature = (feature - np.median(feature[annulus > 0])) * annulus
    return feature.astype(np.float32), window.astype(np.float32), scale


def _estimate_motion_to_anchor(
    items,
    *,
    max_shift_px=30.0,
    min_response=0.35,
    max_dimension=640,
):
    """Estimate per-frame translations that align each dish rim to one anchor."""
    count = len(items)
    identity = [(0.0, 0.0, np.nan, False) for _ in range(count)]
    if count < 2 or any(item.get("frame") is None for item in items):
        return identity
    geometry = []
    for item in items:
        center = item.get("dish_center")
        radius = item.get("dish_radius")
        if center is None or radius is None:
            debug = item.get("pre_debug") or {}
            center = debug.get("dish_center")
            radius = debug.get("dish_radius")
        if center is None or radius is None:
            return identity
        geometry.append((tuple(center), int(radius)))

    anchor_index = count // 2
    anchor_feature, window, scale = _motion_feature(
        items[anchor_index]["frame"], *geometry[anchor_index], max_dimension=max_dimension
    )
    transforms = []
    for index, item in enumerate(items):
        if index == anchor_index:
            transforms.append((0.0, 0.0, 1.0, True))
            continue
        feature, _, item_scale = _motion_feature(
            item["frame"], *geometry[index], max_dimension=max_dimension
        )
        if feature.shape != anchor_feature.shape or abs(item_scale - scale) > 1e-9:
            transforms.append((0.0, 0.0, np.nan, False))
            continue
        try:
            (shift_x, shift_y), response = cv2.phaseCorrelate(
                anchor_feature, feature, window
            )
        except cv2.error:
            transforms.append((0.0, 0.0, np.nan, False))
            continue
        shift_x /= scale
        shift_y /= scale
        valid = (
            np.isfinite(shift_x)
            and np.isfinite(shift_y)
            and np.isfinite(response)
            and float(response) >= float(min_response)
            and float(np.hypot(shift_x, shift_y)) <= float(max_shift_px)
        )
        # phaseCorrelate reports where the current frame moved relative to the
        # anchor.  The negative translation aligns the current frame to it.
        transforms.append(
            (-float(shift_x), -float(shift_y), float(response), bool(valid))
            if valid
            else (0.0, 0.0, float(response), False)
        )
    return transforms


def _warp_binary(mask, dx, dy):
    mask = np.asarray(mask, dtype=bool)
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    warped = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped.astype(bool)


def temporal_pale_branch_candidate(
    frame,
    dish_center,
    dish_radius,
    chroma_min=10,
    min_channel=35,
    max_value=240,
    rim_exclusion_px=30,
    min_obj_size=20,
):
    """Return low-threshold yellow candidate pixels for temporal recovery."""
    bgr = cv2.cvtColor(as_gray_uint8(frame), cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
    blue, green, red = cv2.split(bgr.astype(np.int16))
    yellow_chroma = np.minimum(red - blue, green - blue)
    max_channel = np.maximum(np.maximum(red, green), blue)
    inner_radius = int(dish_radius)
    if rim_exclusion_px is not None and int(rim_exclusion_px) > 0:
        inner_radius = max(1, int(dish_radius) - int(rim_exclusion_px))
    candidate = (
        (yellow_chroma >= int(chroma_min))
        & (red >= int(min_channel))
        & (green >= int(min_channel))
        & (max_channel <= int(max_value))
        & circular_roi_mask(frame.shape[:2], dish_center, inner_radius)
    )
    if min_obj_size and int(min_obj_size) > 0:
        candidate = morphology.remove_small_objects(candidate, min_size=int(min_obj_size))
    return candidate.astype(bool)


def apply_temporal_body_recovery_to_sequence(
    items,
    window=5,
    min_hits=4,
    connect_radius=8,
    exterior_only=True,
    block_hole_max_area=0,
    final_smooth_window=1,
    final_smooth_min_hits=None,
    motion_compensation=True,
    motion_max_shift_px=30.0,
    motion_min_response=0.35,
    motion_max_dimension=640,
):
    """Add persistent candidates without deleting the current strict mask.

    When frames and dish geometry are present, candidate masks are translated
    into a common dish-rim coordinate system before temporal voting.  Invalid
    or low-confidence estimates fall back to the original pixel coordinates.
    """
    if not items:
        return []
    window = max(3, int(window))
    window += int(window % 2 == 0)
    min_hits = max(1, min(int(min_hits), window))
    half_window = window // 2
    connect_radius = max(1, int(connect_radius))
    block_hole_max_area = max(0, int(block_hole_max_area or 0))
    final_smooth_window = max(1, int(final_smooth_window or 1))
    if final_smooth_window > 1 and final_smooth_window % 2 == 0:
        final_smooth_window += 1
    if final_smooth_min_hits is None:
        final_smooth_min_hits = final_smooth_window // 2 + 1
    final_smooth_min_hits = max(1, min(int(final_smooth_min_hits), final_smooth_window))

    original_candidates = [np.asarray(item["candidate"], dtype=bool) for item in items]
    original_strict = [np.asarray(item["mask"], dtype=bool) for item in items]
    if motion_compensation:
        transforms = _estimate_motion_to_anchor(
            items,
            max_shift_px=motion_max_shift_px,
            min_response=motion_min_response,
            max_dimension=motion_max_dimension,
        )
    else:
        transforms = [(0.0, 0.0, np.nan, False) for _ in items]
    candidates = [
        _warp_binary(mask, dx, dy) if valid else mask.copy()
        for mask, (dx, dy, _, valid) in zip(original_candidates, transforms)
    ]
    strict_masks = [
        _warp_binary(mask, dx, dy) if valid else mask.copy()
        for mask, (dx, dy, _, valid) in zip(original_strict, transforms)
    ]
    provisional = []
    for position, item in enumerate(items):
        strict_mask = strict_masks[position]
        addition = np.zeros_like(strict_mask)
        if half_window <= position < len(items) - half_window:
            hits = np.sum(
                np.stack(candidates[position - half_window : position + half_window + 1]),
                axis=0,
            )
            consensus = hits >= min_hits
            near_strict = morphology.binary_dilation(
                strict_mask, morphology.disk(connect_radius)
            )
            addition = keep_components_touching_mask(consensus, near_strict) & ~strict_mask
            if exterior_only:
                addition &= ~ndi.binary_fill_holes(strict_mask)
            elif block_hole_max_area > 0:
                holes = ndi.binary_fill_holes(strict_mask) & ~strict_mask
                labels, count = ndi.label(holes)
                if count:
                    areas = np.bincount(labels.ravel())
                    blocked = np.flatnonzero(
                        (areas <= block_hole_max_area) & (np.arange(areas.size) > 0)
                    )
                    addition &= ~np.isin(labels, blocked)

        provisional.append(keep_largest_component(strict_mask | addition))

    if final_smooth_window >= 3 and len(provisional) >= final_smooth_window:
        half = final_smooth_window // 2
        source_masks = [mask.copy() for mask in provisional]
        for position in range(half, len(provisional) - half):
            hits = np.sum(
                np.stack(source_masks[position - half : position + half + 1]), axis=0
            )
            current_strict = strict_masks[position]
            stable_addition = (hits >= final_smooth_min_hits) & ~current_strict
            provisional[position] = keep_largest_component(current_strict | stable_addition)

    recovered = []
    for item, work_mask, strict_original, transform in zip(
        items, provisional, original_strict, transforms
    ):
        dx, dy, response, valid = transform
        if valid:
            final_mask = _warp_binary(work_mask, -dx, -dy) | strict_original
        else:
            final_mask = work_mask | strict_original
        final_mask = keep_largest_component(final_mask)
        updated = dict(item)
        updated["strict_mask"] = strict_original
        updated["mask"] = final_mask
        updated["temporal_added_mask"] = final_mask & ~strict_original
        updated["motion_compensation_dx_px"] = float(dx)
        updated["motion_compensation_dy_px"] = float(dy)
        updated["motion_compensation_response"] = float(response)
        updated["motion_compensation_applied"] = bool(valid)
        recovered.append(updated)
    return recovered


def required_temporal_halo(window: int, final_smooth_window: int) -> int:
    """Return the composed halo needed by the two temporal stages."""
    return max(3, int(window)) // 2 + max(1, int(final_smooth_window)) // 2
