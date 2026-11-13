"""Spatial HSV, yellow-chroma, and conservative vesselness segmentation."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology

from .mask_utils import (
    as_gray_uint8,
    circular_roi_mask,
    keep_components_touching_mask,
    keep_largest_component,
)


def recover_vesselness_near_body(
    frame,
    body_mask,
    dish_center,
    dish_radius,
    percentile=95.0,
    min_response=0.008,
    connect_radius=12,
    min_obj_size=20,
    max_added_fraction=0.02,
    chroma_min=4,
    min_channel=28,
    max_value=245,
    sigma_min=1.0,
    sigma_max=4.0,
    black_ridges=False,
):
    """Recover ridge-like pixels while enforcing a true component-level cap."""
    body_mask = np.asarray(body_mask, dtype=bool)
    empty = np.zeros_like(body_mask)
    if frame is None or not np.any(body_mask):
        return empty, empty, np.nan, np.nan
    bgr = cv2.cvtColor(as_gray_uint8(frame), cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
    dish_mask = circular_roi_mask(body_mask.shape, dish_center, dish_radius)
    near_body = morphology.binary_dilation(
        body_mask, morphology.disk(max(1, int(connect_radius)))
    ) & dish_mask
    search_mask = near_body & ~body_mask
    if not np.any(search_mask):
        return empty, empty, np.nan, np.nan

    yy, xx = np.where(near_body)
    sigma_min = max(float(sigma_min), 0.5)
    sigma_max = max(float(sigma_max), sigma_min)
    pad = max(12, int(np.ceil(sigma_max * 5)))
    y0, y1 = max(0, int(yy.min()) - pad), min(body_mask.shape[0], int(yy.max()) + pad + 1)
    x0, x1 = max(0, int(xx.min()) - pad), min(body_mask.shape[1], int(xx.max()) + pad + 1)
    gray = as_gray_uint8(bgr[y0:y1, x0:x1]).astype(np.float32) / 255.0
    near_crop = near_body[y0:y1, x0:x1]
    work = exposure.equalize_adapthist(gray, clip_limit=0.02)
    work[~near_crop] = float(np.nanmedian(work[near_crop]))
    sigmas = np.linspace(
        sigma_min,
        sigma_max,
        num=max(2, int(np.ceil(sigma_max - sigma_min)) + 1),
    )
    response_crop = filters.frangi(work, sigmas=sigmas, black_ridges=bool(black_ridges))
    response = np.zeros(body_mask.shape, dtype=float)
    response[y0:y1, x0:x1] = np.nan_to_num(response_crop)
    percentile = float(np.clip(percentile, 0.0, 100.0))
    threshold = max(
        float(np.nanpercentile(response[search_mask], percentile)), float(min_response)
    )

    blue, green, red = cv2.split(bgr.astype(np.int16))
    chroma = np.minimum(red - blue, green - blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    maximum = np.maximum(np.maximum(red, green), blue)
    color_gate = (
        (chroma >= int(chroma_min))
        & (minimum >= int(min_channel))
        & (maximum <= int(max_value))
        & dish_mask
    )
    candidate = (response >= threshold) & search_mask & color_gate
    if min_obj_size and int(min_obj_size) > 0:
        candidate = morphology.remove_small_objects(candidate, min_size=int(min_obj_size))

    max_added = max(0, int(np.floor(body_mask.sum() * max(0.0, float(max_added_fraction)))))
    if max_added == 0:
        candidate = empty
    elif int(candidate.sum()) > max_added:
        labels, count = ndi.label(candidate)
        components = []
        for label_id in range(1, count + 1):
            component = labels == label_id
            components.append((float(np.nanmean(response[component])), int(component.sum()), label_id))
        components.sort(reverse=True)
        capped = np.zeros_like(candidate)
        used = 0
        for _, size, label_id in components:
            if used + size > max_added:
                continue
            capped |= labels == label_id
            used += size
        candidate = capped
    added = candidate & ~body_mask
    if int(added.sum()) > max_added:
        raise AssertionError("vesselness hard cap was exceeded")
    return added, candidate, threshold, float(np.nanmean(response[search_mask]))


def segment_hsv_circle(
    frame,
    dish_center,
    dish_radius,
    h_min=16,
    h_max=48,
    s_min=60,
    v_min=45,
    min_obj_size=150,
    min_hole_size=0,
    morph_open_size=1,
    morph_close_size=0,
    keep_largest=True,
    fill_small_body_holes=False,
    body_hole_area_ratio=0.002,
    body_min_hole_size=150,
    body_max_hole_size=None,
    use_yellow_chroma_extension=False,
    yellow_chroma_min=20,
    yellow_chroma_max_value=225,
    yellow_chroma_min_channel=55,
    yellow_chroma_rim_exclusion_px=8,
    yellow_chroma_connect_radius=0,
    yellow_chroma_connect_iterations=3,
    use_relaxed_body_mask=False,
    relaxed_h_min=14,
    relaxed_h_max=55,
    relaxed_s_min=25,
    relaxed_v_min=35,
    relaxed_connect_radius=30,
    relaxed_min_obj_size=50,
    rim_exclusion_px=40,
    use_vesselness_recovery=False,
    vesselness_percentile=95.0,
    vesselness_min_response=0.008,
    vesselness_connect_radius=12,
    vesselness_min_obj_size=20,
    vesselness_max_added_fraction=0.02,
    vesselness_chroma_min=4,
    vesselness_min_channel=28,
    vesselness_max_value=245,
    vesselness_sigma_min=1.0,
    vesselness_sigma_max=4.0,
    vesselness_black_ridges=False,
):
    bgr = cv2.cvtColor(as_gray_uint8(frame), cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    dish_mask = circular_roi_mask(frame.shape[:2], dish_center, dish_radius)
    seed = (
        (hue >= int(h_min)) & (hue <= int(h_max))
        & (saturation >= int(s_min)) & (value >= int(v_min)) & dish_mask
    )
    if morph_open_size and morph_open_size > 0:
        seed = morphology.binary_opening(seed, morphology.disk(int(morph_open_size)))
    if morph_close_size and morph_close_size > 0:
        seed = morphology.binary_closing(seed, morphology.disk(int(morph_close_size)))
    if min_obj_size and min_obj_size > 0:
        seed = morphology.remove_small_objects(seed, min_size=int(min_obj_size))
    if min_hole_size and min_hole_size > 0:
        seed = morphology.remove_small_holes(seed, area_threshold=int(min_hole_size))
    if keep_largest:
        seed = keep_largest_component(seed)

    yellow_candidate = np.zeros_like(seed)
    yellow_added = np.zeros_like(seed)
    if use_yellow_chroma_extension and np.any(seed):
        blue, green, red = cv2.split(bgr.astype(np.int16))
        chroma = np.minimum(red - blue, green - blue)
        maximum = np.maximum(np.maximum(red, green), blue)
        yellow_candidate = (
            (chroma >= int(yellow_chroma_min))
            & (red >= int(yellow_chroma_min_channel))
            & (green >= int(yellow_chroma_min_channel))
            & (maximum <= int(yellow_chroma_max_value)) & dish_mask
        )
        if yellow_chroma_rim_exclusion_px and dish_radius is not None:
            yellow_candidate &= circular_roi_mask(
                frame.shape[:2], dish_center,
                max(1, int(dish_radius) - int(yellow_chroma_rim_exclusion_px)),
            )
        connected = ndi.binary_propagation(
            seed, structure=np.ones((3, 3), dtype=bool), mask=seed | yellow_candidate
        )
        radius = max(0, int(yellow_chroma_connect_radius))
        for _ in range(max(0, int(yellow_chroma_connect_iterations)) if radius else 0):
            nearby = morphology.binary_dilation(connected, morphology.disk(radius))
            expanded = connected | keep_components_touching_mask(yellow_candidate & ~connected, nearby)
            if np.array_equal(expanded, connected):
                break
            connected = expanded
        yellow_added = connected & ~seed

    relaxed_candidate = np.zeros_like(seed)
    relaxed_added = np.zeros_like(seed)
    if use_relaxed_body_mask and np.any(seed):
        relaxed_candidate = (
            (hue >= int(relaxed_h_min)) & (hue <= int(relaxed_h_max))
            & (saturation >= int(relaxed_s_min)) & (value >= int(relaxed_v_min)) & dish_mask
        )
        if rim_exclusion_px and dish_radius is not None:
            relaxed_candidate &= circular_roi_mask(
                frame.shape[:2], dish_center, max(1, int(dish_radius) - int(rim_exclusion_px))
            )
        near = morphology.binary_dilation(seed, morphology.disk(max(1, int(relaxed_connect_radius))))
        relaxed_candidate &= near
        if relaxed_min_obj_size and int(relaxed_min_obj_size) > 0:
            relaxed_candidate = morphology.remove_small_objects(
                relaxed_candidate, min_size=int(relaxed_min_obj_size)
            )
        relaxed_added = keep_components_touching_mask(relaxed_candidate, near) & ~seed

    occupied = (seed | yellow_added | relaxed_added).astype(bool)
    vessel_candidate = np.zeros_like(seed)
    vessel_added = np.zeros_like(seed)
    vessel_threshold = np.nan
    vessel_mean = np.nan
    if use_vesselness_recovery and np.any(occupied):
        vessel_added, vessel_candidate, vessel_threshold, vessel_mean = recover_vesselness_near_body(
            bgr, occupied, dish_center, dish_radius,
            percentile=vesselness_percentile, min_response=vesselness_min_response,
            connect_radius=vesselness_connect_radius, min_obj_size=vesselness_min_obj_size,
            max_added_fraction=vesselness_max_added_fraction, chroma_min=vesselness_chroma_min,
            min_channel=vesselness_min_channel, max_value=vesselness_max_value,
            sigma_min=vesselness_sigma_min, sigma_max=vesselness_sigma_max,
            black_ridges=vesselness_black_ridges,
        )
        occupied |= vessel_added
    if keep_largest:
        occupied = keep_largest_component(occupied)
        seed &= occupied
        yellow_candidate &= occupied
        yellow_added &= occupied
        relaxed_candidate &= occupied
        relaxed_added &= occupied
        vessel_candidate &= occupied
        vessel_added &= occupied

    envelope = ndi.binary_fill_holes(occupied).astype(bool)
    clean = occupied.copy()
    hole_threshold = 0
    if fill_small_body_holes and np.any(occupied):
        raw = max(int(body_min_hole_size), int(round(occupied.sum() * float(body_hole_area_ratio))))
        hole_threshold = min(int(body_max_hole_size), raw) if body_max_hole_size else raw
        clean = morphology.remove_small_holes(occupied, area_threshold=hole_threshold)

    occupied_area = max(int(occupied.sum()), 1)
    debug = {
        "gray": as_gray_uint8(frame), "background": np.zeros(frame.shape[:2], np.uint8),
        "corrected": saturation, "enhanced": value, "analysis_roi": None,
        "dish_center": dish_center, "dish_radius": dish_radius,
        "seed_mask": seed, "yellow_chroma_candidate_mask": yellow_candidate,
        "yellow_chroma_added_mask": yellow_added, "relaxed_candidate_mask": relaxed_candidate,
        "relaxed_added_mask": relaxed_added, "vesselness_candidate_mask": vessel_candidate,
        "vesselness_added_mask": vessel_added, "occupied_mask": occupied,
        "clean_mask": clean, "envelope_mask": envelope,
        "body_hole_threshold_px": int(hole_threshold),
        "body_hole_area_ratio": float(body_hole_area_ratio),
        "body_min_hole_size": int(body_min_hole_size),
        "body_max_hole_size": int(body_max_hole_size or 0),
        "seed_area_px": int(seed.sum()),
        "yellow_chroma_added_area_px": int(yellow_added.sum()),
        "yellow_chroma_added_fraction": float(yellow_added.sum() / occupied_area),
        "use_yellow_chroma_extension": bool(use_yellow_chroma_extension),
        "yellow_chroma_min": int(yellow_chroma_min),
        "yellow_chroma_max_value": int(yellow_chroma_max_value),
        "relaxed_added_area_px": int(relaxed_added.sum()),
        "relaxed_added_fraction": float(relaxed_added.sum() / occupied_area),
        "use_relaxed_body_mask": bool(use_relaxed_body_mask),
        "relaxed_s_min": int(relaxed_s_min),
        "vesselness_added_area_px": int(vessel_added.sum()),
        "vesselness_added_fraction": float(vessel_added.sum() / occupied_area),
        "use_vesselness_recovery": bool(use_vesselness_recovery),
        "vesselness_threshold": vessel_threshold,
        "vesselness_response_mean": vessel_mean,
        "vesselness_percentile": float(vesselness_percentile),
        "vesselness_connect_radius": int(vesselness_connect_radius),
    }
    return value, clean.astype(bool), debug

