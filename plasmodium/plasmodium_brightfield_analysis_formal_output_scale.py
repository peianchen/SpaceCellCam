"""
Bright-field time-lapse analysis for plasmodium morphology and dynamics.

Install dependencies if needed:
pip install opencv-python numpy pandas matplotlib scipy scikit-image tqdm

This script is written as a plain Python file for direct execution in Spyder.

Parameter tuning notes for common bright-field problems:
1. Severe uneven background:
   - Increase bg_kernel
   - Try bg_method="rolling_ball"
2. Plasmodium is darker than background:
   - Set invert_if_needed=True
3. Many small debris particles:
   - Increase min_obj_size
   - Set keep_largest=True
4. Broken boundaries:
   - Increase morph_close_size
   - Increase gaussian_sigma
5. Segmentation includes too much background:
   - Try method="adaptive"
   - Adjust CLAHE / bg_kernel
   - For full-dish videos, set ANALYSIS_ROI around the plasmodium
   - Try correction_mode="absdiff" and component_selection="center"
6. Focus drift:
   - Use the focus_score column
   - Later exclude frames with obviously low focus_score

Definitions:
1. Area:
   A(t) = sum(M_t) * pixel_size^2
   where M_t is the binary mask.
2. Area growth rate:
   dA/dt = [A(t) - A(t-1)] / dt
3. Newly explored region:
   E_t = M_t AND NOT M_(t-1)
4. Retracted region:
   R_t = M_(t-1) AND NOT M_t
5. Explore ratio:
   explore_ratio = |E_t| / |M_t|
6. Retreat ratio:
   retreat_ratio = |R_t| / |M_(t-1)|
7. Circularity:
   circularity = 4*pi*A / P^2
8. Solidity:
   solidity = A / convex_hull_area
9. Skeleton endpoint / branchpoint:
   For each skeleton pixel, count 8-neighborhood pixels:
   endpoint: neighbor_count == 1
   branchpoint: neighbor_count >= 3
10. ROI grayscale oscillation:
   g(t) = mean intensity inside ROI
   Use detrended FFT to estimate dominant frequency:
   period = 1 / dominant_freq
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import gc
import os
import shutil
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import detrend, find_peaks
from skimage import exposure, filters, measure, morphology, util
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _as_gray_uint8(img):
    """Convert an image array to grayscale uint8."""
    if img is None:
        raise ValueError("Input image is None.")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype == np.uint8:
        return gray

    gray = gray.astype(np.float32)
    finite = np.isfinite(gray)
    if not np.any(finite):
        return np.zeros(gray.shape, dtype=np.uint8)

    lo, hi = np.nanpercentile(gray[finite], [1, 99])
    if hi <= lo:
        hi = np.nanmax(gray[finite])
        lo = np.nanmin(gray[finite])
    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.uint8)

    gray = np.clip((gray - lo) / (hi - lo), 0, 1)
    return (gray * 255).astype(np.uint8)


def _normalize_uint8(img):
    """Normalize numeric image to uint8 using robust percentiles."""
    arr = img.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def load_frames_from_folder(folder, as_gray=True):
    """
    Load image sequence from a folder.

    Supports png, jpg, jpeg, tif, tiff, bmp.
    Returns frames, names.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Input path is not a folder: {folder}")

    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    frames = []
    names = []

    for path in tqdm(files, desc="Loading image frames"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            warnings.warn(f"Could not read image: {path}")
            continue
        if as_gray:
            frames.append(_as_gray_uint8(img))
        else:
            if img.ndim == 2:
                img = cv2.cvtColor(_as_gray_uint8(img), cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frames.append(img)
        names.append(path.name)

    if len(frames) == 0:
        raise ValueError(f"No readable image frames found in: {folder}")

    return frames, names


def load_frames_from_video(video_path, max_frames=None, as_gray=True):
    """
    Load frames from a video file.

    Returns frames, names.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Input path is not a file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    names = []
    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar_total = min(total, max_frames) if max_frames is not None and total > 0 else total

    with tqdm(total=pbar_total if pbar_total > 0 else None, desc="Loading video frames") as pbar:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if as_gray:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                frames.append(frame)
            names.append(f"frame_{idx:06d}")
            idx += 1
            pbar.update(1)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    return frames, names


def focus_score_laplacian(gray):
    """Estimate focus quality using Laplacian variance."""
    gray = _as_gray_uint8(gray)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_background(img, bg_method="morph_open", bg_kernel=81):
    """
    Estimate bright-field background.

    bg_method:
    - "morph_open": morphological opening with elliptical kernel
    - "gaussian": strong Gaussian blur
    - "rolling_ball": skimage restoration.rolling_ball
    """
    gray = _as_gray_uint8(img)
    k = int(bg_kernel)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    if bg_method == "morph_open":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        return bg.astype(np.float32)

    if bg_method == "gaussian":
        bg = cv2.GaussianBlur(gray, (k, k), sigmaX=0)
        return bg.astype(np.float32)

    if bg_method == "rolling_ball":
        try:
            from skimage.restoration import rolling_ball

            radius = max(1, k // 2)
            bg = rolling_ball(gray.astype(np.float32), radius=radius)
            return bg.astype(np.float32)
        except Exception as exc:
            warnings.warn(
                "rolling_ball background estimation failed. "
                "Falling back to bg_method='morph_open'. "
                f"Original error: {exc}"
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            return bg.astype(np.float32)

    raise ValueError("bg_method must be 'morph_open', 'gaussian', or 'rolling_ball'.")


def keep_largest_component(mask):
    """
    Keep only the largest connected component in a binary mask.

    Returns an all-false mask when the input is empty.
    """
    mask = mask.astype(bool)
    label_img = measure.label(mask)
    props = measure.regionprops(label_img)
    if len(props) == 0:
        return np.zeros_like(mask, dtype=bool)

    selected = max(props, key=lambda r: r.area)
    return (label_img == selected.label).astype(bool)


def keep_components_touching_mask(candidate_mask, reference_mask):
    """
    Keep connected components in candidate_mask that overlap reference_mask.

    This is used by relaxed HSV segmentation: low-saturation candidate pixels
    are accepted only when their connected component touches a trusted seed
    neighborhood.
    """
    candidate_mask = candidate_mask.astype(bool)
    reference_mask = reference_mask.astype(bool)
    if candidate_mask.shape != reference_mask.shape:
        raise ValueError(
            "candidate_mask and reference_mask must have the same shape: "
            f"{candidate_mask.shape} vs {reference_mask.shape}"
        )
    if not np.any(candidate_mask) or not np.any(reference_mask):
        return np.zeros_like(candidate_mask, dtype=bool)

    label_img = measure.label(candidate_mask)
    labels = np.unique(label_img[reference_mask & (label_img > 0)])
    if labels.size == 0:
        return np.zeros_like(candidate_mask, dtype=bool)
    return np.isin(label_img, labels).astype(bool)



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
    if frame.ndim == 2:
        bgr = cv2.cvtColor(_as_gray_uint8(frame), cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame

    blue, green, red = cv2.split(bgr.astype(np.int16))
    yellow_chroma = np.minimum(red - blue, green - blue)
    max_channel = np.maximum(np.maximum(red, green), blue)
    inner_radius = int(dish_radius)
    if rim_exclusion_px is not None and int(rim_exclusion_px) > 0:
        inner_radius = max(1, int(dish_radius) - int(rim_exclusion_px))
    inner_dish = circular_roi_mask(frame.shape[:2], dish_center, inner_radius)
    candidate = (
        (yellow_chroma >= int(chroma_min))
        & (red >= int(min_channel))
        & (green >= int(min_channel))
        & (max_channel <= int(max_value))
        & inner_dish
    )
    if min_obj_size and int(min_obj_size) > 0:
        candidate = morphology.remove_small_objects(
            candidate.astype(bool),
            min_size=int(min_obj_size),
        )
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
):
    """
    Add persistent pale branch pixels to ordered mask items.

    Each item must contain: frame_idx, frame, mask, candidate.
    The first/last half-window items are returned unchanged because they cannot
    be checked symmetrically.
    """
    if not items:
        return []
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    min_hits = max(1, min(int(min_hits), window))
    half_window = window // 2
    connect_radius = max(1, int(connect_radius))
    exterior_only = bool(exterior_only)
    block_hole_max_area = max(0, int(block_hole_max_area or 0))
    final_smooth_window = max(1, int(final_smooth_window or 1))
    if final_smooth_window > 1 and final_smooth_window % 2 == 0:
        final_smooth_window += 1
    if final_smooth_min_hits is None:
        final_smooth_min_hits = final_smooth_window // 2 + 1
    final_smooth_min_hits = max(
        1,
        min(int(final_smooth_min_hits), final_smooth_window),
    )
    recovered = []
    candidates = [item["candidate"].astype(bool) for item in items]

    for pos, item in enumerate(items):
        strict_mask = item["mask"].astype(bool)
        addition = np.zeros_like(strict_mask, dtype=bool)
        if half_window <= pos < len(items) - half_window:
            local_candidates = candidates[pos - half_window:pos + half_window + 1]
            hits = np.sum(np.stack(local_candidates, axis=0), axis=0)
            consensus = hits >= min_hits
            near_strict = morphology.binary_dilation(
                strict_mask,
                morphology.disk(connect_radius),
            )
            trusted = keep_components_touching_mask(consensus, near_strict)
            addition = trusted & (~strict_mask)
            if exterior_only:
                strict_envelope = ndi.binary_fill_holes(strict_mask).astype(bool)
                addition &= ~strict_envelope
            elif block_hole_max_area > 0:
                strict_envelope = ndi.binary_fill_holes(strict_mask).astype(bool)
                internal_holes = strict_envelope & (~strict_mask)
                hole_labels, n_holes = ndi.label(internal_holes)
                if n_holes > 0:
                    hole_areas = np.bincount(hole_labels.ravel())
                    blocked_ids = np.flatnonzero(
                        (hole_areas <= block_hole_max_area)
                        & (np.arange(hole_areas.size) > 0)
                    )
                    if blocked_ids.size > 0:
                        addition &= ~np.isin(hole_labels, blocked_ids)

        final_mask = keep_largest_component(strict_mask | addition)
        updated = dict(item)
        updated["strict_mask"] = strict_mask
        updated["temporal_added_mask"] = addition & final_mask
        updated["mask"] = final_mask
        recovered.append(updated)

    if final_smooth_window >= 3 and len(recovered) >= final_smooth_window:
        smooth_half = final_smooth_window // 2
        provisional_masks = [item["mask"].astype(bool) for item in recovered]
        strict_masks = [item["mask"].astype(bool) for item in items]
        for pos in range(smooth_half, len(recovered) - smooth_half):
            local_final = provisional_masks[pos - smooth_half:pos + smooth_half + 1]
            local_strict = strict_masks[pos - smooth_half:pos + smooth_half + 1]
            final_hits = np.sum(np.stack(local_final, axis=0), axis=0)
            strict_hits = np.sum(np.stack(local_strict, axis=0), axis=0)
            stable_mask = keep_largest_component(
                final_hits >= final_smooth_min_hits
            )
            stable_strict = strict_hits >= final_smooth_min_hits
            recovered[pos]["mask"] = stable_mask
            recovered[pos]["temporal_added_mask"] = stable_mask & (~stable_strict)

    return recovered

def preprocess_frame(
    gray,
    bg_method="morph_open",
    bg_kernel=81,
    correction_mode="subtract",
    clahe_clip=2.0,
    clahe_grid=(8, 8),
    median_ksize=3,
    gaussian_sigma=1.0,
    invert_if_needed=False,
    return_debug=False,
):
    """
    Preprocess one bright-field frame.

    Steps:
    - Background estimation and correction
    - Illumination normalization
    - Normalize to 0-255
    - CLAHE contrast enhancement
    - Median denoising
    - Gaussian smoothing
    - Optional inversion
    """
    gray = _as_gray_uint8(gray)
    bg = estimate_background(gray, bg_method=bg_method, bg_kernel=bg_kernel)

    gray_float = gray.astype(np.float32)
    if correction_mode == "subtract":
        corrected = gray_float - bg + np.nanmedian(bg)
    elif correction_mode == "absdiff":
        corrected = np.abs(gray_float - bg)
    elif correction_mode == "divide":
        corrected = gray_float / (bg + 1.0) * np.nanmedian(bg)
    else:
        raise ValueError("correction_mode must be 'subtract', 'absdiff', or 'divide'.")

    corrected_u8 = _normalize_uint8(corrected)

    if invert_if_needed:
        corrected_u8 = cv2.bitwise_not(corrected_u8)

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
    enhanced = clahe.apply(corrected_u8)

    if median_ksize is not None and int(median_ksize) > 1:
        m = int(median_ksize)
        if m % 2 == 0:
            m += 1
        enhanced = cv2.medianBlur(enhanced, m)

    if gaussian_sigma is not None and float(gaussian_sigma) > 0:
        enhanced = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=float(gaussian_sigma))

    if return_debug:
        return enhanced, {
            "gray": gray,
            "background": _normalize_uint8(bg),
            "corrected": corrected_u8,
            "enhanced": enhanced,
        }
    return enhanced


def segment_plasmodium(
    enhanced,
    method="otsu",
    min_obj_size=1000,
    min_hole_size=1000,
    keep_largest=True,
    component_selection="largest",
    morph_open_size=2,
    morph_close_size=4,
    adaptive_block_size=101,
    adaptive_offset=-3,
    threshold_scale=1.0,
    grabcut_margin=0.18,
    grabcut_iter=5,
):
    """
    Segment plasmodium from a preprocessed frame.

    Supports Otsu or adaptive threshold, followed by morphology,
    remove_small_objects, remove_small_holes, and optional component selection.

    component_selection:
    - "largest": keep the largest component
    - "center": keep the component nearest the image center, weighted by area
    - "none": keep all remaining components
    """
    img = _as_gray_uint8(enhanced)

    if method == "otsu":
        thresh = filters.threshold_otsu(img) * float(threshold_scale)
        mask = img > thresh
    elif method == "adaptive":
        block = int(adaptive_block_size)
        if block < 3:
            block = 3
        if block % 2 == 0:
            block += 1
        local_thresh = filters.threshold_local(img, block_size=block, offset=float(adaptive_offset))
        mask = img > local_thresh
    elif method == "grabcut":
        h, w = img.shape[:2]
        margin = float(grabcut_margin)
        margin = min(max(margin, 0.01), 0.45)
        mx = int(round(w * margin))
        my = int(round(h * margin))
        rect_w = max(1, w - 2 * mx)
        rect_h = max(1, h - 2 * my)
        rect = (mx, my, rect_w, rect_h)
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            cv2.grabCut(
                bgr,
                gc_mask,
                rect,
                bgd_model,
                fgd_model,
                int(grabcut_iter),
                cv2.GC_INIT_WITH_RECT,
            )
            mask = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except cv2.error as exc:
            warnings.warn(f"GrabCut failed; falling back to Otsu threshold. Original error: {exc}")
            thresh = filters.threshold_otsu(img) * float(threshold_scale)
            mask = img > thresh
    else:
        raise ValueError("method must be 'otsu', 'adaptive', or 'grabcut'.")

    if morph_open_size and morph_open_size > 0:
        selem = morphology.disk(int(morph_open_size))
        mask = morphology.binary_opening(mask, selem)

    if morph_close_size and morph_close_size > 0:
        selem = morphology.disk(int(morph_close_size))
        mask = morphology.binary_closing(mask, selem)

    if min_obj_size and min_obj_size > 0:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_obj_size))

    if min_hole_size and min_hole_size > 0:
        mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=int(min_hole_size))

    if keep_largest and component_selection == "none":
        component_selection = "largest"

    if component_selection != "none":
        if component_selection == "largest":
            mask = keep_largest_component(mask)
        else:
            label_img = measure.label(mask)
            props = measure.regionprops(label_img)
            if len(props) > 0:
                if component_selection == "center":
                    cy = (mask.shape[0] - 1) / 2.0
                    cx = (mask.shape[1] - 1) / 2.0

                    def center_score(region):
                        dy = region.centroid[0] - cy
                        dx = region.centroid[1] - cx
                        distance = np.sqrt(dx * dx + dy * dy)
                        area_bonus = np.sqrt(max(float(region.area), 1.0))
                        return distance / area_bonus

                    selected = min(props, key=center_score)
                else:
                    raise ValueError("component_selection must be 'largest', 'center', or 'none'.")
                mask = label_img == selected.label
            else:
                mask = np.zeros_like(mask, dtype=bool)
    elif keep_largest:
        mask = keep_largest_component(mask)

    return mask.astype(bool)


def region_metrics(mask, pixel_size=1.0):
    """
    Compute region morphology metrics from a binary mask.

    Returns area_px, area, perimeter_px, perimeter, circularity, solidity,
    bbox_area_px, eccentricity.
    """
    mask = mask.astype(bool)
    area_px = int(np.sum(mask))
    if area_px == 0:
        return {
            "area_px": 0,
            "area": 0.0,
            "perimeter_px": 0.0,
            "perimeter": 0.0,
            "circularity": np.nan,
            "solidity": np.nan,
            "bbox_area_px": 0,
            "eccentricity": np.nan,
        }

    label_img = measure.label(mask)
    props = measure.regionprops(label_img)
    if len(props) == 0:
        return {
            "area_px": area_px,
            "area": area_px * pixel_size**2,
            "perimeter_px": np.nan,
            "perimeter": np.nan,
            "circularity": np.nan,
            "solidity": np.nan,
            "bbox_area_px": np.nan,
            "eccentricity": np.nan,
        }

    prop = max(props, key=lambda r: r.area)
    perimeter_px = float(measure.perimeter(mask, neighborhood=8))
    area = float(area_px * pixel_size**2)
    perimeter = float(perimeter_px * pixel_size)
    circularity = np.nan
    if perimeter_px > 0:
        circularity = float(4.0 * np.pi * area_px / (perimeter_px**2))

    return {
        "area_px": area_px,
        "area": area,
        "perimeter_px": perimeter_px,
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": float(prop.solidity) if np.isfinite(prop.solidity) else np.nan,
        "bbox_area_px": int(prop.bbox_area),
        "eccentricity": float(prop.eccentricity),
    }


def skeleton_metrics(mask, pixel_size=1.0):
    """
    Compute skeleton length, endpoints, and branchpoints.

    Endpoint and branchpoint are based on 8-neighborhood counts.
    """
    mask = mask.astype(bool)
    if np.sum(mask) == 0:
        empty = np.zeros_like(mask, dtype=bool)
        return {
            "skeleton": empty,
            "endpoint_mask": empty,
            "branch_mask": empty,
            "n_endpoints": 0,
            "n_branchpoints": 0,
            "skeleton_length_px": 0,
            "skeleton_length": 0.0,
        }

    skeleton = morphology.skeletonize(mask)
    sk_u8 = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndi.convolve(sk_u8, kernel, mode="constant", cval=0) - sk_u8

    endpoint_mask = skeleton & (neighbor_count == 1)
    branch_mask = skeleton & (neighbor_count >= 3)
    skeleton_length_px = int(np.sum(skeleton))

    return {
        "skeleton": skeleton,
        "endpoint_mask": endpoint_mask,
        "branch_mask": branch_mask,
        "n_endpoints": int(np.sum(endpoint_mask)),
        "n_branchpoints": int(np.sum(branch_mask)),
        "skeleton_length_px": skeleton_length_px,
        "skeleton_length": float(skeleton_length_px * pixel_size),
    }


def empty_skeleton_metrics_like(mask):
    """Return empty skeleton metric placeholders for fast preview mode."""
    empty = np.zeros_like(mask, dtype=bool)
    return {
        "skeleton": empty,
        "endpoint_mask": empty,
        "branch_mask": empty,
        "n_endpoints": np.nan,
        "n_branchpoints": np.nan,
        "skeleton_length_px": np.nan,
        "skeleton_length": np.nan,
    }


def vein_network_metrics(
    frame,
    dish_center=None,
    dish_radius=None,
    body_mask=None,
    method="ridge",
    h_min=16,
    h_max=55,
    s_min=45,
    v_min=80,
    min_obj_size=30,
    morph_open_size=0,
    body_erode_size=1,
    ridge_sigma_min=1,
    ridge_sigma_max=4,
    ridge_threshold_percentile=92,
    ridge_min_response=0.01,
    ridge_black_ridges=False,
    ridge_use_hsv_gate=False,
    pixel_size=1.0,
):
    """
    Detect visible yellow vein/tube network separately from the whole-body mask.

    This is an image-feature mask, not the total plasmodium occupancy mask.
    It is useful for visualizing and quantifying fine yellow tubes.
    """
    if frame is None:
        empty = np.zeros((1, 1), dtype=bool)
        return {
            "vein_mask": empty,
            "vein_skeleton": empty,
            "vein_endpoint_mask": empty,
            "vein_branch_mask": empty,
            "vein_area_px": 0,
            "vein_length_px": 0,
            "vein_length": 0.0,
            "vein_endpoints": 0,
            "vein_branchpoints": 0,
            "vein_ridge_threshold": np.nan,
            "vein_ridge_response_mean": np.nan,
        }

    if frame.ndim == 2:
        bgr = cv2.cvtColor(_as_gray_uint8(frame), cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame

    circular_mask = circular_roi_mask(bgr.shape[:2], dish_center, dish_radius)
    if body_mask is None:
        analysis_mask = circular_mask
    else:
        analysis_mask = body_mask.astype(bool) & circular_mask
        if body_erode_size and body_erode_size > 0 and np.any(analysis_mask):
            analysis_mask = morphology.binary_erosion(
                analysis_mask, morphology.disk(int(body_erode_size))
            )

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    hsv_gate = (
        (hue >= int(h_min))
        & (hue <= int(h_max))
        & (sat >= int(s_min))
        & (val >= int(v_min))
        & analysis_mask
    )

    if method == "hsv":
        vein_mask = hsv_gate
        ridge_response = np.zeros(bgr.shape[:2], dtype=float)
        ridge_threshold = np.nan
    elif method == "ridge":
        sigma_min = max(float(ridge_sigma_min), 0.5)
        sigma_max = max(float(ridge_sigma_max), sigma_min)
        sigmas = np.linspace(sigma_min, sigma_max, num=max(2, int(np.ceil(sigma_max - sigma_min)) + 1))

        ridge_response = np.zeros(bgr.shape[:2], dtype=float)
        if np.any(analysis_mask):
            ys, xs = np.where(analysis_mask)
            pad = int(max(8, np.ceil(sigma_max * 4)))
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(bgr.shape[0], int(ys.max()) + pad + 1)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(bgr.shape[1], int(xs.max()) + pad + 1)

            gray_crop = _as_gray_uint8(bgr[y0:y1, x0:x1]).astype(np.float32) / 255.0
            mask_crop = analysis_mask[y0:y1, x0:x1]
            work = exposure.equalize_adapthist(gray_crop, clip_limit=0.02)
            fill_value = float(np.nanmedian(work[mask_crop]))
            work[~mask_crop] = fill_value
            ridge_crop = filters.frangi(
                work,
                sigmas=sigmas,
                black_ridges=bool(ridge_black_ridges),
            )
            ridge_crop = np.nan_to_num(ridge_crop, nan=0.0, posinf=0.0, neginf=0.0)
            ridge_response[y0:y1, x0:x1] = ridge_crop
            ridge_threshold = float(
                np.nanpercentile(ridge_response[analysis_mask], float(ridge_threshold_percentile))
            )
        else:
            ridge_threshold = np.inf

        ridge_threshold = max(ridge_threshold, float(ridge_min_response))
        vein_mask = (ridge_response >= ridge_threshold) & analysis_mask
        if ridge_use_hsv_gate:
            vein_mask &= hsv_gate
    else:
        raise ValueError('vein method must be "ridge" or "hsv".')

    if morph_open_size and morph_open_size > 0:
        vein_mask = morphology.binary_opening(vein_mask, morphology.disk(int(morph_open_size)))
    if min_obj_size and min_obj_size > 0:
        vein_mask = morphology.remove_small_objects(vein_mask.astype(bool), min_size=int(min_obj_size))

    sk = skeleton_metrics(vein_mask, pixel_size=pixel_size)
    return {
        "vein_mask": vein_mask.astype(bool),
        "vein_skeleton": sk["skeleton"],
        "vein_endpoint_mask": sk["endpoint_mask"],
        "vein_branch_mask": sk["branch_mask"],
        "vein_area_px": int(np.sum(vein_mask)),
        "vein_length_px": sk["skeleton_length_px"],
        "vein_length": sk["skeleton_length"],
        "vein_endpoints": sk["n_endpoints"],
        "vein_branchpoints": sk["n_branchpoints"],
        "vein_ridge_threshold": ridge_threshold,
        "vein_ridge_response_mean": (
            float(np.nanmean(ridge_response[analysis_mask])) if np.any(analysis_mask) else np.nan
        ),
    }


def apply_vein_confidence(
    df,
    warmup_frames=5,
    min_vein_area_px=50,
    min_vein_length_px=80,
    min_length_norm=3.0,
    max_area_fraction=0.35,
    max_rel_change=0.60,
    stable_window=3,
    require_consecutive=3,
):
    """
    Add conservative trusted vein columns.

    Raw vein columns are preserved. The trusted columns are NaN until the
    visible line network is strong enough and stable enough for interpretation.
    """
    if df is None or df.empty or "vein_length_px" not in df.columns:
        return df

    out = df.copy()
    area = out["area_px"].astype(float).replace(0, np.nan)
    vein_area = out["vein_area_px"].astype(float)
    vein_length = out["vein_length_px"].astype(float)
    valid_vein = vein_length.notna() & vein_area.notna()

    area_fraction = pd.Series(np.nan, index=out.index, dtype=float)
    length_norm = pd.Series(np.nan, index=out.index, dtype=float)
    rel_change = pd.Series(np.nan, index=out.index, dtype=float)
    stable_change = pd.Series(np.nan, index=out.index, dtype=float)
    trusted = pd.Series(False, index=out.index, dtype=bool)

    if valid_vein.any():
        valid_index = out.index[valid_vein]
        valid_area = area.loc[valid_index]
        valid_vein_area = vein_area.loc[valid_index]
        valid_vein_length = vein_length.loc[valid_index]

        valid_area_fraction = valid_vein_area / valid_area
        valid_length_norm = valid_vein_length / np.sqrt(valid_area)
        valid_rel_change = valid_vein_length.pct_change(fill_method=None).abs()
        valid_stable_change = valid_rel_change.rolling(
            int(stable_window), min_periods=1
        ).median()

        base_ok_valid = (
            (out.loc[valid_index, "frame"].astype(int) >= int(warmup_frames))
            & (valid_vein_area >= float(min_vein_area_px))
            & (valid_vein_length >= float(min_vein_length_px))
            & (valid_length_norm >= float(min_length_norm))
            & (valid_area_fraction <= float(max_area_fraction))
            & ((valid_stable_change <= float(max_rel_change)) | valid_stable_change.isna())
        )

        run_lengths = []
        run = 0
        for ok in base_ok_valid.fillna(False).to_numpy(dtype=bool):
            run = run + 1 if ok else 0
            run_lengths.append(run)
        trusted_valid = base_ok_valid & (
            pd.Series(run_lengths, index=valid_index) >= int(require_consecutive)
        )

        area_fraction.loc[valid_index] = valid_area_fraction
        length_norm.loc[valid_index] = valid_length_norm
        rel_change.loc[valid_index] = valid_rel_change
        stable_change.loc[valid_index] = valid_stable_change
        trusted.loc[valid_index] = trusted_valid.astype(bool)

    out["vein_area_fraction"] = area_fraction
    out["vein_length_norm"] = length_norm
    out["vein_rel_change"] = rel_change
    out["vein_stable_rel_change"] = stable_change
    out["vein_trusted"] = trusted.astype(bool)
    out["vein_low_confidence"] = valid_vein & ~out["vein_trusted"]

    for col in ["vein_area_px", "vein_length_px", "vein_length", "vein_endpoints", "vein_branchpoints"]:
        trusted_col = f"{col}_trusted"
        out[trusted_col] = out[col].where(out["vein_trusted"], np.nan)

    return out


def apply_body_mask_confidence(df):
    """Add trusted shape and body-skeleton columns from body-mask quality."""
    if df is None or df.empty or "body_mask_confidence" not in df.columns:
        return df
    out = df.copy()
    confidence = out["body_mask_confidence"].astype(str)
    shape_trusted = confidence.isin(["high", "medium"])
    skeleton_trusted = confidence == "high"
    out["shape_metrics_trusted"] = shape_trusted
    out["body_skeleton_trusted"] = skeleton_trusted
    for col in ["perimeter_px", "perimeter", "circularity", "solidity"]:
        if col in out.columns:
            out[f"{col}_trusted"] = out[col].where(shape_trusted, np.nan)
    for col in [
        "skeleton_length_px", "skeleton_length", "n_branchpoints", "n_endpoints"
    ]:
        if col in out.columns:
            out[f"{col}_trusted"] = out[col].where(skeleton_trusted, np.nan)
    return out

def add_body_area_derivatives(df, dt):
    """Add comparable area-rate columns for occupied, clean, and envelope masks."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if dt is None or dt <= 0:
        for name in ["occupied", "clean_body", "envelope"]:
            out[f"{name}_dA_dt"] = np.nan
        return out
    for name, area_col in [
        ("occupied", "occupied_area"),
        ("clean_body", "clean_body_area"),
        ("envelope", "envelope_area"),
    ]:
        if area_col in out.columns:
            out[f"{name}_dA_dt"] = pd.to_numeric(
                out[area_col], errors="coerce"
            ).diff() / float(dt)
    return out

def expansion_retraction_metrics(prev_mask, curr_mask, dt, pixel_size=1.0, compute_front_speed=True):
    """
    Compute exploration and retraction metrics between consecutive masks.
    """
    curr_mask = curr_mask.astype(bool)
    if prev_mask is None:
        empty = np.zeros_like(curr_mask, dtype=bool)
        return {
            "new_area_px": 0,
            "retracted_area_px": 0,
            "explore_ratio": np.nan,
            "retreat_ratio": np.nan,
            "net_growth_px": 0,
            "front_speed_mean": np.nan,
            "front_speed_p95": np.nan,
            "new_region": empty,
            "retracted_region": empty,
        }

    prev_mask = prev_mask.astype(bool)
    new_region = curr_mask & (~prev_mask)
    retracted_region = prev_mask & (~curr_mask)

    new_area_px = int(np.sum(new_region))
    retracted_area_px = int(np.sum(retracted_region))
    curr_area_px = int(np.sum(curr_mask))
    prev_area_px = int(np.sum(prev_mask))

    explore_ratio = float(new_area_px / curr_area_px) if curr_area_px > 0 else np.nan
    retreat_ratio = float(retracted_area_px / prev_area_px) if prev_area_px > 0 else np.nan
    net_growth_px = int(curr_area_px - prev_area_px)

    if (
        not compute_front_speed
        or dt is None
        or dt <= 0
        or prev_area_px == 0
        or new_area_px == 0
    ):
        front_speed_mean = np.nan
        front_speed_p95 = np.nan
    else:
        distance_to_prev = ndi.distance_transform_edt(~prev_mask)
        front_distances = distance_to_prev[new_region] * pixel_size
        front_speeds = front_distances / float(dt)
        front_speed_mean = float(np.nanmean(front_speeds)) if front_speeds.size else np.nan
        front_speed_p95 = float(np.nanpercentile(front_speeds, 95)) if front_speeds.size else np.nan

    return {
        "new_area_px": new_area_px,
        "retracted_area_px": retracted_area_px,
        "explore_ratio": explore_ratio,
        "retreat_ratio": retreat_ratio,
        "net_growth_px": net_growth_px,
        "front_speed_mean": front_speed_mean,
        "front_speed_p95": front_speed_p95,
        "new_region": new_region,
        "retracted_region": retracted_region,
    }


def roi_mean_intensity(img, roi):
    """Return mean intensity inside ROI formatted as x, y, w, h."""
    if roi is None:
        return np.nan

    gray = _as_gray_uint8(img)
    x, y, w, h = [int(v) for v in roi]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(gray.shape[1], x0 + max(0, w))
    y1 = min(gray.shape[0], y0 + max(0, h))
    if x1 <= x0 or y1 <= y0:
        return np.nan
    return float(np.mean(gray[y0:y1, x0:x1]))


def _roi_slices(shape, roi):
    """Convert x, y, w, h ROI to safe image slices."""
    if roi is None:
        return slice(0, shape[0]), slice(0, shape[1])

    x, y, w, h = [int(v) for v in roi]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(shape[1], x0 + max(0, w))
    y1 = min(shape[0], y0 + max(0, h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid analysis ROI for image shape {shape}: {roi}")
    return slice(y0, y1), slice(x0, x1)


def mask_qc_metrics(mask, analysis_roi=None):
    """Compute per-frame mask coverage and ROI-border contact."""
    mask = mask.astype(bool)
    ys, xs = _roi_slices(mask.shape, analysis_roi)
    roi_mask = mask[ys, xs]

    roi_area = roi_mask.size
    fraction = float(np.sum(roi_mask) / roi_area) if roi_area > 0 else np.nan
    touches = False
    if roi_mask.size > 0 and np.any(roi_mask):
        touches = bool(
            np.any(roi_mask[0, :])
            or np.any(roi_mask[-1, :])
            or np.any(roi_mask[:, 0])
            or np.any(roi_mask[:, -1])
        )

    return {
        "mask_area_fraction": fraction,
        "touch_roi_border": touches,
    }


def circular_roi_mask(shape, center, radius):
    """Create a circular ROI mask for dish-inner analysis."""
    if center is None or radius is None:
        return np.ones(shape[:2], dtype=bool)

    cx, cy = [int(v) for v in center]
    radius = int(radius)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def body_mask_layer_metrics(occupied_mask, clean_mask, envelope_mask, pixel_size=1.0):
    """Summarize the three body-mask layers used by HSV segmentation."""
    occupied_mask = occupied_mask.astype(bool)
    clean_mask = clean_mask.astype(bool)
    envelope_mask = envelope_mask.astype(bool)
    occupied_reg = region_metrics(occupied_mask, pixel_size=pixel_size)
    envelope_reg = region_metrics(envelope_mask, pixel_size=pixel_size)
    occupied_area_px = int(np.sum(occupied_mask))
    clean_area_px = int(np.sum(clean_mask))
    envelope_area_px = int(np.sum(envelope_mask))
    filled_small_hole_area_px = max(0, clean_area_px - occupied_area_px)
    remaining_hole_area_px = max(0, envelope_area_px - clean_area_px)
    occupied_hole_fraction = (
        float((envelope_area_px - occupied_area_px) / envelope_area_px)
        if envelope_area_px > 0 else np.nan
    )
    remaining_hole_fraction = (
        float(remaining_hole_area_px / envelope_area_px)
        if envelope_area_px > 0 else np.nan
    )
    mask_confidence = (
        "high" if np.isfinite(remaining_hole_fraction) and remaining_hole_fraction <= 0.05
        else "medium" if np.isfinite(remaining_hole_fraction) and remaining_hole_fraction <= 0.15
        else "low"
    )
    return {
        "analysis_mask_type": "clean_body",
        "occupied_area_px": occupied_area_px,
        "occupied_area": float(occupied_area_px * pixel_size**2),
        "occupied_perimeter_px": occupied_reg["perimeter_px"],
        "occupied_perimeter": occupied_reg["perimeter"],
        "occupied_circularity": occupied_reg["circularity"],
        "occupied_solidity": occupied_reg["solidity"],
        "clean_body_area_px": clean_area_px,
        "clean_body_area": float(clean_area_px * pixel_size**2),
        "envelope_area_px": envelope_area_px,
        "envelope_area": float(envelope_area_px * pixel_size**2),
        "envelope_perimeter_px": envelope_reg["perimeter_px"],
        "envelope_perimeter": envelope_reg["perimeter"],
        "envelope_circularity": envelope_reg["circularity"],
        "envelope_solidity": envelope_reg["solidity"],
        "filled_small_hole_area_px": filled_small_hole_area_px,
        "remaining_hole_area_px": remaining_hole_area_px,
        "occupied_hole_fraction": occupied_hole_fraction,
        "remaining_hole_fraction": remaining_hole_fraction,
        "body_mask_confidence": mask_confidence,
    }

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
    use_relaxed_body_mask=False,  # HSV-only: disable relaxed HSV extension
    relaxed_h_min=14,
    relaxed_h_max=55,
    relaxed_s_min=25,
    relaxed_v_min=35,
    relaxed_connect_radius=30,
    relaxed_min_obj_size=50,
    rim_exclusion_px=40,
):
    """
    Segment yellow plasmodium using HSV inside a dish circle.

    Main output masks:
    - occupied_mask: final visible plasmodium network mask.
      It is the high-confidence HSV seed plus optional connected extension pixels.
    - clean_mask: occupied_mask with optional conservative small-hole filling.
    - envelope_mask: filled outer envelope of occupied_mask, used as coverage reference.

    The high-confidence seed and optional extension masks are kept for debug/QC.
    """
    if frame.ndim == 2:
        bgr = cv2.cvtColor(_as_gray_uint8(frame), cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    dish_mask = circular_roi_mask(frame.shape[:2], dish_center, dish_radius)

    seed_mask = (
        (hue >= int(h_min))
        & (hue <= int(h_max))
        & (sat >= int(s_min))
        & (val >= int(v_min))
        & dish_mask
    )

    if morph_open_size and morph_open_size > 0:
        seed_mask = morphology.binary_opening(seed_mask, morphology.disk(int(morph_open_size)))

    if morph_close_size and morph_close_size > 0:
        seed_mask = morphology.binary_closing(seed_mask, morphology.disk(int(morph_close_size)))

    if min_obj_size and min_obj_size > 0:
        seed_mask = morphology.remove_small_objects(seed_mask.astype(bool), min_size=int(min_obj_size))

    if min_hole_size and min_hole_size > 0:
        seed_mask = morphology.remove_small_holes(seed_mask.astype(bool), area_threshold=int(min_hole_size))

    if keep_largest:
        seed_mask = keep_largest_component(seed_mask)

    yellow_chroma_candidate = np.zeros_like(seed_mask, dtype=bool)
    yellow_chroma_added_mask = np.zeros_like(seed_mask, dtype=bool)
    if use_yellow_chroma_extension and np.any(seed_mask):
        blue, green, red = cv2.split(bgr.astype(np.int16))
        yellow_chroma = np.minimum(red - blue, green - blue)
        max_channel = np.maximum(np.maximum(red, green), blue)
        yellow_chroma_candidate = (
            (yellow_chroma >= int(yellow_chroma_min))
            & (red >= int(yellow_chroma_min_channel))
            & (green >= int(yellow_chroma_min_channel))
            & (max_channel <= int(yellow_chroma_max_value))
            & dish_mask
        )
        if (
            yellow_chroma_rim_exclusion_px is not None
            and int(yellow_chroma_rim_exclusion_px) > 0
            and dish_radius is not None
        ):
            inner_radius = max(1, int(dish_radius) - int(yellow_chroma_rim_exclusion_px))
            yellow_chroma_candidate &= circular_roi_mask(
                frame.shape[:2], dish_center, inner_radius
            )



        connected_yellow = ndi.binary_propagation(
            seed_mask,
            structure=np.ones((3, 3), dtype=bool),
            mask=(seed_mask | yellow_chroma_candidate),
        )
        connect_radius = max(0, int(yellow_chroma_connect_radius))
        connect_iterations = max(0, int(yellow_chroma_connect_iterations))
        if connect_radius > 0 and connect_iterations > 0:
            for _ in range(connect_iterations):
                nearby_mask = morphology.binary_dilation(
                    connected_yellow,
                    morphology.disk(connect_radius),
                )
                nearby_components = keep_components_touching_mask(
                    yellow_chroma_candidate & (~connected_yellow),
                    nearby_mask,
                )
                expanded_yellow = connected_yellow | nearby_components
                if np.array_equal(expanded_yellow, connected_yellow):
                    break
                connected_yellow = expanded_yellow
        yellow_chroma_added_mask = connected_yellow & (~seed_mask)
    relaxed_candidate = np.zeros_like(seed_mask, dtype=bool)
    relaxed_added_mask = np.zeros_like(seed_mask, dtype=bool)
    seed_near_mask = np.zeros_like(seed_mask, dtype=bool)

    if use_relaxed_body_mask and np.any(seed_mask):
        relaxed_candidate = (
            (hue >= int(relaxed_h_min))
            & (hue <= int(relaxed_h_max))
            & (sat >= int(relaxed_s_min))
            & (val >= int(relaxed_v_min))
            & dish_mask
        )

        if rim_exclusion_px is not None and int(rim_exclusion_px) > 0 and dish_radius is not None:
            inner_radius = max(1, int(dish_radius) - int(rim_exclusion_px))
            inner_dish_mask = circular_roi_mask(frame.shape[:2], dish_center, inner_radius)
            relaxed_candidate &= inner_dish_mask

        radius = max(1, int(relaxed_connect_radius))
        seed_near_mask = morphology.binary_dilation(seed_mask, morphology.disk(radius))
        relaxed_candidate &= seed_near_mask

        if relaxed_min_obj_size and int(relaxed_min_obj_size) > 0:
            relaxed_candidate = morphology.remove_small_objects(
                relaxed_candidate.astype(bool),
                min_size=int(relaxed_min_obj_size),
            )

        relaxed_added_mask = keep_components_touching_mask(
            relaxed_candidate,
            seed_near_mask,
        )
        relaxed_added_mask = relaxed_added_mask & (~seed_mask)

    occupied_mask = (
        seed_mask | yellow_chroma_added_mask | relaxed_added_mask
    ).astype(bool)

    if keep_largest:
        occupied_mask = keep_largest_component(occupied_mask)
        seed_mask = seed_mask & occupied_mask
        yellow_chroma_added_mask = yellow_chroma_added_mask & occupied_mask
        yellow_chroma_candidate = yellow_chroma_candidate & occupied_mask
        relaxed_added_mask = relaxed_added_mask & occupied_mask
        relaxed_candidate = relaxed_candidate & occupied_mask

    envelope_mask = ndi.binary_fill_holes(occupied_mask).astype(bool)
    clean_mask = occupied_mask.copy()
    body_hole_threshold_px = 0
    if fill_small_body_holes and np.any(occupied_mask):
        raw_threshold = max(
            int(body_min_hole_size),
            int(round(np.sum(occupied_mask) * float(body_hole_area_ratio))),
        )
        if body_max_hole_size is not None and int(body_max_hole_size) > 0:
            body_hole_threshold_px = min(int(body_max_hole_size), raw_threshold)
        else:
            body_hole_threshold_px = raw_threshold
        clean_mask = morphology.remove_small_holes(
            occupied_mask,
            area_threshold=int(body_hole_threshold_px),
        ).astype(bool)

    seed_area_px = int(np.sum(seed_mask))
    yellow_chroma_added_area_px = int(np.sum(yellow_chroma_added_mask))
    relaxed_added_area_px = int(np.sum(relaxed_added_mask))
    occupied_area_px = int(np.sum(occupied_mask))
    yellow_chroma_added_fraction = float(
        yellow_chroma_added_area_px / max(occupied_area_px, 1)
    )
    relaxed_added_fraction = float(relaxed_added_area_px / max(occupied_area_px, 1))

    debug = {
        "gray": _as_gray_uint8(frame),
        "background": np.zeros(frame.shape[:2], dtype=np.uint8),
        "corrected": sat,
        "enhanced": val,
        "analysis_roi": None,
        "dish_center": dish_center,
        "dish_radius": dish_radius,
        "seed_mask": seed_mask,
        "yellow_chroma_candidate_mask": yellow_chroma_candidate,
        "yellow_chroma_added_mask": yellow_chroma_added_mask,
        "relaxed_candidate_mask": relaxed_candidate,
        "relaxed_added_mask": relaxed_added_mask,
        "occupied_mask": occupied_mask,
        "clean_mask": clean_mask,
        "envelope_mask": envelope_mask,
        "body_hole_threshold_px": int(body_hole_threshold_px),
        "body_hole_area_ratio": float(body_hole_area_ratio),
        "body_min_hole_size": int(body_min_hole_size),
        "body_max_hole_size": int(body_max_hole_size) if body_max_hole_size is not None else 0,
        "seed_area_px": seed_area_px,
        "yellow_chroma_added_area_px": yellow_chroma_added_area_px,
        "yellow_chroma_added_fraction": yellow_chroma_added_fraction,
        "use_yellow_chroma_extension": bool(use_yellow_chroma_extension),
        "yellow_chroma_min": int(yellow_chroma_min),
        "yellow_chroma_max_value": int(yellow_chroma_max_value),
        "relaxed_added_area_px": relaxed_added_area_px,
        "relaxed_added_fraction": relaxed_added_fraction,
        "use_relaxed_body_mask": bool(use_relaxed_body_mask),
        "relaxed_s_min": int(relaxed_s_min),
    }
    return val, clean_mask, debug

def _body_mask_layers(mask, pre_debug, pixel_size=1.0):
    """Return occupied, clean, envelope masks and their audit metrics."""
    occupied_mask = pre_debug.get("occupied_mask", mask).astype(bool)
    clean_mask = pre_debug.get("clean_mask", mask).astype(bool)
    envelope_mask = pre_debug.get("envelope_mask", clean_mask).astype(bool)
    metrics = body_mask_layer_metrics(
        occupied_mask,
        clean_mask,
        envelope_mask,
        pixel_size=pixel_size,
    )
    metrics["body_hole_threshold_px"] = int(
        pre_debug.get("body_hole_threshold_px", 0)
    )
    metrics["seed_area_px"] = int(pre_debug.get("seed_area_px", np.sum(occupied_mask)))
    metrics["yellow_chroma_added_area_px"] = int(
        pre_debug.get("yellow_chroma_added_area_px", 0)
    )
    metrics["yellow_chroma_added_fraction"] = float(
        pre_debug.get("yellow_chroma_added_fraction", 0.0)
    )
    metrics["use_yellow_chroma_extension"] = bool(
        pre_debug.get("use_yellow_chroma_extension", False)
    )
    metrics["relaxed_added_area_px"] = int(pre_debug.get("relaxed_added_area_px", 0))
    metrics["relaxed_added_fraction"] = float(
        pre_debug.get("relaxed_added_fraction", 0.0)
    )
    metrics["use_relaxed_body_mask"] = bool(
        pre_debug.get("use_relaxed_body_mask", False)
    )
    metrics["body_max_hole_size"] = int(pre_debug.get("body_max_hole_size", 0))
    return occupied_mask, clean_mask, envelope_mask, metrics

def preprocess_and_segment_frame(
    gray,
    preprocess_params=None,
    segment_params=None,
    analysis_roi=None,
):
    """
    Preprocess and segment one frame, optionally only inside analysis_roi.

    analysis_roi uses x, y, w, h. This is important when the dish rim, glare,
    labels, or hardware have stronger contrast than the plasmodium.
    """
    gray = _as_gray_uint8(gray)
    preprocess_params = preprocess_params or {}
    segment_params = segment_params or {}

    ys, xs = _roi_slices(gray.shape, analysis_roi)
    gray_roi = gray[ys, xs]
    enhanced_roi, pre_debug_roi = preprocess_frame(
        gray_roi, return_debug=True, **preprocess_params
    )
    mask_roi = segment_plasmodium(enhanced_roi, **segment_params)

    enhanced_full = np.zeros_like(gray, dtype=np.uint8)
    mask_full = np.zeros_like(gray, dtype=bool)
    background_full = np.zeros_like(gray, dtype=np.uint8)
    corrected_full = np.zeros_like(gray, dtype=np.uint8)

    enhanced_full[ys, xs] = enhanced_roi
    mask_full[ys, xs] = mask_roi
    background_full[ys, xs] = pre_debug_roi["background"]
    corrected_full[ys, xs] = pre_debug_roi["corrected"]

    pre_debug = {
        "gray": gray,
        "background": background_full,
        "corrected": corrected_full,
        "enhanced": enhanced_full,
        "analysis_roi": analysis_roi,
    }
    return enhanced_full, mask_full, pre_debug


def analyze_oscillation(signal, dt):
    """
    Analyze detrended ROI intensity oscillation with FFT and peak counting.
    """
    sig = np.asarray(signal, dtype=np.float64)
    valid = np.isfinite(sig)
    if dt is None or dt <= 0 or np.sum(valid) < 4:
        return {
            "dominant_freq": np.nan,
            "period": np.nan,
            "n_peaks": 0,
            "detrended_signal": np.array([], dtype=np.float64),
            "fft_freq": np.array([], dtype=np.float64),
            "fft_power": np.array([], dtype=np.float64),
        }

    x = np.arange(sig.size)
    sig_interp = sig.copy()
    sig_interp[~valid] = np.interp(x[~valid], x[valid], sig[valid])
    sig_detrended = detrend(sig_interp)

    fft_vals = np.fft.rfft(sig_detrended)
    freqs = np.fft.rfftfreq(sig_detrended.size, d=float(dt))
    power = np.abs(fft_vals) ** 2

    if freqs.size <= 1:
        dominant_freq = np.nan
    else:
        power_for_peak = power.copy()
        power_for_peak[0] = 0
        dominant_freq = float(freqs[int(np.argmax(power_for_peak))])

    period = float(1.0 / dominant_freq) if np.isfinite(dominant_freq) and dominant_freq > 0 else np.nan

    prominence = np.nanstd(sig_detrended) * 0.5
    if not np.isfinite(prominence) or prominence <= 0:
        peaks, _ = find_peaks(sig_detrended)
    else:
        peaks, _ = find_peaks(sig_detrended, prominence=prominence)

    return {
        "dominant_freq": dominant_freq,
        "period": period,
        "n_peaks": int(len(peaks)),
        "detrended_signal": sig_detrended,
        "fft_freq": freqs,
        "fft_power": power,
    }


def overlay_mask_skeleton(
    gray,
    mask=None,
    skeleton=None,
    endpoint_mask=None,
    branch_mask=None,
    new_region=None,
    retracted_region=None,
    vein_mask=None,
    vein_skeleton=None,
    relaxed_added_mask=None,
    alpha=0.35,
):
    """
    Create RGB overlay.

    Colors:
    - mask: green
    - skeleton: red
    - endpoints: yellow
    - branchpoints: cyan
    - new_region: magenta
    - retracted_region: orange
    - visible vein mask: blue
    - visible vein skeleton: white
    """
    gray = _as_gray_uint8(gray)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

    def blend(binary, color, blend_alpha):
        nonlocal overlay
        if binary is None:
            return
        binary_bool = binary.astype(bool)
        if not np.any(binary_bool):
            return
        color_arr = np.array(color, dtype=np.float32)
        overlay[binary_bool] = (1.0 - blend_alpha) * overlay[binary_bool] + blend_alpha * color_arr

    blend(mask, (0, 255, 0), alpha)
    blend(relaxed_added_mask, (0, 255, 255), 0.70)
    blend(new_region, (255, 0, 255), 0.65)
    blend(retracted_region, (255, 165, 0), 0.65)
    blend(vein_mask, (0, 120, 255), 0.45)

    if skeleton is not None:
        overlay[skeleton.astype(bool)] = (255, 0, 0)
    if vein_skeleton is not None:
        overlay[vein_skeleton.astype(bool)] = (255, 255, 255)
    if endpoint_mask is not None:
        overlay[endpoint_mask.astype(bool)] = (255, 255, 0)
    if branch_mask is not None:
        overlay[branch_mask.astype(bool)] = (0, 255, 255)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _analyze_single_frame_for_parallel(args):
    """
    Analyze one frame without using previous-frame state.

    Dynamic expansion/retraction metrics are added later in frame order.
    """
    (
        i,
        frame,
        name,
        dt,
        pixel_size,
        roi,
        analysis_roi,
        segmentation_profile,
        dish_center,
        dish_radius,
        hsv_params,
        vein_params,
        vein_every_n_frames,
        compute_body_skeleton,
        preprocess_params,
        segment_params,
        debug_mode,
    ) = args
    if isinstance(debug_mode, bool):
        debug_mode = "full" if debug_mode else "none"

    gray = _as_gray_uint8(frame)
    focus = focus_score_laplacian(gray)

    if segmentation_profile == "hsv_circle":
        enhanced, mask, pre_debug = segment_hsv_circle(
            frame,
            dish_center=dish_center,
            dish_radius=dish_radius,
            **hsv_params,
        )
    elif segmentation_profile in {"classic", "grabcut_blob"}:
        enhanced, mask, pre_debug = preprocess_and_segment_frame(
            gray,
            preprocess_params=preprocess_params,
            segment_params=segment_params,
            analysis_roi=analysis_roi,
        )
    else:
        raise ValueError(
            "segmentation_profile must be 'classic', 'grabcut_blob', or 'hsv_circle'."
        )

    occupied_mask, mask, envelope_mask, body_layers = _body_mask_layers(
        mask, pre_debug, pixel_size=pixel_size
    )
    reg = region_metrics(mask, pixel_size=pixel_size)
    mask_qc = mask_qc_metrics(mask, analysis_roi=analysis_roi)
    skel = (
        skeleton_metrics(mask, pixel_size=pixel_size)
        if compute_body_skeleton
        else empty_skeleton_metrics_like(mask)
    )
    compute_vein = (
        segmentation_profile == "hsv_circle"
        and vein_every_n_frames > 0
        and i % vein_every_n_frames == 0
    )
    vein = vein_network_metrics(
        frame,
        dish_center=dish_center,
        dish_radius=dish_radius,
        body_mask=occupied_mask,
        pixel_size=pixel_size,
        **vein_params,
    ) if compute_vein else None

    row = {
        "frame": i,
        "name": name,
        "time": float(i * dt) if dt is not None else np.nan,
        "focus_score": focus,
        **reg,
        **mask_qc,
        **body_layers,
        "n_endpoints": skel["n_endpoints"],
        "n_branchpoints": skel["n_branchpoints"],
        "skeleton_length_px": skel["skeleton_length_px"],
        "skeleton_length": skel["skeleton_length"],
    }

    if vein is not None:
        row.update(
            {
                "vein_area_px": vein["vein_area_px"],
                "vein_length_px": vein["vein_length_px"],
                "vein_length": vein["vein_length"],
                "vein_endpoints": vein["vein_endpoints"],
                "vein_branchpoints": vein["vein_branchpoints"],
            }
        )

    if roi is not None:
        row["roi_mean_intensity"] = roi_mean_intensity(gray, roi)

    debug_item = None
    if debug_mode != "none":
        debug_item = {
            "frame": i,
            "name": name,
            "gray": gray,
            "enhanced": enhanced,
            "mask": mask,
            "occupied_mask": occupied_mask,
            "seed_mask": pre_debug.get("seed_mask"),
            "yellow_chroma_added_mask": pre_debug.get("yellow_chroma_added_mask"),
            "relaxed_added_mask": pre_debug.get("relaxed_added_mask"),
            "envelope_mask": envelope_mask,
            "skeleton": skel["skeleton"],
            "endpoint_mask": skel["endpoint_mask"],
            "branch_mask": skel["branch_mask"],
            "vein_mask": vein["vein_mask"] if vein is not None else None,
            "vein_skeleton": vein["vein_skeleton"] if vein is not None else None,
        }
        if debug_mode == "full":
            debug_item["background"] = pre_debug["background"]
            debug_item["corrected"] = pre_debug["corrected"]

    return i, row, debug_item, mask, reg["area"]


def _finish_rows_with_temporal_metrics(
    frame_results,
    dt,
    pixel_size,
    compute_front_speed,
):
    """Add previous-frame metrics to already ordered per-frame results."""
    rows = []
    debug_data = []
    prev_mask = None
    prev_area = np.nan

    for i, row, debug_item, mask, area in frame_results:
        dyn = expansion_retraction_metrics(
            prev_mask,
            mask,
            dt=dt,
            pixel_size=pixel_size,
            compute_front_speed=compute_front_speed,
        )

        if i == 0 or not np.isfinite(prev_area) or dt is None or dt <= 0:
            dA_dt = np.nan
        else:
            dA_dt = float((area - prev_area) / dt)

        row.update(
            {
                "dA_dt": dA_dt,
                "new_area_px": dyn["new_area_px"],
                "retracted_area_px": dyn["retracted_area_px"],
                "explore_ratio": dyn["explore_ratio"],
                "retreat_ratio": dyn["retreat_ratio"],
                "net_growth_px": dyn["net_growth_px"],
                "front_speed_mean": dyn["front_speed_mean"],
                "front_speed_p95": dyn["front_speed_p95"],
            }
        )
        rows.append(row)

        if debug_item is not None:
            debug_item["new_region"] = dyn["new_region"]
            debug_item["retracted_region"] = dyn["retracted_region"]
            debug_data.append(debug_item)

        prev_mask = mask
        prev_area = area

    return rows, debug_data


def analyze_timelapse(
    frames,
    names=None,
    dt=30.0,
    pixel_size=1.0,
    roi=None,
    analysis_roi=None,
    segmentation_profile="classic",
    dish_center=None,
    dish_radius=None,
    hsv_params=None,
    vein_params=None,
    vein_every_n_frames=1,
    compute_body_skeleton=True,
    compute_front_speed=True,
    preprocess_params=None,
    segment_params=None,
    keep_debug=True,
    n_jobs=1,
):
    """
    Main time-lapse analysis function.

    Performs preprocessing, segmentation, morphology analysis, skeleton analysis,
    dynamic expansion/retraction analysis, and optional ROI oscillation analysis.

    Returns df, debug_data, osc.
    """
    if frames is None or len(frames) == 0:
        raise ValueError("No frames were provided for analysis.")

    if names is None:
        names = [f"frame_{i:06d}" for i in range(len(frames))]

    preprocess_params = preprocess_params or {}
    segment_params = segment_params or {}
    hsv_params = hsv_params or {}
    vein_params = vein_params or {}
    vein_every_n_frames = int(vein_every_n_frames or 0)
    n_jobs = int(n_jobs or 1)

    if n_jobs > 1:
        work_items = [
            (
                i,
                frame,
                names[i] if i < len(names) else f"frame_{i:06d}",
                dt,
                pixel_size,
                roi,
                analysis_roi,
                segmentation_profile,
                dish_center,
                dish_radius,
                hsv_params,
                vein_params,
                vein_every_n_frames,
                compute_body_skeleton,
                preprocess_params,
                segment_params,
                keep_debug,
            )
            for i, frame in enumerate(frames)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            frame_results = list(
                tqdm(
                    executor.map(_analyze_single_frame_for_parallel, work_items),
                    total=len(work_items),
                    desc=f"Analyzing frames with {n_jobs} CPU workers",
                )
            )

        frame_results.sort(key=lambda item: item[0])
        rows, debug_data = _finish_rows_with_temporal_metrics(
            frame_results,
            dt=dt,
            pixel_size=pixel_size,
            compute_front_speed=compute_front_speed,
        )

        df = pd.DataFrame(rows)
        df = apply_vein_confidence(df)
        df = apply_body_mask_confidence(df)
        df = add_body_area_derivatives(df, dt)
        if roi is not None and "roi_mean_intensity" in df.columns:
            osc = analyze_oscillation(df["roi_mean_intensity"].to_numpy(), dt)
        else:
            osc = {"dominant_freq": np.nan, "period": np.nan, "n_peaks": 0}

        return df, debug_data, osc

    rows = []
    debug_data = []
    prev_mask = None
    prev_area = np.nan

    for i, frame in enumerate(tqdm(frames, desc="Analyzing frames")):
        gray = _as_gray_uint8(frame)
        focus = focus_score_laplacian(gray)

        if segmentation_profile == "hsv_circle":
            enhanced, mask, pre_debug = segment_hsv_circle(
                frame,
                dish_center=dish_center,
                dish_radius=dish_radius,
                **hsv_params,
            )
        elif segmentation_profile in {"classic", "grabcut_blob"}:
            enhanced, mask, pre_debug = preprocess_and_segment_frame(
                gray,
                preprocess_params=preprocess_params,
                segment_params=segment_params,
                analysis_roi=analysis_roi,
            )
        else:
            raise ValueError(
                "segmentation_profile must be 'classic', 'grabcut_blob', or 'hsv_circle'."
            )

        occupied_mask, mask, envelope_mask, body_layers = _body_mask_layers(
            mask, pre_debug, pixel_size=pixel_size
        )
        reg = region_metrics(mask, pixel_size=pixel_size)
        mask_qc = mask_qc_metrics(mask, analysis_roi=analysis_roi)
        skel = (
            skeleton_metrics(mask, pixel_size=pixel_size)
            if compute_body_skeleton
            else empty_skeleton_metrics_like(mask)
        )
        compute_vein = (
            segmentation_profile == "hsv_circle"
            and vein_every_n_frames > 0
            and i % vein_every_n_frames == 0
        )
        vein = vein_network_metrics(
            frame,
            dish_center=dish_center,
            dish_radius=dish_radius,
            body_mask=occupied_mask,
            pixel_size=pixel_size,
            **vein_params,
        ) if compute_vein else None
        dyn = expansion_retraction_metrics(
            prev_mask,
            mask,
            dt=dt,
            pixel_size=pixel_size,
            compute_front_speed=compute_front_speed,
        )

        area = reg["area"]
        if i == 0 or not np.isfinite(prev_area) or dt is None or dt <= 0:
            dA_dt = np.nan
        else:
            dA_dt = float((area - prev_area) / dt)

        row = {
            "frame": i,
            "name": names[i] if i < len(names) else f"frame_{i:06d}",
            "time": float(i * dt) if dt is not None else np.nan,
            "focus_score": focus,
            **reg,
            **mask_qc,
            **body_layers,
            "dA_dt": dA_dt,
            "new_area_px": dyn["new_area_px"],
            "retracted_area_px": dyn["retracted_area_px"],
            "explore_ratio": dyn["explore_ratio"],
            "retreat_ratio": dyn["retreat_ratio"],
            "net_growth_px": dyn["net_growth_px"],
            "front_speed_mean": dyn["front_speed_mean"],
            "front_speed_p95": dyn["front_speed_p95"],
            "n_endpoints": skel["n_endpoints"],
            "n_branchpoints": skel["n_branchpoints"],
            "skeleton_length_px": skel["skeleton_length_px"],
            "skeleton_length": skel["skeleton_length"],
        }

        if vein is not None:
            row.update(
                {
                    "vein_area_px": vein["vein_area_px"],
                    "vein_length_px": vein["vein_length_px"],
                    "vein_length": vein["vein_length"],
                    "vein_endpoints": vein["vein_endpoints"],
                    "vein_branchpoints": vein["vein_branchpoints"],
                }
            )

        if roi is not None:
            row["roi_mean_intensity"] = roi_mean_intensity(gray, roi)

        rows.append(row)

        if keep_debug:
            debug_data.append(
                {
                    "frame": i,
                    "name": row["name"],
                    "gray": gray,
                    "background": pre_debug["background"],
                    "corrected": pre_debug["corrected"],
                    "enhanced": enhanced,
                    "mask": mask,
                    "occupied_mask": occupied_mask,
                    "seed_mask": pre_debug.get("seed_mask"),
                    "yellow_chroma_added_mask": pre_debug.get("yellow_chroma_added_mask"),
                    "relaxed_added_mask": pre_debug.get("relaxed_added_mask"),
                    "envelope_mask": envelope_mask,
                    "skeleton": skel["skeleton"],
                    "endpoint_mask": skel["endpoint_mask"],
                    "branch_mask": skel["branch_mask"],
                    "vein_mask": vein["vein_mask"] if vein is not None else None,
                    "vein_skeleton": vein["vein_skeleton"] if vein is not None else None,
                    "new_region": dyn["new_region"],
                    "retracted_region": dyn["retracted_region"],
                }
            )

        prev_mask = mask
        prev_area = area

    df = pd.DataFrame(rows)
    df = apply_vein_confidence(df)
    df = apply_body_mask_confidence(df)
    df = add_body_area_derivatives(df, dt)
    if roi is not None and "roi_mean_intensity" in df.columns:
        osc = analyze_oscillation(df["roi_mean_intensity"].to_numpy(), dt)
    else:
        osc = {"dominant_freq": np.nan, "period": np.nan, "n_peaks": 0}

    return df, debug_data, osc


def analyze_video_stream(
    video_path,
    dt=30.0,
    pixel_size=1.0,
    roi=None,
    max_frames=None,
    debug_frame_indices=None,
    segmentation_profile="hsv_circle",
    dish_center=None,
    dish_radius=None,
    hsv_params=None,
    vein_params=None,
    vein_every_n_frames=1,
    compute_body_skeleton=True,
    compute_front_speed=True,
    preprocess_params=None,
    segment_params=None,
    analysis_roi=None,
    overlay_video_path=None,
    overlay_video_fps=10,
):
    """
    Stream a video frame-by-frame and analyze every frame without loading all
    frames into memory. This is preferred for long color videos.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_to_process = min(total, int(max_frames)) if total > 0 else int(max_frames)
    else:
        total_to_process = total

    debug_frame_indices = set(debug_frame_indices or [])
    hsv_params = hsv_params or {}
    vein_params = vein_params or {}
    vein_every_n_frames = int(vein_every_n_frames or 0)
    preprocess_params = preprocess_params or {}
    segment_params = segment_params or {}

    writer = None
    if overlay_video_path is not None:
        overlay_video_path = Path(overlay_video_path)
        overlay_video_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    debug_data = []
    prev_mask = None
    prev_area = np.nan
    roi_signal = []
    idx = 0

    with tqdm(total=total_to_process if total_to_process > 0 else None, desc="Analyzing video stream") as pbar:
        while True:
            if max_frames is not None and idx >= int(max_frames):
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if segmentation_profile == "hsv_circle":
                frame_for_analysis = frame_bgr
            else:
                frame_for_analysis = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            gray = _as_gray_uint8(frame_for_analysis)
            focus = focus_score_laplacian(gray)

            if segmentation_profile == "hsv_circle":
                enhanced, mask, pre_debug = segment_hsv_circle(
                    frame_for_analysis,
                    dish_center=dish_center,
                    dish_radius=dish_radius,
                    **hsv_params,
                )
            elif segmentation_profile in {"classic", "grabcut_blob"}:
                enhanced, mask, pre_debug = preprocess_and_segment_frame(
                    gray,
                    preprocess_params=preprocess_params,
                    segment_params=segment_params,
                    analysis_roi=analysis_roi,
                )
            else:
                raise ValueError(
                    "segmentation_profile must be 'classic', 'grabcut_blob', or 'hsv_circle'."
                )

            occupied_mask, mask, envelope_mask, body_layers = _body_mask_layers(
                mask, pre_debug, pixel_size=pixel_size
            )
            reg = region_metrics(mask, pixel_size=pixel_size)
            mask_qc = mask_qc_metrics(mask, analysis_roi=analysis_roi)
            skel = (
                skeleton_metrics(mask, pixel_size=pixel_size)
                if compute_body_skeleton
                else empty_skeleton_metrics_like(mask)
            )
            compute_vein = (
                segmentation_profile == "hsv_circle"
                and vein_every_n_frames > 0
                and idx % vein_every_n_frames == 0
            )
            vein = vein_network_metrics(
                frame_for_analysis,
                dish_center=dish_center,
                dish_radius=dish_radius,
                body_mask=occupied_mask,
                pixel_size=pixel_size,
                **vein_params,
            ) if compute_vein else None
            dyn = expansion_retraction_metrics(
                prev_mask,
                mask,
                dt=dt,
                pixel_size=pixel_size,
                compute_front_speed=compute_front_speed,
            )

            area = reg["area"]
            if idx == 0 or not np.isfinite(prev_area) or dt is None or dt <= 0:
                dA_dt = np.nan
            else:
                dA_dt = float((area - prev_area) / dt)

            row = {
                "frame": idx,
                "name": f"frame_{idx:06d}",
                "time": float(idx * dt) if dt is not None else np.nan,
                "focus_score": focus,
                **reg,
                **mask_qc,
                **body_layers,
                "dA_dt": dA_dt,
                "new_area_px": dyn["new_area_px"],
                "retracted_area_px": dyn["retracted_area_px"],
                "explore_ratio": dyn["explore_ratio"],
                "retreat_ratio": dyn["retreat_ratio"],
                "net_growth_px": dyn["net_growth_px"],
                "front_speed_mean": dyn["front_speed_mean"],
                "front_speed_p95": dyn["front_speed_p95"],
                "n_endpoints": skel["n_endpoints"],
                "n_branchpoints": skel["n_branchpoints"],
                "skeleton_length_px": skel["skeleton_length_px"],
                "skeleton_length": skel["skeleton_length"],
            }

            if vein is not None:
                row.update(
                    {
                        "vein_area_px": vein["vein_area_px"],
                        "vein_length_px": vein["vein_length_px"],
                        "vein_length": vein["vein_length"],
                        "vein_endpoints": vein["vein_endpoints"],
                        "vein_branchpoints": vein["vein_branchpoints"],
                    }
                )

            if roi is not None:
                roi_value = roi_mean_intensity(gray, roi)
                row["roi_mean_intensity"] = roi_value
                roi_signal.append(roi_value)

            rows.append(row)

            current_debug = None
            if idx in debug_frame_indices:
                current_debug = {
                    "frame": idx,
                    "name": row["name"],
                    "gray": gray,
                    "background": pre_debug["background"],
                    "corrected": pre_debug["corrected"],
                    "enhanced": enhanced,
                    "mask": mask,
                    "occupied_mask": occupied_mask,
                    "seed_mask": pre_debug.get("seed_mask"),
                    "yellow_chroma_added_mask": pre_debug.get("yellow_chroma_added_mask"),
                    "relaxed_added_mask": pre_debug.get("relaxed_added_mask"),
                    "envelope_mask": envelope_mask,
                    "skeleton": skel["skeleton"],
                    "endpoint_mask": skel["endpoint_mask"],
                    "branch_mask": skel["branch_mask"],
                    "vein_mask": vein["vein_mask"] if vein is not None else None,
                    "vein_skeleton": vein["vein_skeleton"] if vein is not None else None,
                    "new_region": dyn["new_region"],
                    "retracted_region": dyn["retracted_region"],
                }
                debug_data.append(current_debug)

            if overlay_video_path is not None:
                overlay_rgb = overlay_mask_skeleton(
                    gray,
                    mask=mask,
                    skeleton=skel["skeleton"],
                    endpoint_mask=skel["endpoint_mask"],
                    branch_mask=skel["branch_mask"],
                    vein_mask=vein["vein_mask"] if vein is not None else None,
                    vein_skeleton=vein["vein_skeleton"] if vein is not None else None,
                    relaxed_added_mask=pre_debug.get("relaxed_added_mask"),
                    new_region=dyn["new_region"],
                    retracted_region=dyn["retracted_region"],
                )
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                if writer is None:
                    height, width = overlay_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(overlay_video_path),
                        fourcc,
                        float(overlay_video_fps),
                        (width, height),
                        True,
                    )
                    if not writer.isOpened():
                        raise ValueError(f"Could not open video writer for: {overlay_video_path}")
                writer.write(overlay_bgr)

            prev_mask = mask
            prev_area = area
            idx += 1
            pbar.update(1)

    cap.release()
    if writer is not None:
        writer.release()

    if len(rows) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    df = pd.DataFrame(rows)
    df = apply_vein_confidence(df)
    df = apply_body_mask_confidence(df)
    df = add_body_area_derivatives(df, dt)
    if roi is not None and "roi_mean_intensity" in df.columns:
        osc = analyze_oscillation(df["roi_mean_intensity"].to_numpy(), dt)
    else:
        osc = {"dominant_freq": np.nan, "period": np.nan, "n_peaks": 0}

    return df, debug_data, osc


def analyze_video_parallel_batches(
    video_path,
    dt=30.0,
    pixel_size=1.0,
    roi=None,
    max_frames=None,
    debug_frame_indices=None,
    segmentation_profile="hsv_circle",
    dish_center=None,
    dish_radius=None,
    hsv_params=None,
    vein_params=None,
    vein_every_n_frames=1,
    compute_body_skeleton=True,
    compute_front_speed=True,
    preprocess_params=None,
    segment_params=None,
    analysis_roi=None,
    overlay_video_path=None,
    overlay_video_fps=10,
    n_jobs=4,
    batch_frames=24,
):
    """
    Analyze a video in bounded parallel batches.

    Only one batch of source frames and analysis arrays is retained at a time.
    Temporal metrics and overlay output remain ordered across batch boundaries.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    n_jobs = max(1, int(n_jobs or 1))
    batch_frames = max(n_jobs, int(batch_frames or n_jobs))
    debug_frame_indices = set(debug_frame_indices or [])
    hsv_params = hsv_params or {}
    vein_params = vein_params or {}
    preprocess_params = preprocess_params or {}
    segment_params = segment_params or {}
    vein_every_n_frames = int(vein_every_n_frames or 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_to_process = min(total, int(max_frames)) if total > 0 else int(max_frames)
    else:
        total_to_process = total

    writer = None
    if overlay_video_path is not None:
        overlay_video_path = Path(overlay_video_path)
        overlay_video_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    debug_data = []
    prev_mask = None
    prev_area = np.nan
    idx = 0

    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            with tqdm(
                total=total_to_process if total_to_process > 0 else None,
                desc=f"Analyzing video in batches with {n_jobs} CPU workers",
            ) as pbar:
                while True:
                    batch = []
                    while len(batch) < batch_frames:
                        if max_frames is not None and idx + len(batch) >= int(max_frames):
                            break
                        ok, frame_bgr = cap.read()
                        if not ok:
                            break
                        frame_idx = idx + len(batch)
                        frame_for_analysis = (
                            frame_bgr
                            if segmentation_profile == "hsv_circle"
                            else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                        )
                        batch.append((frame_idx, frame_for_analysis))

                    if not batch:
                        break

                    work_items = []
                    for frame_idx, frame in batch:
                        need_debug = (
                            overlay_video_path is not None
                            or frame_idx in debug_frame_indices
                        )
                        debug_mode = "overlay" if need_debug else "none"
                        work_items.append(
                            (
                                frame_idx,
                                frame,
                                f"frame_{frame_idx:06d}",
                                dt,
                                pixel_size,
                                roi,
                                analysis_roi,
                                segmentation_profile,
                                dish_center,
                                dish_radius,
                                hsv_params,
                                vein_params,
                                vein_every_n_frames,
                                compute_body_skeleton,
                                preprocess_params,
                                segment_params,
                                debug_mode,
                            )
                        )
                    frame_results = list(
                        executor.map(_analyze_single_frame_for_parallel, work_items)
                    )
                    frame_results.sort(key=lambda item: item[0])

                    for frame_idx, row, debug_item, mask, area in frame_results:
                        dyn = expansion_retraction_metrics(
                            prev_mask,
                            mask,
                            dt=dt,
                            pixel_size=pixel_size,
                            compute_front_speed=compute_front_speed,
                        )
                        if (
                            frame_idx == 0
                            or not np.isfinite(prev_area)
                            or dt is None
                            or dt <= 0
                        ):
                            dA_dt = np.nan
                        else:
                            dA_dt = float((area - prev_area) / dt)

                        row.update(
                            {
                                "dA_dt": dA_dt,
                                "new_area_px": dyn["new_area_px"],
                                "retracted_area_px": dyn["retracted_area_px"],
                                "explore_ratio": dyn["explore_ratio"],
                                "retreat_ratio": dyn["retreat_ratio"],
                                "net_growth_px": dyn["net_growth_px"],
                                "front_speed_mean": dyn["front_speed_mean"],
                                "front_speed_p95": dyn["front_speed_p95"],
                            }
                        )
                        rows.append(row)

                        if debug_item is not None:
                            debug_item["new_region"] = dyn["new_region"]
                            debug_item["retracted_region"] = dyn["retracted_region"]

                        if overlay_video_path is not None:
                            if debug_item is None:
                                raise RuntimeError(
                                    f"Missing overlay data for frame {frame_idx}."
                                )
                            overlay_rgb = overlay_mask_skeleton(
                                debug_item["gray"],
                                mask=mask,
                                skeleton=debug_item["skeleton"],
                                endpoint_mask=debug_item["endpoint_mask"],
                                branch_mask=debug_item["branch_mask"],
                                vein_mask=debug_item["vein_mask"],
                                vein_skeleton=debug_item["vein_skeleton"],
                                relaxed_added_mask=debug_item.get("relaxed_added_mask"),
                                new_region=dyn["new_region"],
                                retracted_region=dyn["retracted_region"],
                            )
                            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                            if writer is None:
                                height, width = overlay_bgr.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(
                                    str(overlay_video_path),
                                    fourcc,
                                    float(overlay_video_fps),
                                    (width, height),
                                    True,
                                )
                                if not writer.isOpened():
                                    raise ValueError(
                                        f"Could not open video writer for: {overlay_video_path}"
                                    )
                            writer.write(overlay_bgr)

                        if frame_idx in debug_frame_indices and debug_item is not None:
                            debug_data.append(debug_item)

                        prev_mask = mask
                        prev_area = area

                    processed = len(batch)
                    idx += processed
                    pbar.update(processed)
                    del batch, work_items, frame_results
                    gc.collect()

                    if processed < batch_frames:
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not rows:
        raise ValueError(f"No frames could be read from video: {video_path}")

    df = apply_vein_confidence(pd.DataFrame(rows))
    df = apply_body_mask_confidence(df)
    df = add_body_area_derivatives(df, dt)
    if roi is not None and "roi_mean_intensity" in df.columns:
        osc = analyze_oscillation(df["roi_mean_intensity"].to_numpy(), dt)
    else:
        osc = {"dominant_freq": np.nan, "period": np.nan, "n_peaks": 0}

    return df, debug_data, osc


def plot_summary(df, out_prefix=None):
    """
    Plot and optionally save summary time-series:
    - area
    - dA/dt
    - front_speed_mean / front_speed_p95
    - explore_ratio / retreat_ratio
    - circularity / solidity
    - endpoints / branchpoints / skeleton_length
    """
    if df is None or df.empty:
        raise ValueError("df is empty; cannot plot summary.")

    t = df["time"] if "time" in df.columns else df["frame"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(t, df["area"], color="tab:green")
    axes[0].set_title("Area")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Area")

    axes[1].plot(t, df["dA_dt"], color="tab:blue")
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("dA/dt")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Area / time")

    axes[2].plot(t, df["front_speed_mean"], label="mean", color="tab:red")
    axes[2].plot(t, df["front_speed_p95"], label="p95", color="tab:orange")
    axes[2].set_title("Front speed")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Distance / time")
    axes[2].legend()

    axes[3].plot(t, df["explore_ratio"], label="explore", color="tab:purple")
    axes[3].plot(t, df["retreat_ratio"], label="retreat", color="tab:brown")
    axes[3].set_title("Explore / retreat ratio")
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Ratio")
    axes[3].legend()

    axes[4].plot(t, df["circularity"], label="circularity", color="tab:cyan")
    axes[4].plot(t, df["solidity"], label="solidity", color="tab:gray")
    axes[4].set_title("Shape compactness")
    axes[4].set_xlabel("Time")
    axes[4].set_ylabel("Metric")
    axes[4].legend()

    ax_left = axes[5]
    ax_right = ax_left.twinx()
    ax_left.plot(t, df["n_endpoints"], label="endpoints", color="gold")
    ax_left.plot(t, df["n_branchpoints"], label="branchpoints", color="teal")
    ax_right.plot(t, df["skeleton_length"], label="skeleton length", color="crimson")
    ax_left.set_title("Skeleton complexity")
    ax_left.set_xlabel("Time")
    ax_left.set_ylabel("Count")
    ax_right.set_ylabel("Skeleton length")
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")

    if out_prefix is not None:
        out_path = Path(out_prefix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}:
            out_path = out_path.with_suffix(".png")
        fig.savefig(out_path, dpi=200)

        raw_vein_cols = {"vein_length", "vein_endpoints", "vein_branchpoints"}
        trusted_vein_cols = {
            "vein_length_trusted",
            "vein_endpoints_trusted",
            "vein_branchpoints_trusted",
        }
        has_raw_vein = raw_vein_cols.issubset(df.columns) and df["vein_length"].notna().any()
        has_trusted_vein = (
            trusted_vein_cols.issubset(df.columns)
            and df["vein_length_trusted"].notna().any()
        )
        if has_raw_vein or has_trusted_vein:
            vein_fig, vein_axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
            if has_raw_vein:
                vein_axes[0].plot(
                    t,
                    df["vein_length"],
                    color="0.65",
                    marker="o",
                    linestyle="-",
                    markersize=3,
                    label="raw visible vein length",
                )
                vein_axes[1].plot(
                    t,
                    df["vein_endpoints"],
                    color="gold",
                    marker="o",
                    linestyle="none",
                    markersize=3,
                    label="raw vein endpoints",
                )
                vein_axes[1].plot(
                    t,
                    df["vein_branchpoints"],
                    color="teal",
                    marker="o",
                    linestyle="none",
                    markersize=3,
                    label="raw vein branchpoints",
                )
            if has_trusted_vein:
                vein_axes[0].plot(
                    t,
                    df["vein_length_trusted"],
                    color="tab:blue",
                    marker="o",
                    linestyle="-",
                    markersize=3,
                    label="trusted visible vein length",
                )
                vein_axes[1].plot(
                    t,
                    df["vein_endpoints_trusted"],
                    color="orange",
                    marker="o",
                    linestyle="-",
                    markersize=3,
                    label="trusted vein endpoints",
                )
                vein_axes[1].plot(
                    t,
                    df["vein_branchpoints_trusted"],
                    color="tab:green",
                    marker="o",
                    linestyle="-",
                    markersize=3,
                    label="trusted vein branchpoints",
                )
            vein_axes[0].set_title("Visible vein network")
            vein_axes[0].set_xlabel("Time")
            vein_axes[0].set_ylabel("Length")
            vein_axes[0].legend()
            vein_axes[1].set_xlabel("Time")
            vein_axes[1].set_ylabel("Count")
            vein_axes[1].legend()
            vein_fig.savefig(out_path.with_name(out_path.stem + "_veins.png"), dpi=200)

    return fig


def save_debug_examples(
    debug_data,
    out_dir,
    frame_indices,
    use_frame_number=False,
):
    """
    Save gray, prep, mask, and overlay images for selected frames.

    Overlay contains mask, skeleton, endpoints, branchpoints, new region,
    and retracted region.

    By default, frame_indices are positions in debug_data. When
    use_frame_number=True, they are matched against item["frame"] instead.
    """
    if debug_data is None or len(debug_data) == 0:
        warnings.warn("No debug_data available to save.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_to_item = None
    if use_frame_number:
        frame_to_item = {int(item["frame"]): item for item in debug_data}

    for idx in frame_indices:
        idx = int(idx)
        if use_frame_number:
            item = frame_to_item.get(idx)
            if item is None:
                warnings.warn(f"Debug frame number not available: {idx}")
                continue
        else:
            if idx < 0 or idx >= len(debug_data):
                warnings.warn(f"Debug frame index out of range: {idx}")
                continue

            item = debug_data[idx]
        stem = f"frame_{int(item['frame']):06d}"

        gray = _as_gray_uint8(item["gray"])
        prep = _as_gray_uint8(item["enhanced"])
        mask_u8 = (item["mask"].astype(np.uint8) * 255)

        overlay_rgb = overlay_mask_skeleton(
            gray,
            mask=item["mask"],
            skeleton=item["skeleton"],
            endpoint_mask=item["endpoint_mask"],
            branch_mask=item["branch_mask"],
            new_region=item["new_region"],
            retracted_region=item["retracted_region"],
            vein_mask=item.get("vein_mask"),
            vein_skeleton=item.get("vein_skeleton"),
            relaxed_added_mask=item.get("relaxed_added_mask"),
        )
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_dir / f"{stem}_gray.png"), gray)
        cv2.imwrite(str(out_dir / f"{stem}_prep.png"), prep)
        cv2.imwrite(str(out_dir / f"{stem}_enhanced.png"), prep)
        if "background" in item:
            cv2.imwrite(str(out_dir / f"{stem}_background.png"), _as_gray_uint8(item["background"]))
        if "corrected" in item:
            cv2.imwrite(str(out_dir / f"{stem}_corrected.png"), _as_gray_uint8(item["corrected"]))
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask_u8)
        if "occupied_mask" in item:
            cv2.imwrite(
                str(out_dir / f"{stem}_occupied_mask.png"),
                item["occupied_mask"].astype(np.uint8) * 255,
            )
        if item.get("seed_mask") is not None:
            cv2.imwrite(
                str(out_dir / f"{stem}_seed_mask.png"),
                item["seed_mask"].astype(np.uint8) * 255,
            )
        if item.get("yellow_chroma_added_mask") is not None:
            cv2.imwrite(
                str(out_dir / f"{stem}_yellow_chroma_added_mask.png"),
                item["yellow_chroma_added_mask"].astype(np.uint8) * 255,
            )
        if item.get("relaxed_added_mask") is not None:
            cv2.imwrite(
                str(out_dir / f"{stem}_relaxed_added_mask.png"),
                item["relaxed_added_mask"].astype(np.uint8) * 255,
            )
        if "envelope_mask" in item:
            cv2.imwrite(
                str(out_dir / f"{stem}_envelope_mask.png"),
                item["envelope_mask"].astype(np.uint8) * 255,
            )
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay_bgr)


def clean_debug_outputs(out_dir, clean_segment_tests=False):
    """
    Remove stale debug-only output folders from a previous run.

    This intentionally does not remove the main output directory, metrics CSV,
    summary CSV, summary plots, or overlay video.
    """
    out_dir = Path(out_dir)
    folders = ["debug_examples", "qc_review"]
    if clean_segment_tests:
        folders.append("segment_tests")

    for folder in folders:
        path = out_dir / folder
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def save_overlay_video(
    debug_data,
    out_path,
    fps=10,
    draw_frame_text=True,
    output_scale=1.0,
):
    """
    Save an overlay video from debug_data.

    The fps here is only for playback speed of the exported overlay video.
    It does not change biological time calculations, which use FRAME_INTERVAL.

    output_scale controls the exported video resolution:
    - 1.0 = original frame size
    - 0.5 = half width and half height
    - 0.25 = quarter width and quarter height
    Values are clipped to 0.1--1.0 to avoid accidental huge or invalid videos.
    """
    if debug_data is None or len(debug_data) == 0:
        warnings.warn("No debug_data available to save overlay video.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_gray = _as_gray_uint8(debug_data[0]["gray"])
    height, width = first_gray.shape[:2]
    try:
        output_scale = float(output_scale)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid output_scale={output_scale!r}; falling back to 1.0."
        )
        output_scale = 1.0
    if not np.isfinite(output_scale):
        warnings.warn(
            f"Invalid output_scale={output_scale!r}; falling back to 1.0."
        )
        output_scale = 1.0
    output_scale = float(np.clip(output_scale, 0.1, 1.0))
    out_width = max(1, int(round(width * output_scale)))
    out_height = max(1, int(round(height * output_scale)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        float(fps),
        (out_width, out_height),
        True,
    )

    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {out_path}")

    print(
        "Writing overlay video "
        f"at scale {output_scale:.3f}: "
        f"{width}x{height} -> {out_width}x{out_height}"
    )

    try:
        for item in tqdm(debug_data, desc="Writing overlay video"):
            gray = _as_gray_uint8(item["gray"])
            overlay_rgb = overlay_mask_skeleton(
                gray,
                mask=item["mask"],
                skeleton=item["skeleton"],
                endpoint_mask=item["endpoint_mask"],
                branch_mask=item["branch_mask"],
                new_region=item["new_region"],
                retracted_region=item["retracted_region"],
                vein_mask=item.get("vein_mask"),
                vein_skeleton=item.get("vein_skeleton"),
            )
            frame_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            if (out_width, out_height) != (width, height):
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (out_width, out_height),
                    interpolation=cv2.INTER_AREA,
                )

            if draw_frame_text:
                text = f"frame {int(item['frame'])}"
                text_origin = (
                    max(10, int(round(30 * output_scale))),
                    max(20, int(round(50 * output_scale))),
                )
                font_scale = max(0.4, 1.2 * output_scale)
                outer_thickness = max(2, int(round(3 * output_scale)))
                inner_thickness = max(1, int(round(1 * output_scale)))
                cv2.putText(
                    frame_bgr,
                    text,
                    text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    outer_thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_bgr,
                    text,
                    text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    inner_thickness,
                    cv2.LINE_AA,
                )

            writer.write(frame_bgr)
    finally:
        writer.release()


def validate_analysis_qc(
    df,
    debug_data,
    out_dir,
    analysis_roi=None,
    area_jump_threshold=0.35,
    min_area_px=100,
    mask_area_fraction_limits=(0.001, 0.85),
    save_review_images=True,
):
    """
    Produce automatic QC flags for segmentation accuracy review.

    This does not prove biological correctness. It identifies frames that should
    be visually checked: empty masks, abrupt area jumps, low focus, masks that
    touch the ROI border, and implausibly small/large mask fractions.
    """
    if df is None or df.empty:
        raise ValueError("df is empty; cannot run QC validation.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qc = df[["frame", "time", "focus_score", "area_px", "dA_dt"]].copy()
    area = qc["area_px"].astype(float)
    prev_area = area.shift(1)
    qc["area_rel_change"] = (area - prev_area).abs() / prev_area.replace(0, np.nan)
    qc["flag_empty_or_tiny"] = area < float(min_area_px)
    qc["flag_area_jump"] = qc["area_rel_change"] > float(area_jump_threshold)

    focus = qc["focus_score"].astype(float)
    focus_median = np.nanmedian(focus)
    focus_mad = np.nanmedian(np.abs(focus - focus_median))
    if np.isfinite(focus_mad) and focus_mad > 0:
        qc["focus_robust_z"] = 0.6745 * (focus - focus_median) / focus_mad
        qc["flag_low_focus"] = qc["focus_robust_z"] < -3.0
    else:
        qc["focus_robust_z"] = np.nan
        qc["flag_low_focus"] = False

    has_df_mask_fraction = "mask_area_fraction" in df.columns
    has_df_touch_border = "touch_roi_border" in df.columns
    qc["mask_area_fraction"] = (
        pd.to_numeric(df["mask_area_fraction"], errors="coerce")
        if has_df_mask_fraction
        else np.nan
    )
    qc["touch_roi_border"] = (
        df["touch_roi_border"].astype("boolean")
        if has_df_touch_border
        else pd.Series(pd.NA, index=df.index, dtype="boolean")
    )

    if debug_data and (not has_df_mask_fraction or not has_df_touch_border):
        for item in debug_data:
            idx = int(item["frame"])
            fallback_qc = mask_qc_metrics(item["mask"], analysis_roi=analysis_roi)
            frame_match = qc["frame"] == idx
            if not has_df_mask_fraction:
                qc.loc[frame_match, "mask_area_fraction"] = fallback_qc[
                    "mask_area_fraction"
                ]
            if not has_df_touch_border:
                qc.loc[frame_match, "touch_roi_border"] = fallback_qc[
                    "touch_roi_border"
                ]

    lo, hi = mask_area_fraction_limits
    fraction = pd.to_numeric(qc["mask_area_fraction"], errors="coerce")
    qc["flag_area_fraction"] = fraction.notna() & (
        (fraction < float(lo)) | (fraction > float(hi))
    )
    qc["flag_touch_roi_border"] = (
        qc["touch_roi_border"].fillna(False).astype(bool)
    )
    qc["remaining_hole_fraction"] = (
        pd.to_numeric(df["remaining_hole_fraction"], errors="coerce")
        if "remaining_hole_fraction" in df.columns else np.nan
    )
    qc["body_mask_confidence"] = (
        df["body_mask_confidence"].astype(str)
        if "body_mask_confidence" in df.columns else "not_available"
    )
    qc["flag_low_body_mask_confidence"] = (
        qc["body_mask_confidence"].astype(str) == "low"
    )

    flag_cols = [
        "flag_empty_or_tiny",
        "flag_area_jump",
        "flag_low_focus",
        "flag_area_fraction",
        "flag_touch_roi_border",
        "flag_low_body_mask_confidence",
    ]
    qc["needs_review"] = qc[flag_cols].any(axis=1)

    qc_csv = out_dir / "plasmodium_qc_flags.csv"
    qc.to_csv(qc_csv, index=False, encoding="utf-8-sig")

    if save_review_images and debug_data:
        review_dir = out_dir / "qc_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        flagged = qc.loc[qc["needs_review"], "frame"].astype(int).tolist()
        sampled_indices = np.linspace(
            0,
            len(debug_data) - 1,
            min(12, len(debug_data)),
            dtype=int,
        ).tolist()
        sampled_frames = [int(debug_data[idx]["frame"]) for idx in sampled_indices]
        review_frame_numbers = sorted(set(flagged[:40] + sampled_frames))
        save_debug_examples(
            debug_data,
            review_dir,
            review_frame_numbers,
            use_frame_number=True,
        )

    print(f"Saved QC flags CSV: {qc_csv}")
    print(f"QC frames needing review: {int(qc['needs_review'].sum())} / {len(qc)}")
    return qc


def export_manual_validation_package(
    video_path,
    out_dir,
    dish_center,
    dish_radius,
    hsv_params,
    samples_per_section=10,
    section_frames=300,
):
    """Export front/middle/rear images for manual ground-truth annotation."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video frame count is unavailable: {video_path}")

    section_frames = max(1, min(int(section_frames), total_frames))
    starts = {
        "front": 0,
        "middle": max(0, (total_frames - section_frames) // 2),
        "rear": max(0, total_frames - section_frames),
    }
    package_dir = Path(out_dir) / "manual_validation_package"
    source_dir = package_dir / "source_images"
    auto_dir = package_dir / "auto_masks"
    overlay_dir = package_dir / "overlays"
    manual_dir = package_dir / "manual_masks"
    for folder in [source_dir, auto_dir, overlay_dir, manual_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for section, start_frame in starts.items():
        end_frame = min(total_frames - 1, start_frame + section_frames - 1)
        sample_frames = np.linspace(
            start_frame,
            end_frame,
            min(int(samples_per_section), end_frame - start_frame + 1),
            dtype=int,
        )
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                warnings.warn(f"Could not read manual-validation frame {frame_idx}.")
                continue
            _, mask, _ = segment_hsv_circle(
                frame,
                dish_center=dish_center,
                dish_radius=dish_radius,
                **hsv_params,
            )
            mask = mask.astype(bool)
            overlay = frame.copy().astype(np.float32)
            overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array(
                [0, 255, 0], dtype=np.float32
            )
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
            stem = f"frame_{int(frame_idx):06d}"
            source_path = source_dir / f"{stem}_original.jpg"
            auto_path = auto_dir / f"{stem}_mask.png"
            overlay_path = overlay_dir / f"{stem}_overlay.jpg"
            manual_path = manual_dir / f"{stem}_mask.png"
            cv2.imwrite(
                str(source_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            cv2.imwrite(str(auto_path), mask.astype(np.uint8) * 255)
            cv2.imwrite(
                str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            rows.append(
                {
                    "section": section,
                    "frame": int(frame_idx),
                    "video_total_frames": total_frames,
                    "source_image": str(source_path),
                    "auto_mask": str(auto_path),
                    "overlay": str(overlay_path),
                    "expected_manual_mask": str(manual_path),
                    "manual_mask_ready": manual_path.exists(),
                }
            )
    cap.release()

    manifest = pd.DataFrame(rows).sort_values(["section", "frame"])
    manifest_csv = package_dir / "manual_validation_manifest.csv"
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    print(f"Saved manual validation package: {package_dir}")
    print(f"Manual masks must be binary PNG files in: {manual_dir}")
    return manifest


def annotate_manual_validation_package(
    package_dir,
    display_scale=0.5,
    brush_radius=4,
    overlay_alpha=0.23,
    start_image=None,
):
    """Correct draft masks in a fixed Tkinter viewport without built-in pan tools."""
    import tkinter as tk
    from PIL import Image, ImageTk

    package_dir = Path(package_dir)
    manifest_csv = package_dir / "manual_validation_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_csv}")
    manifest = pd.read_csv(manifest_csv)
    display_scale = float(np.clip(display_scale, 0.15, 4.0))
    brush_radius = max(1, int(brush_radius))
    overlay_alpha = float(np.clip(overlay_alpha, 0.0, 0.8))

    if start_image is not None:
        index = int(np.clip(int(start_image) - 1, 0, len(manifest) - 1))
    else:
        index = next(
            (
                i
                for i, row in manifest.iterrows()
                if not Path(row["expected_manual_mask"]).exists()
            ),
            0,
        )
    if index > 0:
        print(f"Resuming manual annotation at {index + 1}/{len(manifest)}.")

    while 0 <= index < len(manifest):
        item = manifest.iloc[index]
        source = cv2.imread(str(item["source_image"]), cv2.IMREAD_COLOR)
        auto = cv2.imread(str(item["auto_mask"]), cv2.IMREAD_GRAYSCALE)
        manual_path = Path(item["expected_manual_mask"])
        existing = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
        draft_path = (
            package_dir
            / "draft_masks"
            / f"frame_{int(item['frame']):06d}_mask.png"
        )
        draft = cv2.imread(str(draft_path), cv2.IMREAD_GRAYSCALE)
        if source is None or auto is None:
            warnings.warn(f"Could not read annotation files for frame {item['frame']}.")
            index += 1
            continue

        initial_mask = existing if existing is not None else draft if draft is not None else auto
        mask = (initial_mask > 0).astype(np.uint8) * 255
        image_height, image_width = mask.shape
        state = {
            "mode": "add",
            "radius": brush_radius,
            "last_point": None,
            "drawing": False,
            "panning": False,
            "pan_start": None,
            "center_start": None,
            "show_overlay": True,
            "action": "quit",
            "scale": display_scale,
            "center_x": image_width / 2.0,
            "center_y": image_height / 2.0,
            "photo": None,
            "render_pending": False,
        }

        root = tk.Tk()
        root.title("Manual mask editor")
        try:
            root.state("zoomed")
        except tk.TclError:
            root.geometry(
                f"{max(960, root.winfo_screenwidth() - 40)}x"
                f"{max(640, root.winfo_screenheight() - 90)}+0+0"
            )

        status_var = tk.StringVar()
        status = tk.Label(
            root,
            textvariable=status_var,
            anchor="w",
            bg="#202020",
            fg="white",
            padx=10,
            pady=6,
            font=("Segoe UI", 9),
        )
        status.pack(fill=tk.X)
        canvas = tk.Canvas(
            root,
            bg="#282828",
            highlightthickness=0,
            cursor="crosshair",
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        image_id = canvas.create_image(0, 0, anchor=tk.NW)
        root.update_idletasks()

        def viewport_size():
            return max(1, canvas.winfo_width()), max(1, canvas.winfo_height())

        def fit_view():
            width, height = viewport_size()
            state["scale"] = max(
                0.05,
                min(width / image_width, height / image_height),
            )
            state["center_x"] = image_width / 2.0
            state["center_y"] = image_height / 2.0
            request_render()

        def clamp_center():
            width, height = viewport_size()
            half_x = width / (2.0 * state["scale"])
            half_y = height / (2.0 * state["scale"])
            if half_x >= image_width / 2.0:
                state["center_x"] = image_width / 2.0
            else:
                state["center_x"] = float(
                    np.clip(state["center_x"], half_x, image_width - half_x)
                )
            if half_y >= image_height / 2.0:
                state["center_y"] = image_height / 2.0
            else:
                state["center_y"] = float(
                    np.clip(state["center_y"], half_y, image_height - half_y)
                )

        def local_coordinates(event):
            return (
                event.x_root - canvas.winfo_rootx(),
                event.y_root - canvas.winfo_rooty(),
            )

        def screen_to_image(x, y):
            width, height = viewport_size()
            image_x = state["center_x"] + (x - width / 2.0) / state["scale"]
            image_y = state["center_y"] + (y - height / 2.0) / state["scale"]
            return (
                int(np.clip(round(image_x), 0, image_width - 1)),
                int(np.clip(round(image_y), 0, image_height - 1)),
            )

        def update_status():
            status_var.set(
                f"{index + 1}/{len(manifest)} {item['section']} "
                f"frame {int(item['frame'])} | {state['mode'].upper()} "
                f"brush={state['radius']} zoom={state['scale']:.2f}x | "
                "Left paint | Middle/right pan | Wheel zoom | A add | E erase | "
                "[ ] brush | F fit | V overlay | R reset | C clear | "
                "S save | P previous | Q quit"
            )

        def render_now():
            state["render_pending"] = False
            width, height = viewport_size()
            preview = source.copy().astype(np.float32)
            selected = mask > 0
            if state["show_overlay"]:
                preview[selected] = (
                    (1.0 - overlay_alpha) * preview[selected]
                    + overlay_alpha * np.array([0, 255, 0], dtype=np.float32)
                )
            preview = np.clip(preview, 0, 255).astype(np.uint8)
            if state["show_overlay"]:
                contours, _ = cv2.findContours(
                    selected.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(preview, contours, -1, (0, 0, 255), 1)
            transform = np.array(
                [
                    [
                        state["scale"],
                        0.0,
                        width / 2.0 - state["center_x"] * state["scale"],
                    ],
                    [
                        0.0,
                        state["scale"],
                        height / 2.0 - state["center_y"] * state["scale"],
                    ],
                ],
                dtype=np.float32,
            )
            rendered = cv2.warpAffine(
                preview,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(40, 40, 40),
            )
            rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            state["photo"] = ImageTk.PhotoImage(
                Image.fromarray(rendered),
                master=root,
            )
            canvas.itemconfigure(image_id, image=state["photo"])
            update_status()

        def request_render():
            if not state["render_pending"] and root.winfo_exists():
                state["render_pending"] = True
                root.after_idle(render_now)

        def paint_to(point, start=False):
            value = 255 if state["mode"] == "add" else 0
            radius = state["radius"]
            if start or state["last_point"] is None:
                cv2.circle(mask, point, radius, value, -1)
            else:
                cv2.line(
                    mask,
                    state["last_point"],
                    point,
                    value,
                    radius * 2,
                    cv2.LINE_8,
                )
                cv2.circle(mask, point, radius, value, -1)
            state["last_point"] = point
            request_render()

        def on_left_press(event):
            state["drawing"] = True
            state["panning"] = False
            state["last_point"] = None
            canvas.focus_set()
            try:
                canvas.grab_set()
            except tk.TclError:
                pass
            x, y = local_coordinates(event)
            paint_to(screen_to_image(x, y), start=True)
            return "break"

        def on_left_release(event):
            if state["drawing"]:
                x, y = local_coordinates(event)
                paint_to(screen_to_image(x, y))
            state["drawing"] = False
            state["last_point"] = None
            try:
                canvas.grab_release()
            except tk.TclError:
                pass
            return "break"

        def on_pan_press(event):
            state["panning"] = True
            state["drawing"] = False
            state["last_point"] = None
            state["pan_start"] = local_coordinates(event)
            state["center_start"] = (state["center_x"], state["center_y"])
            canvas.config(cursor="fleur")
            try:
                canvas.grab_set()
            except tk.TclError:
                pass
            return "break"

        def on_pan_release(_event):
            state["panning"] = False
            canvas.config(cursor="crosshair")
            try:
                canvas.grab_release()
            except tk.TclError:
                pass
            return "break"

        def on_motion(event):
            x, y = local_coordinates(event)
            if state["drawing"]:
                paint_to(screen_to_image(x, y))
            elif state["panning"]:
                dx = (x - state["pan_start"][0]) / state["scale"]
                dy = (y - state["pan_start"][1]) / state["scale"]
                state["center_x"] = state["center_start"][0] - dx
                state["center_y"] = state["center_start"][1] - dy
                clamp_center()
                request_render()
            return "break"

        def zoom_at(x, y, factor):
            anchor_x, anchor_y = screen_to_image(x, y)
            width, height = viewport_size()
            fit_scale = min(width / image_width, height / image_height)
            new_scale = float(
                np.clip(state["scale"] * factor, fit_scale * 0.5, 8.0)
            )
            if np.isclose(new_scale, state["scale"]):
                return
            state["scale"] = new_scale
            state["center_x"] = anchor_x - (x - width / 2.0) / new_scale
            state["center_y"] = anchor_y - (y - height / 2.0) / new_scale
            clamp_center()
            request_render()

        def on_mouse_wheel(event):
            x, y = local_coordinates(event)
            zoom_at(x, y, 1.20 if event.delta > 0 else 1.0 / 1.20)
            return "break"

        def finish(action):
            state["action"] = action
            root.destroy()

        def on_key(event):
            key = event.keysym.lower()
            if key in {"s", "n", "return"}:
                manual_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(manual_path), mask)
                finish("next")
            elif key == "p":
                finish("previous")
            elif key == "a":
                state["mode"] = "add"
                update_status()
            elif key == "e":
                state["mode"] = "erase"
                update_status()
            elif key == "v":
                state["show_overlay"] = not state["show_overlay"]
                request_render()
            elif key == "r":
                reset_mask = draft if draft is not None else auto
                mask[:] = (reset_mask > 0).astype(np.uint8) * 255
                request_render()
            elif key == "c":
                mask[:] = 0
                request_render()
            elif key == "bracketright":
                state["radius"] = min(100, state["radius"] + 1)
                update_status()
            elif key == "bracketleft":
                state["radius"] = max(1, state["radius"] - 1)
                update_status()
            elif key in {"plus", "equal", "kp_add"}:
                width, height = viewport_size()
                zoom_at(width // 2, height // 2, 1.20)
            elif key in {"minus", "underscore", "kp_subtract"}:
                width, height = viewport_size()
                zoom_at(width // 2, height // 2, 1.0 / 1.20)
            elif key == "f":
                fit_view()
            elif key in {"q", "escape"}:
                finish("quit")

        canvas.bind("<ButtonPress-1>", on_left_press)
        canvas.bind("<ButtonRelease-1>", on_left_release)
        for button in (2, 3):
            canvas.bind(f"<ButtonPress-{button}>", on_pan_press)
            canvas.bind(f"<ButtonRelease-{button}>", on_pan_release)
        root.bind("<Motion>", on_motion)
        canvas.bind("<MouseWheel>", on_mouse_wheel)
        root.bind("<Key>", on_key)
        root.bind("<Configure>", lambda _event: request_render())
        root.protocol("WM_DELETE_WINDOW", lambda: finish("quit"))
        fit_view()
        root.focus_force()
        root.mainloop()

        if state["action"] == "next":
            index += 1
        elif state["action"] == "previous":
            index = max(0, index - 1)
        else:
            print(f"Annotation stopped at {index}/{len(manifest)} images.")
            return

    print(f"Manual annotation complete: {len(manifest)} masks.")
    validate_manual_validation_package(package_dir)

def validate_manual_validation_package(package_dir):
    """Calculate ground-truth accuracy after manual masks are completed."""
    package_dir = Path(package_dir)
    manifest_csv = package_dir / "manual_validation_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_csv}")
    manifest = pd.read_csv(manifest_csv)
    rows = []
    for item in manifest.itertuples(index=False):
        auto_path = Path(item.auto_mask)
        manual_path = Path(item.expected_manual_mask)
        if not manual_path.exists():
            continue
        auto = cv2.imread(str(auto_path), cv2.IMREAD_GRAYSCALE)
        manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
        if auto is None or manual is None:
            warnings.warn(f"Could not read mask pair for frame {item.frame}.")
            continue
        if manual.shape != auto.shape:
            manual = cv2.resize(
                manual, (auto.shape[1], auto.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        auto = auto > 0
        manual = manual > 0
        inter = int(np.sum(auto & manual))
        union = int(np.sum(auto | manual))
        auto_area = int(np.sum(auto))
        manual_area = int(np.sum(manual))
        iou = float(inter / union) if union > 0 else np.nan
        dice = (
            float(2 * inter / (auto_area + manual_area))
            if auto_area + manual_area > 0 else np.nan
        )
        precision = float(inter / auto_area) if auto_area > 0 else np.nan
        recall = float(inter / manual_area) if manual_area > 0 else np.nan
        signed_area_error = (
            float((auto_area - manual_area) / manual_area)
            if manual_area > 0 else np.nan
        )
        rows.append(
            {
                "section": item.section,
                "frame": int(item.frame),
                "auto_area_px": auto_area,
                "manual_area_px": manual_area,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "signed_area_error": signed_area_error,
                "absolute_area_error": abs(signed_area_error),
                "confidence": (
                    "high" if dice >= 0.90
                    else "trend_usable" if dice >= 0.80
                    else "needs_tuning"
                ),
            }
        )

    result = pd.DataFrame(rows)
    detail_csv = package_dir / "manual_validation_metrics.csv"
    result.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    if result.empty:
        print(
            "No completed manual masks found. Add binary masks to "
            f"{package_dir / 'manual_masks'}"
        )
        return result, pd.DataFrame()

    section_summary = (
        result.groupby("section", sort=False)
        .agg(
            n_masks=("frame", "count"),
            iou_mean=("iou", "mean"),
            iou_min=("iou", "min"),
            dice_mean=("dice", "mean"),
            dice_min=("dice", "min"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            absolute_area_error_mean=("absolute_area_error", "mean"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [{
            "section": "overall",
            "n_masks": len(result),
            "iou_mean": result["iou"].mean(),
            "iou_min": result["iou"].min(),
            "dice_mean": result["dice"].mean(),
            "dice_min": result["dice"].min(),
            "precision_mean": result["precision"].mean(),
            "recall_mean": result["recall"].mean(),
            "absolute_area_error_mean": result["absolute_area_error"].mean(),
        }]
    )
    summary = pd.concat([section_summary, overall], ignore_index=True)
    summary["confidence"] = np.select(
        [summary["dice_mean"] >= 0.90, summary["dice_mean"] >= 0.80],
        ["high", "trend_usable"],
        default="needs_tuning",
    )
    summary_csv = package_dir / "manual_validation_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved manual validation metrics: {detail_csv}")
    print(f"Saved manual validation summary: {summary_csv}")
    return result, summary

def validate_against_manual_masks(debug_data, manual_mask_dir, out_csv):
    """
    Compare automatic masks against manually annotated masks.

    Manual masks should be binary images named like one of:
    frame_000123_mask.png, frame_000123.png, or 000123.png.
    Non-zero pixels are treated as foreground.
    """
    manual_mask_dir = Path(manual_mask_dir)
    if not manual_mask_dir.exists():
        raise FileNotFoundError(f"Manual mask folder does not exist: {manual_mask_dir}")
    if not debug_data:
        raise ValueError("debug_data is empty; cannot validate manual masks.")

    rows = []
    for item in debug_data:
        frame_idx = int(item["frame"])
        candidates = [
            manual_mask_dir / f"frame_{frame_idx:06d}_mask.png",
            manual_mask_dir / f"frame_{frame_idx:06d}.png",
            manual_mask_dir / f"{frame_idx:06d}.png",
        ]
        manual_path = next((p for p in candidates if p.exists()), None)
        if manual_path is None:
            continue

        manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
        if manual is None:
            warnings.warn(f"Could not read manual mask: {manual_path}")
            continue

        auto = item["mask"].astype(bool)
        manual = cv2.resize(
            manual,
            (auto.shape[1], auto.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

        inter = int(np.sum(auto & manual))
        union = int(np.sum(auto | manual))
        auto_area = int(np.sum(auto))
        manual_area = int(np.sum(manual))
        iou = float(inter / union) if union > 0 else np.nan
        dice = float(2 * inter / (auto_area + manual_area)) if (auto_area + manual_area) > 0 else np.nan
        precision = float(inter / auto_area) if auto_area > 0 else np.nan
        recall = float(inter / manual_area) if manual_area > 0 else np.nan

        rows.append(
            {
                "frame": frame_idx,
                "manual_mask": str(manual_path),
                "auto_area_px": auto_area,
                "manual_area_px": manual_area,
                "intersection_px": inter,
                "union_px": union,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
            }
        )

    result = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved manual-mask validation CSV: {out_csv}")
    if not result.empty:
        print(
            "Manual validation mean: "
            f"IoU={result['iou'].mean():.3f}, Dice={result['dice'].mean():.3f}"
        )
    return result


def _reliability_level(value, high_limit, medium_limit):
    """Classify a non-negative sensitivity value; lower is better."""
    if not np.isfinite(value):
        return "not_available"
    if value <= float(high_limit):
        return "high"
    if value <= float(medium_limit):
        return "medium"
    return "low"


def validate_segmentation_reliability(
    video_path,
    metrics_df,
    out_dir,
    dish_center,
    dish_radius,
    hsv_params,
    vein_params=None,
    pixel_size=1.0,
    sample_count=5,
    top_area_jump_count=8,
    section_frames=None,
    samples_per_section=6,
    top_jumps_per_section=4,
    validate_veins=True,
    vein_sample_count=3,
):
    """
    Estimate internal segmentation reliability without manual ground truth.

    Representative and high-area-change frames are re-segmented after small
    HSV perturbations. This is not a true accuracy score; manual masks are
    still required for ground-truth IoU/Dice.
    """
    if metrics_df is None or metrics_df.empty:
        raise ValueError("metrics_df is empty; cannot validate reliability.")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hsv_params = dict(hsv_params or {})
    vein_params = dict(vein_params or {})

    frame_numbers = metrics_df["frame"].astype(int).to_numpy()
    area_jump = pd.to_numeric(metrics_df["area_px"], errors="coerce").diff().abs()
    section_ranges = []
    if section_frames is not None and int(section_frames) > 0:
        section_frames = min(int(section_frames), len(frame_numbers))
        first_frame = int(frame_numbers.min())
        last_frame = int(frame_numbers.max())
        middle_start = first_frame + max(0, (len(frame_numbers) - section_frames) // 2)
        starts = [first_frame, middle_start, last_frame - section_frames + 1]
        labels = ["front", "middle", "rear"]
        selected = set()
        for label, start in zip(labels, starts):
            end = min(last_frame, start + section_frames - 1)
            section_ranges.append((label, int(start), int(end)))
            subset = metrics_df[
                metrics_df["frame"].astype(int).between(int(start), int(end))
            ]
            if subset.empty:
                continue
            sampled = np.linspace(
                int(start), int(end),
                min(int(samples_per_section), len(subset)), dtype=int,
            )
            jump_count = min(int(top_jumps_per_section), max(len(subset) - 1, 0))
            jump_frames = metrics_df.loc[
                area_jump.loc[subset.index].nlargest(jump_count).index, "frame"
            ].astype(int).to_numpy()
            selected.update(sampled.tolist())
            selected.update(jump_frames.tolist())
        selected_frames = sorted(selected)
    else:
        evenly_sampled = np.linspace(
            int(frame_numbers.min()), int(frame_numbers.max()),
            min(int(sample_count), len(frame_numbers)), dtype=int,
        )
        top_count = min(int(top_area_jump_count), max(len(metrics_df) - 1, 0))
        top_jump_frames = metrics_df.loc[
            area_jump.nlargest(top_count).index, "frame"
        ].astype(int).to_numpy()
        selected_frames = sorted(set(evenly_sampled.tolist() + top_jump_frames.tolist()))

    def section_label(frame_idx):
        for label, start, end in section_ranges:
            if start <= int(frame_idx) <= end:
                return label
        return "global"

    hsv_variants = [
        {"h_min": max(0, int(hsv_params.get("h_min", 16)) - 2)},
        {"h_min": min(179, int(hsv_params.get("h_min", 16)) + 2)},
        {"h_max": max(0, int(hsv_params.get("h_max", 48)) - 2)},
        {"h_max": min(179, int(hsv_params.get("h_max", 48)) + 2)},
        {"s_min": max(0, int(hsv_params.get("s_min", 60)) - 5)},
        {"s_min": min(255, int(hsv_params.get("s_min", 60)) + 5)},
        {"v_min": max(0, int(hsv_params.get("v_min", 45)) - 5)},
        {"v_min": min(255, int(hsv_params.get("v_min", 45)) + 5)},
    ]
    metric_names = [
        "area_px", "perimeter_px", "circularity", "solidity",
        "skeleton_length_px", "n_branchpoints", "n_endpoints",
    ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for reliability validation: {video_path}")
    frame_rows = []
    try:
        for frame_idx in tqdm(selected_frames, desc="Validating segmentation reliability"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                warnings.warn(f"Could not read validation frame {frame_idx}.")
                continue

            _, base_mask, _ = segment_hsv_circle(
                frame, dish_center=dish_center, dish_radius=dish_radius, **hsv_params
            )
            base_mask = base_mask.astype(bool)
            base_reg = region_metrics(base_mask, pixel_size=pixel_size)
            base_skel = skeleton_metrics(base_mask, pixel_size=pixel_size)
            base_values = {
                **base_reg,
                "skeleton_length_px": base_skel["skeleton_length_px"],
                "n_branchpoints": base_skel["n_branchpoints"],
                "n_endpoints": base_skel["n_endpoints"],
            }
            variant_ious = []
            relative_changes = {name: [] for name in metric_names}
            for changes in hsv_variants:
                params = dict(hsv_params)
                params.update(changes)
                _, alt_mask, _ = segment_hsv_circle(
                    frame, dish_center=dish_center, dish_radius=dish_radius, **params
                )
                alt_mask = alt_mask.astype(bool)
                inter = int(np.sum(base_mask & alt_mask))
                union = int(np.sum(base_mask | alt_mask))
                variant_ious.append(float(inter / union) if union > 0 else 1.0)
                alt_reg = region_metrics(alt_mask, pixel_size=pixel_size)
                alt_skel = skeleton_metrics(alt_mask, pixel_size=pixel_size)
                alt_values = {
                    **alt_reg,
                    "skeleton_length_px": alt_skel["skeleton_length_px"],
                    "n_branchpoints": alt_skel["n_branchpoints"],
                    "n_endpoints": alt_skel["n_endpoints"],
                }
                for name in metric_names:
                    base_value = float(base_values[name])
                    alt_value = float(alt_values[name])
                    relative_changes[name].append(
                        float(abs(alt_value - base_value) / max(abs(base_value), np.finfo(float).eps))
                    )

            gray = _as_gray_uint8(frame)
            grad = cv2.magnitude(
                cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
            )
            boundary = cv2.morphologyEx(
                base_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                np.ones((3, 3), dtype=np.uint8),
            ).astype(bool)
            ring = (
                cv2.dilate(base_mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)).astype(bool)
                ^ cv2.erode(base_mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)).astype(bool)
            )
            boundary_gradient = float(np.mean(grad[boundary])) if np.any(boundary) else np.nan
            ring_gradient = float(np.mean(grad[ring])) if np.any(ring) else np.nan
            boundary_support_ratio = (
                float(boundary_gradient / ring_gradient)
                if np.isfinite(ring_gradient) and ring_gradient > 0 else np.nan
            )

            temporal_mask_iou = np.nan
            temporal_changed_fraction = np.nan
            temporal_image_support_ratio = np.nan
            if frame_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx) - 1)
                prev_ok, prev_frame = cap.read()
                if prev_ok:
                    _, prev_mask, _ = segment_hsv_circle(
                        prev_frame, dish_center=dish_center,
                        dish_radius=dish_radius, **hsv_params
                    )
                    prev_mask = prev_mask.astype(bool)
                    mask_inter = int(np.sum(prev_mask & base_mask))
                    mask_union = int(np.sum(prev_mask | base_mask))
                    temporal_mask_iou = float(mask_inter / mask_union) if mask_union > 0 else 1.0
                    changed = prev_mask ^ base_mask
                    temporal_changed_fraction = (
                        float(np.sum(changed) / mask_union) if mask_union > 0 else 0.0
                    )
                    stable = prev_mask & base_mask
                    edge_band = (
                        cv2.dilate((prev_mask | base_mask).astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(bool)
                        ^ cv2.erode(stable.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(bool)
                    )
                    stable_edge = edge_band & (~changed)
                    frame_difference = cv2.absdiff(
                        _as_gray_uint8(prev_frame), gray
                    ).astype(np.float32)
                    changed_difference = float(np.mean(frame_difference[changed])) if np.any(changed) else np.nan
                    stable_difference = float(np.mean(frame_difference[stable_edge])) if np.any(stable_edge) else np.nan
                    if np.isfinite(changed_difference) and np.isfinite(stable_difference) and stable_difference > 0:
                        temporal_image_support_ratio = float(changed_difference / stable_difference)

            csv_match = metrics_df.loc[metrics_df["frame"] == frame_idx, "area_px"]
            csv_area_px = float(csv_match.iloc[0]) if not csv_match.empty else np.nan
            csv_area_relative_error = (
                float(abs(base_reg["area_px"] - csv_area_px) / max(abs(csv_area_px), 1.0))
                if np.isfinite(csv_area_px) else np.nan
            )
            row = {
                "frame": int(frame_idx),
                "section": section_label(frame_idx),
                "base_area_px": int(base_reg["area_px"]),
                "csv_area_px": csv_area_px,
                "csv_area_relative_error": csv_area_relative_error,
                "min_perturbed_mask_iou": float(np.min(variant_ious)),
                "median_perturbed_mask_iou": float(np.median(variant_ious)),
                "boundary_gradient_ratio": boundary_support_ratio,
                "temporal_mask_iou": temporal_mask_iou,
                "temporal_changed_fraction": temporal_changed_fraction,
                "temporal_image_support_ratio": temporal_image_support_ratio,
            }
            for name in metric_names:
                row[f"{name}_max_relative_change"] = float(np.max(relative_changes[name]))
            row["body_mask_confidence"] = (
                "high"
                if row["min_perturbed_mask_iou"] >= 0.98
                and row["area_px_max_relative_change"] <= 0.03
                and row["boundary_gradient_ratio"] >= 1.20
                else "medium"
                if row["min_perturbed_mask_iou"] >= 0.95
                and row["area_px_max_relative_change"] <= 0.08
                and row["boundary_gradient_ratio"] >= 1.00
                else "low"
            )
            frame_rows.append(row)
    finally:
        cap.release()

    frame_result = pd.DataFrame(frame_rows)
    frame_csv = out_dir / "plasmodium_reliability_frames.csv"
    frame_result.to_csv(frame_csv, index=False, encoding="utf-8-sig")

    sensitivity_rules = {
        "area_px": (0.03, 0.08),
        "solidity": (0.03, 0.08),
        "perimeter_px": (0.08, 0.15),
        "circularity": (0.10, 0.20),
        "skeleton_length_px": (0.10, 0.25),
        "n_branchpoints": (0.15, 0.35),
        "n_endpoints": (0.15, 0.35),
    }
    summary_rows = []
    mask_min_iou = pd.to_numeric(frame_result["min_perturbed_mask_iou"], errors="coerce")
    summary_rows.append({
        "metric": "body_mask",
        "validation_type": "hsv_parameter_perturbation_iou",
        "n_tested_frames": int(mask_min_iou.notna().sum()),
        "median_relative_change": np.nan,
        "worst_relative_change": float(1.0 - np.nanmin(mask_min_iou)),
        "confidence": "high" if np.nanmin(mask_min_iou) >= 0.98 else "medium" if np.nanmin(mask_min_iou) >= 0.95 else "low",
        "interpretation": "internally stable; manual masks still required for true accuracy",
    })
    for name, (high_limit, medium_limit) in sensitivity_rules.items():
        values = pd.to_numeric(frame_result[f"{name}_max_relative_change"], errors="coerce")
        worst = float(np.nanmax(values)) if np.isfinite(values).any() else np.nan
        median = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
        summary_rows.append({
            "metric": name,
            "validation_type": "hsv_parameter_sensitivity",
            "n_tested_frames": int(values.notna().sum()),
            "median_relative_change": median,
            "worst_relative_change": worst,
            "confidence": _reliability_level(worst, high_limit, medium_limit),
            "interpretation": "quantitative trend" if name in {"area_px", "solidity"} else "relative trend",
        })

    if validate_veins and vein_params and "vein_length_px" in metrics_df.columns:
        valid_vein_frames = metrics_df.loc[
            pd.to_numeric(metrics_df["vein_length_px"], errors="coerce").notna(), "frame"
        ].astype(int)
        if not valid_vein_frames.empty:
            vein_frames = np.linspace(
                int(valid_vein_frames.min()), int(valid_vein_frames.max()),
                min(int(vein_sample_count), len(valid_vein_frames)), dtype=int,
            )
            threshold = float(vein_params.get("ridge_threshold_percentile", 93))
            response = float(vein_params.get("ridge_min_response", 0.01))
            erode_size = int(vein_params.get("body_erode_size", 2))
            vein_variants = [
                {"ridge_threshold_percentile": max(0.0, threshold - 1.0)},
                {"ridge_threshold_percentile": min(100.0, threshold + 1.0)},
                {"ridge_min_response": max(0.0, response * 0.5)},
                {"ridge_min_response": response * 2.0},
                {"body_erode_size": max(0, erode_size - 1)},
                {"body_erode_size": erode_size + 1},
            ]
            vein_changes = {
                "vein_area_px": [], "vein_length_px": [],
                "vein_endpoints": [], "vein_branchpoints": [],
            }
            cap = cv2.VideoCapture(str(video_path))
            try:
                for frame_idx in tqdm(vein_frames, desc="Validating vein reliability"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    _, body_mask, _ = segment_hsv_circle(
                        frame, dish_center=dish_center, dish_radius=dish_radius, **hsv_params
                    )
                    base_vein = vein_network_metrics(
                        frame, dish_center=dish_center, dish_radius=dish_radius,
                        body_mask=body_mask, pixel_size=pixel_size, **vein_params
                    )
                    for changes in vein_variants:
                        params = dict(vein_params)
                        params.update(changes)
                        alt_vein = vein_network_metrics(
                            frame, dish_center=dish_center, dish_radius=dish_radius,
                            body_mask=body_mask, pixel_size=pixel_size, **params
                        )
                        for name in vein_changes:
                            base_value = float(base_vein[name])
                            alt_value = float(alt_vein[name])
                            vein_changes[name].append(
                                float(abs(alt_value - base_value) / max(abs(base_value), 1.0))
                            )
            finally:
                cap.release()

            vein_rules = {
                "vein_area_px": (0.15, 0.35),
                "vein_length_px": (0.15, 0.35),
                "vein_endpoints": (0.20, 0.40),
                "vein_branchpoints": (0.20, 0.40),
            }
            for name, values in vein_changes.items():
                values = np.asarray(values, dtype=float)
                worst = float(np.nanmax(values)) if np.isfinite(values).any() else np.nan
                median = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
                high_limit, medium_limit = vein_rules[name]
                summary_rows.append({
                    "metric": name,
                    "validation_type": "vein_parameter_sensitivity",
                    "n_tested_frames": int(len(vein_frames)),
                    "median_relative_change": median,
                    "worst_relative_change": worst,
                    "confidence": _reliability_level(worst, high_limit, medium_limit),
                    "interpretation": "trend only; not an absolute anatomical measurement",
                })

    summary_result = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "plasmodium_reliability_summary.csv"
    summary_result.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved reliability frame CSV: {frame_csv}")
    print(f"Saved reliability summary CSV: {summary_csv}")
    if not frame_result.empty:
        print(
            "Body-mask reliability: "
            f"minimum perturbed IoU={frame_result['min_perturbed_mask_iou'].min():.3f}, "
            f"high-confidence frames={int((frame_result['body_mask_confidence'] == 'high').sum())}/{len(frame_result)}"
        )
    return frame_result, summary_result


def _render_lightweight_validation_frame_for_parallel(args):
    """
    Render one lightweight validation overlay frame.

    This helper is intentionally top-level so ProcessPoolExecutor can pickle it
    on Windows/Spyder. Video reading and video writing stay in the parent
    process; CPU-heavy HSV segmentation and overlay rendering run in workers.
    """
    (
        frame_idx,
        frame,
        label,
        total_frames,
        dish_center,
        dish_radius,
        hsv_params,
        output_scale,
        overlay_alpha,
        save_original,
    ) = args

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    _, clean_mask, mask_debug = segment_hsv_circle(
        frame,
        dish_center=dish_center,
        dish_radius=dish_radius,
        **hsv_params,
    )
    occupied_mask = mask_debug["occupied_mask"].astype(bool)
    yellow_chroma_added_mask = mask_debug.get(
        "yellow_chroma_added_mask",
        np.zeros_like(occupied_mask, dtype=bool),
    ).astype(bool)
    relaxed_added_mask = mask_debug.get(
        "relaxed_added_mask",
        np.zeros_like(occupied_mask, dtype=bool),
    ).astype(bool)
    clean_mask = clean_mask.astype(bool)
    envelope_mask = mask_debug["envelope_mask"].astype(bool)
    filled_small_holes = clean_mask & (~occupied_mask)
    remaining_envelope_holes = envelope_mask & (~clean_mask)

    overlay = frame.copy().astype(np.float32)

    def blend_layer(binary, color, alpha):
        if np.any(binary):
            color_arr = np.asarray(color, dtype=np.float32)
            overlay[binary] = (1.0 - alpha) * overlay[binary] + alpha * color_arr

    # BGR colors because frame is BGR from cv2.VideoCapture.
    blend_layer(occupied_mask, (0, 255, 0), float(overlay_alpha))
    blend_layer(yellow_chroma_added_mask, (255, 255, 0), 0.70)  # cyan in BGR
    blend_layer(relaxed_added_mask, (255, 255, 0), 0.70)
    blend_layer(filled_small_holes, (255, 0, 255), 0.70)
    # Keep internal gaps unpainted so the original image remains easy to inspect.
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    clean_contours, _ = cv2.findContours(
        clean_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    envelope_contours, _ = cv2.findContours(
        envelope_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, envelope_contours, -1, (255, 255, 0), 2)
    cv2.drawContours(overlay, clean_contours, -1, (0, 0, 255), 3)
    cv2.putText(
        overlay,
        f"{label} | frame {frame_idx}/{total_frames - 1}",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "green=occupied, cyan=focus/chroma-added, red=outer boundary",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_size = (
        max(1, int(round(frame.shape[1] * float(output_scale)))),
        max(1, int(round(frame.shape[0] * float(output_scale)))),
    )
    overlay_small = cv2.resize(overlay, output_size, interpolation=cv2.INTER_AREA)
    original_small = None
    if save_original:
        original_small = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)

    return int(frame_idx), overlay_small, original_small



def _scale_temporal_hsv_params(hsv_params, scale):
    """Scale morphology/rim parameters when validating on resized frames."""
    params = dict(hsv_params or {})
    for key in ("min_obj_size", "min_hole_size", "body_min_hole_size", "body_max_hole_size"):
        if key in params and params[key] is not None:
            params[key] = max(1, int(round(float(params[key]) * scale * scale)))
    for key in (
        "morph_open_size",
        "morph_close_size",
        "yellow_chroma_rim_exclusion_px",
        "yellow_chroma_connect_radius",
        "relaxed_connect_radius",
        "relaxed_min_obj_size",
        "rim_exclusion_px",
    ):
        if key in params and params[key] is not None:
            params[key] = max(0, int(round(float(params[key]) * scale)))
    return params


def _render_temporal_validation_overlay(
    frame,
    mask,
    temporal_added_mask,
    label,
    frame_idx,
    total_frames,
    overlay_alpha,
):
    overlay = frame.copy().astype(np.float32)
    mask = mask.astype(bool)
    if np.any(mask):
        overlay[mask] = (1.0 - float(overlay_alpha)) * overlay[mask] + float(overlay_alpha) * np.array(
            [0, 255, 0], dtype=np.float32
        )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.putText(
        overlay,
        f"{label} | frame {frame_idx}/{total_frames - 1}",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "green=body (strict+temporal), red=outer boundary",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay



def _prepare_temporal_validation_frame_for_parallel(args):
    """Prepare one scaled frame for temporal validation in a worker process."""
    (
        frame_idx,
        frame_small,
        scaled_center,
        scaled_radius,
        scaled_hsv_params,
        temporal_chroma_min,
        temporal_min_channel,
        temporal_max_value,
        scaled_temporal_rim,
        scaled_temporal_min_obj,
    ) = args
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    _, mask, _ = segment_hsv_circle(
        frame_small,
        dish_center=scaled_center,
        dish_radius=scaled_radius,
        **scaled_hsv_params,
    )
    candidate = temporal_pale_branch_candidate(
        frame_small,
        dish_center=scaled_center,
        dish_radius=scaled_radius,
        chroma_min=temporal_chroma_min,
        min_channel=temporal_min_channel,
        max_value=temporal_max_value,
        rim_exclusion_px=scaled_temporal_rim,
        min_obj_size=scaled_temporal_min_obj,
    )
    return {
        "frame_idx": int(frame_idx),
        "frame": frame_small,
        "mask": mask.astype(bool),
        "candidate": candidate.astype(bool),
    }

def save_temporal_lightweight_validation_clips(
    video_path,
    out_dir,
    dish_center,
    dish_radius,
    hsv_params,
    sections=("front", "middle", "rear"),
    section_frames=300,
    output_scale=0.5,
    stills_per_section=6,
    overlay_alpha=0.40,
    write_video=True,
    skip_edge_frames=3,
    temporal_window=5,
    temporal_min_hits=4,
    temporal_chroma_min=10,
    temporal_min_channel=35,
    temporal_max_value=240,
    temporal_rim_exclusion_px=30,
    temporal_connect_radius=8,
    temporal_min_obj_size=20,
    temporal_exterior_only=True,
    temporal_block_hole_max_area=0,
    temporal_final_smooth_window=1,
    temporal_final_smooth_min_hits=None,
    n_jobs=1,
    batch_frames=32,
):
    """Save validation clips using hits-based temporal body-mask recovery."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if total_frames <= 0:
        raise ValueError(f"Video frame count is unavailable: {video_path}")

    output_scale = float(np.clip(output_scale, 0.1, 1.0))
    temporal_window = max(3, int(temporal_window or 3))
    if temporal_window % 2 == 0:
        temporal_window += 1
    temporal_min_hits = max(1, min(int(temporal_min_hits), temporal_window))
    temporal_final_smooth_window = max(1, int(temporal_final_smooth_window or 1))
    if temporal_final_smooth_window > 1 and temporal_final_smooth_window % 2 == 0:
        temporal_final_smooth_window += 1
    if temporal_final_smooth_min_hits is None:
        temporal_final_smooth_min_hits = temporal_final_smooth_window // 2 + 1
    temporal_final_smooth_min_hits = max(
        1,
        min(int(temporal_final_smooth_min_hits), temporal_final_smooth_window),
    )
    scaled_block_hole_max_area = max(
        0,
        int(round(float(temporal_block_hole_max_area or 0) * output_scale ** 2)),
    )
    skip_edge_frames = max(0, int(skip_edge_frames or 0))
    usable_start = min(skip_edge_frames, max(0, total_frames - 1))
    usable_end = max(usable_start, total_frames - skip_edge_frames - 1)
    usable_count = max(1, usable_end - usable_start + 1)
    section_frames = max(1, min(int(section_frames), usable_count))
    write_video = bool(write_video)
    n_jobs = max(1, int(n_jobs or 1))
    batch_frames = max(1, int(batch_frames or 1))
    if n_jobs > 1:
        batch_frames = max(n_jobs, batch_frames)

    valid_sections = ("front", "middle", "rear")
    if isinstance(sections, str):
        sections = (sections,)
    selected_sections = tuple(str(section).strip().lower() for section in sections)
    invalid_sections = sorted(set(selected_sections) - set(valid_sections))
    if invalid_sections:
        raise ValueError(
            f"Unknown validation section(s): {invalid_sections}. Use any of {valid_sections}."
        )

    starts = {
        "front": usable_start,
        "middle": usable_start + max(0, (usable_count - section_frames) // 2),
        "rear": usable_end - section_frames + 1,
    }
    ranges = [
        (label, starts[label])
        for label in valid_sections
        if label in selected_sections
    ]

    validation_dir = Path(out_dir) / "validation_clips"
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)

    scaled_center = (
        int(round(dish_center[0] * output_scale)),
        int(round(dish_center[1] * output_scale)),
    )
    scaled_radius = int(round(dish_radius * output_scale))
    scaled_hsv_params = _scale_temporal_hsv_params(hsv_params, output_scale)
    scaled_temporal_rim = max(1, int(round(temporal_rim_exclusion_px * output_scale)))
    scaled_temporal_connect = max(1, int(round(temporal_connect_radius * output_scale)))
    scaled_temporal_min_obj = max(1, int(round(temporal_min_obj_size * output_scale * output_scale)))

    summary_rows = []
    completed_starts = set()
    for label, start_frame in ranges:
        if start_frame in completed_starts:
            continue
        completed_starts.add(start_frame)
        end_frame = min(usable_end, start_frame + section_frames - 1)
        section_dir = validation_dir / label
        still_dir = section_dir / "stills"
        still_dir.mkdir(parents=True, exist_ok=True)
        still_frames = set(
            np.linspace(
                start_frame,
                end_frame,
                min(int(stills_per_section), end_frame - start_frame + 1),
                dtype=int,
            ).tolist()
        )

        temporal_halo = temporal_window // 2 + temporal_final_smooth_window // 2
        clip_path = section_dir / f"{label}_color_mask_overlay.mp4"
        writer = None
        candidate_areas = []
        recoverable_candidate_areas = []
        added_areas = []
        added_fractions = []
        processed_count = 0
        peak_prepared_frames = 0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not reopen video: {video_path}")
        executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        try:
            with tqdm(
                total=end_frame - start_frame + 1,
                desc=(
                    f"Streaming temporal {label} validation with {n_jobs} CPU workers"
                    if n_jobs > 1 else f"Streaming temporal {label} validation"
                ),
            ) as pbar:
                core_start = start_frame
                while core_start <= end_frame:
                    core_end = min(end_frame, core_start + batch_frames - 1)
                    read_start = max(start_frame, core_start - temporal_halo)
                    read_end = min(end_frame, core_end + temporal_halo)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)

                    work_items = []
                    for frame_idx in range(read_start, read_end + 1):
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError(
                                f"Could not read validation frame {frame_idx} from {video_path}"
                            )
                        if output_scale == 1.0:
                            frame_small = frame
                        else:
                            frame_small = cv2.resize(
                                frame,
                                None,
                                fx=output_scale,
                                fy=output_scale,
                                interpolation=cv2.INTER_AREA,
                            )
                        work_items.append(
                            (
                                frame_idx,
                                frame_small,
                                scaled_center,
                                scaled_radius,
                                scaled_hsv_params,
                                temporal_chroma_min,
                                temporal_min_channel,
                                temporal_max_value,
                                scaled_temporal_rim,
                                scaled_temporal_min_obj,
                            )
                        )

                    if executor is not None:
                        prepared = list(
                            executor.map(
                                _prepare_temporal_validation_frame_for_parallel,
                                work_items,
                            )
                        )
                    else:
                        prepared = [
                            _prepare_temporal_validation_frame_for_parallel(item)
                            for item in work_items
                        ]
                    prepared.sort(key=lambda item: item["frame_idx"])
                    peak_prepared_frames = max(peak_prepared_frames, len(prepared))
                    recovered = apply_temporal_body_recovery_to_sequence(
                        prepared,
                        window=temporal_window,
                        min_hits=temporal_min_hits,
                        connect_radius=scaled_temporal_connect,
                        exterior_only=temporal_exterior_only,
                        block_hole_max_area=scaled_block_hole_max_area,
                        final_smooth_window=temporal_final_smooth_window,
                        final_smooth_min_hits=temporal_final_smooth_min_hits,
                    )
                    core_items = [
                        item
                        for item in recovered
                        if core_start <= int(item["frame_idx"]) <= core_end
                    ]
                    if len(core_items) != core_end - core_start + 1:
                        raise RuntimeError(
                            f"Temporal batch returned {len(core_items)} core frames; "
                            f"expected {core_end - core_start + 1}."
                        )

                    for item in core_items:
                        frame_idx = int(item["frame_idx"])
                        mask = item["mask"].astype(bool)
                        strict_mask = item.get("strict_mask", item["mask"]).astype(bool)
                        candidate = item["candidate"].astype(bool)
                        addition = item.get(
                            "temporal_added_mask",
                            np.zeros_like(mask, dtype=bool),
                        ).astype(bool)
                        candidate_area = int(candidate.sum())
                        recoverable_candidate_area = int((candidate & (~strict_mask)).sum())
                        added_area = int(addition.sum())
                        candidate_areas.append(candidate_area)
                        recoverable_candidate_areas.append(recoverable_candidate_area)
                        added_areas.append(added_area)
                        if np.any(mask):
                            added_fractions.append(float(added_area / mask.sum()))
                        else:
                            added_fractions.append(0.0)
                        overlay = _render_temporal_validation_overlay(
                            item["frame"],
                            mask,
                            addition,
                            label,
                            frame_idx,
                            total_frames,
                            overlay_alpha,
                        )
                        if write_video:
                            if writer is None:
                                height, width = overlay.shape[:2]
                                writer = cv2.VideoWriter(
                                    str(clip_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    source_fps if source_fps > 0 else 30.0,
                                    (width, height),
                                    True,
                                )
                                if not writer.isOpened():
                                    raise ValueError(
                                        f"Could not open validation writer: {clip_path}"
                                    )
                            writer.write(overlay)

                        if frame_idx in still_frames:
                            cv2.imwrite(
                                str(still_dir / f"frame_{frame_idx:06d}_original.jpg"),
                                item["frame"],
                                [cv2.IMWRITE_JPEG_QUALITY, 92],
                            )
                            cv2.imwrite(
                                str(still_dir / f"frame_{frame_idx:06d}_overlay.jpg"),
                                overlay,
                                [cv2.IMWRITE_JPEG_QUALITY, 92],
                            )
                            cv2.imwrite(
                                str(still_dir / f"frame_{frame_idx:06d}_strict_occupied_mask.png"),
                                strict_mask.astype(np.uint8) * 255,
                            )
                            cv2.imwrite(
                                str(still_dir / f"frame_{frame_idx:06d}_temporal_candidate_mask.png"),
                                candidate.astype(np.uint8) * 255,
                            )
                            cv2.imwrite(
                                str(still_dir / f"frame_{frame_idx:06d}_temporal_added_mask.png"),
                                addition.astype(np.uint8) * 255,
                            )
                            cv2.imwrite(
                                str(still_dir / f"frame_{frame_idx:06d}_occupied_temporal_mask.png"),
                                mask.astype(np.uint8) * 255,
                            )

                    processed_count += len(core_items)
                    pbar.update(len(core_items))
                    core_start = core_end + 1
                    del work_items, prepared, recovered, core_items
                    gc.collect()
        finally:
            cap.release()
            if executor is not None:
                executor.shutdown(wait=True)
            if writer is not None:
                writer.release()

        candidate_area_px_mean = (
            float(np.mean(candidate_areas)) if candidate_areas else 0.0
        )
        recoverable_candidate_area_px_mean = (
            float(np.mean(recoverable_candidate_areas))
            if recoverable_candidate_areas else 0.0
        )
        added_area_px_mean = float(np.mean(added_areas)) if added_areas else 0.0
        added_fraction_mean = (
            float(np.mean(added_fractions)) if added_fractions else 0.0
        )
        added_fraction_p95 = (
            float(np.percentile(added_fractions, 95)) if added_fractions else 0.0
        )
        frames_with_added = int(np.count_nonzero(np.asarray(added_areas) > 0))
        print(f"[{label}] temporal_candidate_area_px_mean = {candidate_area_px_mean:.2f}")
        print(f"[{label}] temporal_added_area_px_mean = {added_area_px_mean:.2f}")
        print(f"[{label}] temporal_added_fraction_mean = {added_fraction_mean:.6f}")
        print(f"[{label}] temporal_added_fraction_p95 = {added_fraction_p95:.6f}")
        print(
            f"[{label}] number_of_frames_with_temporal_added_area_px > 0 = "
            f"{frames_with_added}"
        )

        candidate_low_threshold = max(
            float(scaled_temporal_min_obj * 5),
            float(np.pi * scaled_radius ** 2 * 0.001),
        )
        if candidate_area_px_mean < candidate_low_threshold:
            warnings.warn(
                "Temporal candidate is too strict; consider lowering "
                "TEMPORAL_CHROMA_MIN or TEMPORAL_MIN_CHANNEL."
            )
        elif (
            recoverable_candidate_area_px_mean >= candidate_low_threshold
            and added_area_px_mean
            < max(1.0, recoverable_candidate_area_px_mean * 0.05)
        ):
            warnings.warn(
                "Temporal consensus/connectivity is too strict; consider lowering "
                "TEMPORAL_MIN_HITS or increasing TEMPORAL_CONNECT_RADIUS."
            )

        summary_rows.append(
            {
                "section": label,
                "video_total_frames": total_frames,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "processed_frames": processed_count,
                "output_scale": output_scale,
                "write_video": write_video,
                "temporal_window": temporal_window,
                "temporal_min_hits": temporal_min_hits,
                "temporal_chroma_min": temporal_chroma_min,
                "temporal_rim_exclusion_px": temporal_rim_exclusion_px,
                "temporal_connect_radius": temporal_connect_radius,
                "temporal_exterior_only": bool(temporal_exterior_only),
                "temporal_block_hole_max_area": int(temporal_block_hole_max_area),
                "temporal_final_smooth_window": int(temporal_final_smooth_window),
                "temporal_final_smooth_min_hits": int(temporal_final_smooth_min_hits),
                "temporal_halo_frames": int(temporal_halo),
                "peak_prepared_frames": int(peak_prepared_frames),
                "temporal_candidate_area_px_mean": candidate_area_px_mean,
                "temporal_recoverable_candidate_area_px_mean": (
                    recoverable_candidate_area_px_mean
                ),
                "temporal_added_area_px_mean": added_area_px_mean,
                "temporal_added_fraction_mean": added_fraction_mean,
                "temporal_added_fraction_p95": added_fraction_p95,
                "frames_with_temporal_added_area_px_gt_0": frames_with_added,
                "mean_temporal_added_fraction": (
                    float(np.nanmean(added_fractions))
                    if added_fractions else np.nan
                ),
                "clip_path": str(clip_path) if write_video else "not_written",
                "stills_path": str(still_dir),
            }
        )
        gc.collect()
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = validation_dir / "validation_clips_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved temporal validation clips: {validation_dir}")
    print(f"Saved validation summary: {summary_csv}")
    return summary_df

def save_lightweight_validation_clips(
    video_path,
    out_dir,
    dish_center,
    dish_radius,
    hsv_params,
    sections=("front", "middle", "rear"),
    section_frames=300,
    output_scale=0.5,
    stills_per_section=6,
    overlay_alpha=0.40,
    n_jobs=1,
    batch_frames=32,
    write_video=True,
):
    """
    Save color overlay clips/stills for selected parts of the full video.

    Body segmentation and overlay rendering can run in parallel. The parent
    process still reads source frames and writes mp4 files in frame order, because
    VideoCapture/VideoWriter are safer and more reliable when kept single-writer.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if total_frames <= 0:
        raise ValueError(f"Video frame count is unavailable: {video_path}")

    n_jobs = max(1, int(n_jobs or 1))
    batch_frames = max(1, int(batch_frames or 1))
    if n_jobs > 1:
        batch_frames = max(n_jobs, batch_frames)
    section_frames = max(1, min(int(section_frames), total_frames))
    output_scale = float(np.clip(output_scale, 0.1, 1.0))
    write_video = bool(write_video)
    n_jobs = max(1, int(n_jobs or 1))
    batch_frames = max(1, int(batch_frames or 1))
    if n_jobs > 1:
        batch_frames = max(n_jobs, batch_frames)

    valid_sections = ("front", "middle", "rear")
    if isinstance(sections, str):
        sections = (sections,)
    selected_sections = tuple(str(section).strip().lower() for section in sections)
    invalid_sections = sorted(set(selected_sections) - set(valid_sections))
    if invalid_sections:
        raise ValueError(
            f"Unknown validation section(s): {invalid_sections}. "
            f"Use any of {valid_sections}."
        )
    if not selected_sections:
        raise ValueError("At least one validation section must be selected.")

    starts = {
        "front": 0,
        "middle": max(0, (total_frames - section_frames) // 2),
        "rear": max(0, total_frames - section_frames),
    }
    ranges = [
        (label, starts[label])
        for label in valid_sections
        if label in selected_sections
    ]
    validation_dir = Path(out_dir) / "validation_clips"
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    completed_starts = set()
    for label, start_frame in ranges:
        if start_frame in completed_starts:
            continue
        completed_starts.add(start_frame)
        end_frame = min(total_frames - 1, start_frame + section_frames - 1)
        section_dir = validation_dir / label
        still_dir = section_dir / "stills"
        still_dir.mkdir(parents=True, exist_ok=True)
        still_frames = set(
            np.linspace(
                start_frame,
                end_frame,
                min(int(stills_per_section), end_frame - start_frame + 1),
                dtype=int,
            ).tolist()
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not reopen video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = None
        clip_path = section_dir / f"{label}_color_mask_overlay.mp4"
        processed = 0
        next_frame = start_frame

        executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        try:
            with tqdm(
                total=end_frame - start_frame + 1,
                desc=(
                    f"Saving {label} validation clip with {n_jobs} CPU workers"
                    if n_jobs > 1 else f"Saving {label} validation clip"
                ),
            ) as pbar:
                while next_frame <= end_frame:
                    work_items = []
                    while len(work_items) < batch_frames and next_frame <= end_frame:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        work_items.append(
                            (
                                next_frame,
                                frame,
                                label,
                                total_frames,
                                dish_center,
                                dish_radius,
                                hsv_params,
                                output_scale,
                                overlay_alpha,
                                next_frame in still_frames,
                            )
                        )
                        next_frame += 1

                    if not work_items:
                        break

                    if executor is not None:
                        rendered = list(executor.map(_render_lightweight_validation_frame_for_parallel, work_items))
                    else:
                        rendered = [_render_lightweight_validation_frame_for_parallel(item) for item in work_items]
                    rendered.sort(key=lambda item: item[0])

                    for frame_idx, overlay_small, original_small in rendered:
                        if write_video:
                            if writer is None:
                                height, width = overlay_small.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(
                                    str(clip_path),
                                    fourcc,
                                    source_fps if source_fps > 0 else 30.0,
                                    (width, height),
                                    True,
                                )
                                if not writer.isOpened():
                                    raise ValueError(f"Could not open validation writer: {clip_path}")
                            writer.write(overlay_small)

                        if frame_idx in still_frames:
                            if original_small is not None:
                                cv2.imwrite(
                                    str(still_dir / f"frame_{frame_idx:06d}_original.jpg"),
                                    original_small,
                                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                                )
                            cv2.imwrite(
                                str(still_dir / f"frame_{frame_idx:06d}_overlay.jpg"),
                                overlay_small,
                                [cv2.IMWRITE_JPEG_QUALITY, 92],
                            )
                        processed += 1
                        pbar.update(1)

                    del work_items, rendered
                    gc.collect()
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if executor is not None:
                executor.shutdown(wait=True)

        summary_rows.append(
            {
                "section": label,
                "video_total_frames": total_frames,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "processed_frames": processed,
                "output_scale": output_scale,
                "n_jobs": n_jobs,
                "batch_frames": batch_frames,
                "write_video": write_video,
                "clip_path": str(clip_path) if write_video else "not_written",
                "stills_path": str(still_dir),
            }
        )
        if write_video:
            print(f"Saved {label} validation clip: {clip_path}")
        else:
            print(f"Saved {label} validation stills: {still_dir}")

    summary = pd.DataFrame(summary_rows)
    summary_csv = validation_dir / "validation_clips_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved validation clip summary: {summary_csv}")
    return summary


def summarize_run(df, label):
    """
    Generate a one-row summary table for one experimental run.
    """
    if df is None or df.empty:
        raise ValueError("df is empty; cannot summarize run.")

    def mean_col(col):
        if col not in df.columns:
            return np.nan
        vals = df[col].to_numpy(dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

    def final_col(col):
        if col not in df.columns:
            return np.nan
        vals = df[col].to_numpy(dtype=float)
        valid = vals[np.isfinite(vals)]
        return float(valid[-1]) if valid.size else np.nan

    return pd.DataFrame(
        [
            {
                "label": label,
                "n_frames": int(len(df)),
                "area_final": final_col("area"),
                "area_mean": mean_col("area"),
                "dA_dt_mean": mean_col("dA_dt"),
                "front_speed_mean": mean_col("front_speed_mean"),
                "explore_ratio_mean": mean_col("explore_ratio"),
                "retreat_ratio_mean": mean_col("retreat_ratio"),
                "circularity_mean": mean_col(
                    "circularity_trusted" if "circularity_trusted" in df.columns else "circularity"
                ),
                "solidity_mean": mean_col(
                    "solidity_trusted" if "solidity_trusted" in df.columns else "solidity"
                ),
                "branchpoints_mean": mean_col(
                    "n_branchpoints_trusted" if "n_branchpoints_trusted" in df.columns else "n_branchpoints"
                ),
                "endpoints_mean": mean_col(
                    "n_endpoints_trusted" if "n_endpoints_trusted" in df.columns else "n_endpoints"
                ),
                "skeleton_length_mean": mean_col(
                    "skeleton_length_trusted" if "skeleton_length_trusted" in df.columns else "skeleton_length"
                ),
                "body_mask_high_fraction": float(
                    np.mean(df["body_mask_confidence"].astype(str) == "high")
                ) if "body_mask_confidence" in df.columns else np.nan,
                "body_mask_low_fraction": float(
                    np.mean(df["body_mask_confidence"].astype(str) == "low")
                ) if "body_mask_confidence" in df.columns else np.nan,
                "remaining_hole_fraction_mean": mean_col("remaining_hole_fraction"),
            }
        ]
    )


def _show_preprocess_debug(pre_debug):
    """Display first-frame preprocessing debug figure."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    axes[0].imshow(pre_debug["gray"], cmap="gray")
    axes[0].set_title("Gray")
    axes[1].imshow(pre_debug["background"], cmap="gray")
    axes[1].set_title("Background")
    axes[2].imshow(pre_debug["corrected"], cmap="gray")
    axes[2].set_title("Corrected")
    axes[3].imshow(pre_debug["enhanced"], cmap="gray")
    axes[3].set_title("Enhanced")
    for ax in axes:
        ax.axis("off")
    roi = pre_debug.get("analysis_roi")
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        for ax in axes:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=1.5)
            ax.add_patch(rect)
    if pre_debug.get("dish_center") is not None and pre_debug.get("dish_radius") is not None:
        cx, cy = pre_debug["dish_center"]
        radius = pre_debug["dish_radius"]
        for ax in axes:
            circle = plt.Circle((cx, cy), radius, fill=False, edgecolor="yellow", linewidth=1.5)
            ax.add_patch(circle)
    plt.show()
    return fig


def _show_segmentation_overlay(gray, mask):
    """Display first-frame segmentation and skeleton overlay."""
    skel = skeleton_metrics(mask)
    overlay = overlay_mask_skeleton(
        gray,
        mask=mask,
        skeleton=skel["skeleton"],
        endpoint_mask=skel["endpoint_mask"],
        branch_mask=skel["branch_mask"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Mask")
    axes[1].imshow(overlay)
    axes[1].set_title("Mask + skeleton overlay")
    for ax in axes:
        ax.axis("off")
    plt.show()
    return fig


def _load_first_frame_for_preview(input_path, input_mode, as_gray=True):
    """
    Load only the first frame for pre-analysis preview.

    Returns first_frame, total_hint.
    """
    input_path = Path(input_path)
    if input_mode == "folder":
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a folder: {input_path}")
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if len(files) == 0:
            raise ValueError(f"No readable image frames found in: {input_path}")
        img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read first image frame: {files[0]}")
        if as_gray:
            img = _as_gray_uint8(img)
        elif img.ndim == 2:
            img = cv2.cvtColor(_as_gray_uint8(img), cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img, len(files)

    if input_mode == "video":
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Could not read first video frame: {input_path}")
        if as_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, total

    raise ValueError('INPUT_MODE must be "folder" or "video".')


def preview_first_frame(
    first_frame,
    segmentation_profile="classic",
    analysis_roi=None,
    dish_center=None,
    dish_radius=None,
    hsv_params=None,
    preprocess_params=None,
    segment_params=None,
):
    """Display preprocessing and segmentation preview for the first frame."""
    hsv_params = hsv_params or {}
    preprocess_params = preprocess_params or {}
    segment_params = segment_params or {}

    first_gray = _as_gray_uint8(first_frame)
    if segmentation_profile == "hsv_circle":
        first_enhanced, first_mask, first_pre_debug = segment_hsv_circle(
            first_frame,
            dish_center=dish_center,
            dish_radius=dish_radius,
            **hsv_params,
        )
    else:
        first_enhanced, first_mask, first_pre_debug = preprocess_and_segment_frame(
            first_gray,
            preprocess_params=preprocess_params,
            segment_params=segment_params,
            analysis_roi=analysis_roi,
        )

    _show_preprocess_debug(first_pre_debug)
    _show_segmentation_overlay(first_gray, first_mask)
    return first_enhanced, first_mask, first_pre_debug


def save_segment_test_examples(
    video_path,
    out_dir,
    segment_names=("early", "middle", "late"),
    frames_per_part=3,
    dt=30.0,
    pixel_size=1.0,
    dish_center=None,
    dish_radius=None,
    hsv_params=None,
    vein_params=None,
):
    """
    Save quick early/middle/late visual checks for long videos.

    This is meant for parameter validation before running every frame. It writes
    overlay, body mask, vein mask, and a CSV with the sampled metrics.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read frame count from video: {video_path}")

    n = max(1, int(frames_per_part))
    offsets = np.linspace(0, max(0, min(30, total - 1)), n, dtype=int)
    starts = {
        "early": 0,
        "middle": max(0, total // 2 - int(offsets[-1] // 2)),
        "late": max(0, total - int(offsets[-1]) - 1),
    }

    hsv_params = hsv_params or {}
    vein_params = vein_params or {}
    rows = []

    for label in segment_names:
        if label not in starts:
            continue
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for off in offsets:
            frame_idx = min(total - 1, int(starts[label] + off))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            _, mask, _ = segment_hsv_circle(
                frame_bgr,
                dish_center=dish_center,
                dish_radius=dish_radius,
                **hsv_params,
            )
            reg = region_metrics(mask, pixel_size=pixel_size)
            skel = skeleton_metrics(mask, pixel_size=pixel_size)
            vein = vein_network_metrics(
                frame_bgr,
                dish_center=dish_center,
                dish_radius=dish_radius,
                body_mask=mask,
                pixel_size=pixel_size,
                **vein_params,
            )
            overlay = overlay_mask_skeleton(
                frame_bgr,
                mask=mask,
                skeleton=skel["skeleton"],
                endpoint_mask=skel["endpoint_mask"],
                branch_mask=skel["branch_mask"],
                vein_mask=vein["vein_mask"],
                vein_skeleton=vein["vein_skeleton"],
                alpha=0.28,
            )

            prefix = label_dir / f"frame_{frame_idx:06d}"
            cv2.imwrite(str(prefix) + "_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(prefix) + "_mask.png", (mask.astype(np.uint8) * 255))
            cv2.imwrite(str(prefix) + "_vein_mask.png", (vein["vein_mask"].astype(np.uint8) * 255))

            area_px = reg["area_px"]
            rows.append(
                {
                    "segment": label,
                    "frame": frame_idx,
                    "time": float(frame_idx * dt) if dt is not None else np.nan,
                    "area_px": area_px,
                    "area": reg["area"],
                    "vein_area_px": vein["vein_area_px"],
                    "vein_length_px": vein["vein_length_px"],
                    "vein_length": vein["vein_length"],
                    "vein_area_fraction": (
                        float(vein["vein_area_px"] / area_px) if area_px else np.nan
                    ),
                    "vein_branchpoints": vein["vein_branchpoints"],
                    "focus_score": focus_score_laplacian(_as_gray_uint8(frame_bgr)),
                }
            )

    cap.release()
    sample_df = pd.DataFrame(rows)
    sample_csv = out_dir / "segment_test_metrics.csv"
    sample_df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    print(f"Saved segment test examples: {out_dir}")
    print(f"Saved segment test metrics: {sample_csv}")
    return sample_df


if __name__ == "__main__":
    INPUT_MODE = "video"  # "folder" or "video"
    INPUT_PATH = r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\output2.mp4"
    OUTPUT_DIR = r"results_plasmodium"

    # ============================================================
    # Run mode selector
    # ============================================================
    # Choose one mode before running:
    #
    # "formal"
    #     Run the normal full HSV analysis and save formal metrics/summary/overlay.
    #
    # "test_front_middle_rear"
    #     Only export lightweight validation clips/stills from front, middle,
    #     and rear sections. Full formal analysis is skipped.
    #     If USE_TEMPORAL_BODY_RECOVERY is True, these clips use temporal recovery.
    #
    # "test_rear"
    #     Only export the rear lightweight validation clip/stills.
    #     Full formal analysis is skipped.
    #     If USE_TEMPORAL_BODY_RECOVERY is True, these clips use temporal recovery.
    #
    # "formal_plus_front_middle_rear_test"
    #     Run full formal HSV analysis first, then export front/middle/rear
    #     lightweight validation clips/stills.
    RUN_MODE = "test_front_middle_rear"

    VALID_RUN_MODES = {
        "formal",
        "test_front_middle_rear",
        "test_rear",
        "formal_plus_front_middle_rear_test",
    }
    if RUN_MODE not in VALID_RUN_MODES:
        raise ValueError(
            f"RUN_MODE must be one of {sorted(VALID_RUN_MODES)}, got {RUN_MODE!r}"
        )

    RUN_FORMAL_ANALYSIS = RUN_MODE in {
        "formal",
        "formal_plus_front_middle_rear_test",
    }
    RUN_TEST_CLIPS = RUN_MODE in {
        "test_front_middle_rear",
        "test_rear",
        "formal_plus_front_middle_rear_test",
    }
    TEST_ONLY_MODE = RUN_MODE in {"test_front_middle_rear", "test_rear"}
    if RUN_MODE == "test_rear":
        SELECTED_VALIDATION_SECTIONS = ("rear",)
    else:
        SELECTED_VALIDATION_SECTIONS = ("front", "middle", "rear")

    # ============================================================
    # Basic processing settings
    # ============================================================
    CLEAN_OUTPUT_DEBUG = True
    ENABLE_CPU_PARALLEL = True
    CPU_WORKERS = 8
    LOW_MEMORY_VIDEO_STREAM = True
    PARALLEL_BATCH_FRAMES = 32

    # Use the real acquisition interval, not the exported video playback FPS.
    # Example: images captured every 10 s and exported as 30 FPS video still use 10.0 here.
    FRAME_INTERVAL = 10.0
    PIXEL_SIZE = 1.0
    ROI = None  # Example: ROI = (x, y, w, h)
    SEGMENTATION_PROFILE = "hsv_circle"  # "hsv_circle", "classic", or "grabcut_blob"
    ANALYSIS_ROI = (700, 520, 850, 700)  # Used only by "classic" / "grabcut_blob".
    DISH_CENTER = (1280, 920)  # x, y. Adjust once per camera setup if needed.
    DISH_RADIUS = 680

    # Formal analysis time window. Use the same cutoff for microgravity and 1g.
    # None = analyze the whole video. Example: 40000 s / 10 s = 4000 frames.
    FORMAL_ANALYSIS_MAX_TIME_SEC = None
    if FORMAL_ANALYSIS_MAX_TIME_SEC is None:
        FORMAL_ANALYSIS_MAX_FRAMES = None
    else:
        FORMAL_ANALYSIS_MAX_FRAMES = int(
            round(float(FORMAL_ANALYSIS_MAX_TIME_SEC) / float(FRAME_INTERVAL))
        )

    MANUAL_MASK_DIR = None  # Optional folder with hand-labeled masks for IoU/Dice validation.
    SEGMENT_TEST_FRAMES_PER_PART = 3
    RUN_RELIABILITY_VALIDATION = False
    RELIABILITY_SAMPLE_COUNT = 5
    RELIABILITY_TOP_AREA_JUMPS = 8
    RELIABILITY_VALIDATE_VEINS = True

    # Validation clip settings. These are controlled by RUN_MODE.
    RUN_LIGHTWEIGHT_VALIDATION_CLIPS = RUN_TEST_CLIPS
    VALIDATION_ONLY_MODE = TEST_ONLY_MODE
    VALIDATION_CLIP_SECTIONS = SELECTED_VALIDATION_SECTIONS
    VALIDATION_CLIP_SECTION_FRAMES = 300
    VALIDATION_CLIP_OUTPUT_SCALE = 0.5
    VALIDATION_CLIP_STILLS_PER_SECTION = 6
    VALIDATION_CLIP_CPU_WORKERS = CPU_WORKERS
    VALIDATION_CLIP_BATCH_FRAMES = 48
    VALIDATION_CLIP_WRITE_VIDEO = True

    # Temporal recovery is validation-only in this file.
    # Formal metrics and the formal overlay still use the strict HSV mask.
    USE_TEMPORAL_BODY_RECOVERY = True
    TEMPORAL_SKIP_EDGE_FRAMES = 3
    TEMPORAL_WINDOW = 5
    TEMPORAL_MIN_HITS = 3
    TEMPORAL_CHROMA_MIN = 9
    TEMPORAL_MIN_CHANNEL = 40
    TEMPORAL_MAX_VALUE = 240
    TEMPORAL_RIM_EXCLUSION_PX = 30
    TEMPORAL_CONNECT_RADIUS = 10
    TEMPORAL_MIN_OBJ_SIZE = 15
    TEMPORAL_EXTERIOR_ONLY = False
    TEMPORAL_BLOCK_HOLE_MAX_AREA = 1500
    TEMPORAL_FINAL_SMOOTH_WINDOW = 5
    TEMPORAL_FINAL_SMOOTH_MIN_HITS = 3

    # Formal-analysis settings. Test-only modes skip the formal analysis, so
    # these values only matter when RUN_FORMAL_ANALYSIS is True.
    MAX_FRAMES = FORMAL_ANALYSIS_MAX_FRAMES
    SAVE_OVERLAY_VIDEO = True
    OVERLAY_VIDEO_FPS = 30
    OVERLAY_VIDEO_OUTPUT_SCALE = 0.5  # 1.0 = original size; 0.5 = half width/height.
    VEIN_EVERY_N_FRAMES = 0  # Disable separate vein/ridge analysis for HSV body-mask comparison.
    COMPUTE_BODY_SKELETON = True
    COMPUTE_FRONT_SPEED = True
    RUN_QC_VALIDATION = bool(RUN_FORMAL_ANALYSIS)
    RUN_SEGMENT_TESTS = False

    print("Run configuration:")
    print("  MODE = HSV body-mask analysis")
    print(f"  RUN_MODE = {RUN_MODE}")
    print(f"  RUN_FORMAL_ANALYSIS = {RUN_FORMAL_ANALYSIS}")
    print(f"  RUN_TEST_CLIPS = {RUN_TEST_CLIPS}")
    print(f"  TEST_ONLY_MODE = {TEST_ONLY_MODE}")
    print(f"  FORMAL_ANALYSIS_MAX_TIME_SEC = {FORMAL_ANALYSIS_MAX_TIME_SEC}")
    print(f"  FORMAL_ANALYSIS_MAX_FRAMES = {FORMAL_ANALYSIS_MAX_FRAMES}")
    print(f"  MAX_FRAMES = {MAX_FRAMES}")
    print(f"  OVERLAY_VIDEO_OUTPUT_SCALE = {OVERLAY_VIDEO_OUTPUT_SCALE}")
    print(f"  SAVE_OVERLAY_VIDEO = {SAVE_OVERLAY_VIDEO}")
    print(f"  VEIN_EVERY_N_FRAMES = {VEIN_EVERY_N_FRAMES}")
    print(f"  COMPUTE_BODY_SKELETON = {COMPUTE_BODY_SKELETON}")
    print(f"  COMPUTE_FRONT_SPEED = {COMPUTE_FRONT_SPEED}")
    print(f"  RUN_QC_VALIDATION = {RUN_QC_VALIDATION}")
    print(f"  CLEAN_OUTPUT_DEBUG = {CLEAN_OUTPUT_DEBUG}")
    print(f"  ENABLE_CPU_PARALLEL = {ENABLE_CPU_PARALLEL}")
    print(f"  CPU_WORKERS = {CPU_WORKERS}")
    print(f"  LOW_MEMORY_VIDEO_STREAM = {LOW_MEMORY_VIDEO_STREAM}")
    print(f"  PARALLEL_BATCH_FRAMES = {PARALLEL_BATCH_FRAMES}")
    print(f"  RUN_RELIABILITY_VALIDATION = {RUN_RELIABILITY_VALIDATION}")
    print(f"  RELIABILITY_VALIDATE_VEINS = {RELIABILITY_VALIDATE_VEINS}")
    print(f"  RUN_LIGHTWEIGHT_VALIDATION_CLIPS = {RUN_LIGHTWEIGHT_VALIDATION_CLIPS}")
    print(f"  VALIDATION_ONLY_MODE = {VALIDATION_ONLY_MODE}")
    print(f"  VALIDATION_CLIP_SECTIONS = {VALIDATION_CLIP_SECTIONS}")
    print(f"  VALIDATION_CLIP_SECTION_FRAMES = {VALIDATION_CLIP_SECTION_FRAMES}")
    print(f"  VALIDATION_CLIP_CPU_WORKERS = {VALIDATION_CLIP_CPU_WORKERS}")
    print(f"  VALIDATION_CLIP_BATCH_FRAMES = {VALIDATION_CLIP_BATCH_FRAMES}")
    print(f"  VALIDATION_CLIP_WRITE_VIDEO = {VALIDATION_CLIP_WRITE_VIDEO}")
    print(f"  USE_TEMPORAL_BODY_RECOVERY = {USE_TEMPORAL_BODY_RECOVERY}")
    print("  TEMPORAL_SCOPE = validation_only; formal metrics still use strict HSV")
    print(f"  TEMPORAL_SKIP_EDGE_FRAMES = {TEMPORAL_SKIP_EDGE_FRAMES}")
    print(f"  TEMPORAL_WINDOW = {TEMPORAL_WINDOW}")
    print(f"  TEMPORAL_MIN_HITS = {TEMPORAL_MIN_HITS}")
    print(f"  TEMPORAL_CHROMA_MIN = {TEMPORAL_CHROMA_MIN}")
    print(f"  TEMPORAL_RIM_EXCLUSION_PX = {TEMPORAL_RIM_EXCLUSION_PX}")
    print(f"  TEMPORAL_CONNECT_RADIUS = {TEMPORAL_CONNECT_RADIUS}")
    print(f"  TEMPORAL_EXTERIOR_ONLY = {TEMPORAL_EXTERIOR_ONLY}")
    print(f"  TEMPORAL_BLOCK_HOLE_MAX_AREA = {TEMPORAL_BLOCK_HOLE_MAX_AREA}")
    print(f"  TEMPORAL_FINAL_SMOOTH_WINDOW = {TEMPORAL_FINAL_SMOOTH_WINDOW}")
    print(f"  TEMPORAL_FINAL_SMOOTH_MIN_HITS = {TEMPORAL_FINAL_SMOOTH_MIN_HITS}")

    hsv_params = dict(
        h_min=15,
        h_max=49,
        s_min=54,
        v_min=30,
        min_obj_size=150,
        min_hole_size=0,
        morph_open_size=1,
        morph_close_size=0,
        keep_largest=True,
        # One global parameter set calibrated against 30 front/middle/rear masks.
        # The rear reference masks use AI threshold 0.70 to recover pale branches.
        fill_small_body_holes=False,
        use_yellow_chroma_extension=False,  # HSV-only: disable RGB/yellow-chroma extension
        yellow_chroma_min=23,
        yellow_chroma_max_value=225,
        yellow_chroma_min_channel=55,
        yellow_chroma_rim_exclusion_px=20,
        #yellow_chroma_min=30,
        #yellow_chroma_max_value=236,
        #yellow_chroma_min_channel=68,
        #yellow_chroma_rim_exclusion_px=8,
        yellow_chroma_connect_radius=0,
        yellow_chroma_connect_iterations=3,
        body_hole_area_ratio=0.0005,
        body_min_hole_size=50,
        body_max_hole_size=150,
        use_relaxed_body_mask=False,
        relaxed_h_min=14,
        relaxed_h_max=55,
        relaxed_s_min=25,
        relaxed_v_min=35,
        relaxed_connect_radius=30,
        relaxed_min_obj_size=50,
        rim_exclusion_px=40,
    )

    vein_params = dict(
        method="ridge",
        h_min=16,
        h_max=55,
        s_min=45,
        v_min=80,
        min_obj_size=20,
        morph_open_size=1,
        body_erode_size=2,
        ridge_sigma_min=1,
        ridge_sigma_max=5,
        ridge_threshold_percentile=93,
        ridge_min_response=0.01,
        ridge_black_ridges=False,
        ridge_use_hsv_gate=False,
    )

    preprocess_params = dict(
        bg_method="gaussian",
        bg_kernel=301,
        correction_mode="divide",
        clahe_clip=2.0,
        clahe_grid=(8, 8),
        median_ksize=3,
        gaussian_sigma=1.0,
        invert_if_needed=False,
    )

    segment_params = dict(
        method="grabcut",
        threshold_scale=1.0,
        min_obj_size=1000,
        min_hole_size=3000,
        keep_largest=True,
        component_selection="center",
        morph_open_size=2,
        morph_close_size=8,
        grabcut_margin=0.18,
        grabcut_iter=5,
    )

    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"INPUT_PATH does not exist: {input_path}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_mode_indicator = {
        "run_mode": RUN_MODE,
        "run_formal_analysis": bool(RUN_FORMAL_ANALYSIS),
        "run_test_clips": bool(RUN_TEST_CLIPS),
        "test_only_mode": bool(TEST_ONLY_MODE),
        "validation_sections": ";".join(VALIDATION_CLIP_SECTIONS),
        "segmentation_profile": SEGMENTATION_PROFILE,
        "hsv_only": True,
        "use_temporal_body_recovery": bool(USE_TEMPORAL_BODY_RECOVERY),
        "temporal_scope": "validation_only",
        "formal_metrics_mask": "strict_hsv_clean_mask",
        "formal_analysis_max_time_sec": FORMAL_ANALYSIS_MAX_TIME_SEC,
        "formal_analysis_max_frames": FORMAL_ANALYSIS_MAX_FRAMES,
        "formal_overlay_video_output_scale": OVERLAY_VIDEO_OUTPUT_SCALE,
        "use_yellow_chroma_extension": bool(hsv_params.get("use_yellow_chroma_extension", False)),
        "use_relaxed_body_mask": bool(hsv_params.get("use_relaxed_body_mask", False)),
    }
    run_mode_indicator_csv = out_dir / "run_mode_indicator.csv"
    pd.DataFrame([run_mode_indicator]).to_csv(
        run_mode_indicator_csv, index=False, encoding="utf-8-sig"
    )
    print(f"Saved run mode indicator: {run_mode_indicator_csv}")

    if VALIDATION_ONLY_MODE:
        if INPUT_MODE != "video" or SEGMENTATION_PROFILE != "hsv_circle":
            raise ValueError(
                "VALIDATION_ONLY_MODE requires INPUT_MODE='video' and "
                "SEGMENTATION_PROFILE='hsv_circle'."
            )
        if USE_TEMPORAL_BODY_RECOVERY:
            save_temporal_lightweight_validation_clips(
                input_path,
                out_dir=out_dir,
                dish_center=DISH_CENTER,
                dish_radius=DISH_RADIUS,
                hsv_params=hsv_params,
                sections=VALIDATION_CLIP_SECTIONS,
                section_frames=VALIDATION_CLIP_SECTION_FRAMES,
                output_scale=VALIDATION_CLIP_OUTPUT_SCALE,
                stills_per_section=VALIDATION_CLIP_STILLS_PER_SECTION,
                write_video=VALIDATION_CLIP_WRITE_VIDEO,
                skip_edge_frames=TEMPORAL_SKIP_EDGE_FRAMES,
                temporal_window=TEMPORAL_WINDOW,
                temporal_min_hits=TEMPORAL_MIN_HITS,
                temporal_chroma_min=TEMPORAL_CHROMA_MIN,
                temporal_min_channel=TEMPORAL_MIN_CHANNEL,
                temporal_max_value=TEMPORAL_MAX_VALUE,
                temporal_rim_exclusion_px=TEMPORAL_RIM_EXCLUSION_PX,
                temporal_connect_radius=TEMPORAL_CONNECT_RADIUS,
                temporal_min_obj_size=TEMPORAL_MIN_OBJ_SIZE,
                temporal_exterior_only=TEMPORAL_EXTERIOR_ONLY,
                temporal_block_hole_max_area=TEMPORAL_BLOCK_HOLE_MAX_AREA,
                temporal_final_smooth_window=TEMPORAL_FINAL_SMOOTH_WINDOW,
                temporal_final_smooth_min_hits=TEMPORAL_FINAL_SMOOTH_MIN_HITS,
                n_jobs=VALIDATION_CLIP_CPU_WORKERS if ENABLE_CPU_PARALLEL else 1,
                batch_frames=VALIDATION_CLIP_BATCH_FRAMES,
            )
        else:
            save_lightweight_validation_clips(
                input_path,
                out_dir=out_dir,
                dish_center=DISH_CENTER,
                dish_radius=DISH_RADIUS,
                hsv_params=hsv_params,
                sections=VALIDATION_CLIP_SECTIONS,
                section_frames=VALIDATION_CLIP_SECTION_FRAMES,
                output_scale=VALIDATION_CLIP_OUTPUT_SCALE,
                stills_per_section=VALIDATION_CLIP_STILLS_PER_SECTION,
                n_jobs=VALIDATION_CLIP_CPU_WORKERS if ENABLE_CPU_PARALLEL else 1,
                batch_frames=VALIDATION_CLIP_BATCH_FRAMES,
                write_video=VALIDATION_CLIP_WRITE_VIDEO,
            )
        print("Validation-only run completed; full analysis was skipped.")
        raise SystemExit(0)

    if CLEAN_OUTPUT_DEBUG:
        clean_debug_outputs(out_dir, clean_segment_tests=True)

    load_as_gray = SEGMENTATION_PROFILE != "hsv_circle"
    first_frame, total_hint = _load_first_frame_for_preview(
        input_path,
        INPUT_MODE,
        as_gray=load_as_gray,
    )
    if INPUT_MODE == "folder":
        print(f"Found {total_hint} image frames. Showing first-frame preview before analysis.")
    else:
        total_text = str(total_hint) if total_hint > 0 else "unknown"
        if MAX_FRAMES is None:
            print(f"Video opened with {total_text} frames. Showing first-frame preview before analysis.")
        else:
            print(
                f"Video opened with {total_text} frames; analyzing up to {MAX_FRAMES}. "
                "Showing first-frame preview before analysis."
            )

    preview_first_frame(
        first_frame,
        segmentation_profile=SEGMENTATION_PROFILE,
        analysis_roi=ANALYSIS_ROI,
        dish_center=DISH_CENTER,
        dish_radius=DISH_RADIUS,
        hsv_params=hsv_params,
        preprocess_params=preprocess_params,
        segment_params=segment_params,
    )

    if RUN_SEGMENT_TESTS and INPUT_MODE == "video" and SEGMENTATION_PROFILE == "hsv_circle":
        save_segment_test_examples(
            input_path,
            out_dir / "segment_tests",
            frames_per_part=SEGMENT_TEST_FRAMES_PER_PART,
            dt=FRAME_INTERVAL,
            pixel_size=PIXEL_SIZE,
            dish_center=DISH_CENTER,
            dish_radius=DISH_RADIUS,
            hsv_params=hsv_params,
            vein_params=vein_params,
        )

    use_streaming_video = (
        INPUT_MODE == "video"
        and SEGMENTATION_PROFILE == "hsv_circle"
        and (LOW_MEMORY_VIDEO_STREAM or not ENABLE_CPU_PARALLEL)
    )

    if use_streaming_video:
        if ENABLE_CPU_PARALLEL and LOW_MEMORY_VIDEO_STREAM:
            print(
                "Low-memory batched video analysis is enabled; CPU workers will "
                "process a bounded number of frames at a time."
            )
        cap_probe = cv2.VideoCapture(str(input_path))
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()
        if MAX_FRAMES is not None:
            total_frames = min(total_frames, int(MAX_FRAMES))
        if total_frames <= 5:
            debug_indices = list(range(max(total_frames, 0)))
        else:
            debug_indices = sorted(
                set([0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1])
            )

        overlay_video = out_dir / "plasmodium_overlay.mp4" if SAVE_OVERLAY_VIDEO else None
        video_analysis_kwargs = dict(
            dt=FRAME_INTERVAL,
            pixel_size=PIXEL_SIZE,
            roi=ROI,
            max_frames=MAX_FRAMES,
            debug_frame_indices=debug_indices,
            segmentation_profile=SEGMENTATION_PROFILE,
            dish_center=DISH_CENTER,
            dish_radius=DISH_RADIUS,
            hsv_params=hsv_params,
            vein_params=vein_params,
            vein_every_n_frames=VEIN_EVERY_N_FRAMES,
            compute_body_skeleton=COMPUTE_BODY_SKELETON,
            compute_front_speed=COMPUTE_FRONT_SPEED,
            overlay_video_path=overlay_video,
            overlay_video_fps=OVERLAY_VIDEO_FPS,
        )
        if ENABLE_CPU_PARALLEL and LOW_MEMORY_VIDEO_STREAM:
            df, debug_data, osc = analyze_video_parallel_batches(
                input_path,
                **video_analysis_kwargs,
                n_jobs=CPU_WORKERS,
                batch_frames=PARALLEL_BATCH_FRAMES,
            )
        else:
            df, debug_data, osc = analyze_video_stream(
                input_path,
                **video_analysis_kwargs,
            )
        print(f"Analyzed {len(df)} frames.")
        if overlay_video is not None:
            print(f"Saved overlay video: {overlay_video}")
    else:
        if INPUT_MODE == "folder":
            frames, names = load_frames_from_folder(input_path, as_gray=load_as_gray)
        elif INPUT_MODE == "video":
            frames, names = load_frames_from_video(input_path, max_frames=MAX_FRAMES, as_gray=load_as_gray)
        else:
            raise ValueError('INPUT_MODE must be "folder" or "video".')

        if len(frames) == 0:
            raise ValueError("No frames were loaded.")

        print(f"Loaded {len(frames)} frames.")

        df, debug_data, osc = analyze_timelapse(
            frames,
            names=names,
            dt=FRAME_INTERVAL,
            pixel_size=PIXEL_SIZE,
            roi=ROI,
            analysis_roi=ANALYSIS_ROI,
            segmentation_profile=SEGMENTATION_PROFILE,
            dish_center=DISH_CENTER,
            dish_radius=DISH_RADIUS,
            hsv_params=hsv_params,
            vein_params=vein_params,
            vein_every_n_frames=VEIN_EVERY_N_FRAMES,
            compute_body_skeleton=COMPUTE_BODY_SKELETON,
            compute_front_speed=COMPUTE_FRONT_SPEED,
            preprocess_params=preprocess_params,
            segment_params=segment_params,
            keep_debug=True,
            n_jobs=CPU_WORKERS if ENABLE_CPU_PARALLEL else 1,
        )
        del frames, names
        gc.collect()

        if INPUT_MODE == "video" and SAVE_OVERLAY_VIDEO:
            overlay_video = out_dir / "plasmodium_overlay.mp4"
            save_overlay_video(
                debug_data,
                overlay_video,
                fps=OVERLAY_VIDEO_FPS,
                output_scale=OVERLAY_VIDEO_OUTPUT_SCALE,
            )
            print(f"Saved overlay video: {overlay_video}")

    if "run_mode" not in df.columns:
        df.insert(0, "run_mode", RUN_MODE)
    if "run_formal_analysis" not in df.columns:
        df.insert(1, "run_formal_analysis", bool(RUN_FORMAL_ANALYSIS))
    if "validation_sections" not in df.columns:
        df.insert(2, "validation_sections", ";".join(VALIDATION_CLIP_SECTIONS))

    metrics_csv = out_dir / "plasmodium_metrics.csv"
    df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"Saved metrics CSV: {metrics_csv}")

    if RUN_LIGHTWEIGHT_VALIDATION_CLIPS:
        if INPUT_MODE == "video" and SEGMENTATION_PROFILE == "hsv_circle":
            if USE_TEMPORAL_BODY_RECOVERY:
                save_temporal_lightweight_validation_clips(
                    input_path,
                    out_dir=out_dir,
                    dish_center=DISH_CENTER,
                    dish_radius=DISH_RADIUS,
                    hsv_params=hsv_params,
                    sections=VALIDATION_CLIP_SECTIONS,
                    section_frames=VALIDATION_CLIP_SECTION_FRAMES,
                    output_scale=VALIDATION_CLIP_OUTPUT_SCALE,
                    stills_per_section=VALIDATION_CLIP_STILLS_PER_SECTION,
                    write_video=VALIDATION_CLIP_WRITE_VIDEO,
                    skip_edge_frames=TEMPORAL_SKIP_EDGE_FRAMES,
                    temporal_window=TEMPORAL_WINDOW,
                    temporal_min_hits=TEMPORAL_MIN_HITS,
                    temporal_chroma_min=TEMPORAL_CHROMA_MIN,
                    temporal_min_channel=TEMPORAL_MIN_CHANNEL,
                    temporal_max_value=TEMPORAL_MAX_VALUE,
                    temporal_rim_exclusion_px=TEMPORAL_RIM_EXCLUSION_PX,
                    temporal_connect_radius=TEMPORAL_CONNECT_RADIUS,
                    temporal_min_obj_size=TEMPORAL_MIN_OBJ_SIZE,
                    temporal_exterior_only=TEMPORAL_EXTERIOR_ONLY,
                    temporal_block_hole_max_area=TEMPORAL_BLOCK_HOLE_MAX_AREA,
                    temporal_final_smooth_window=TEMPORAL_FINAL_SMOOTH_WINDOW,
                    temporal_final_smooth_min_hits=TEMPORAL_FINAL_SMOOTH_MIN_HITS,
                    n_jobs=VALIDATION_CLIP_CPU_WORKERS if ENABLE_CPU_PARALLEL else 1,
                    batch_frames=VALIDATION_CLIP_BATCH_FRAMES,
                )
            else:
                save_lightweight_validation_clips(
                    input_path,
                    out_dir=out_dir,
                    dish_center=DISH_CENTER,
                    dish_radius=DISH_RADIUS,
                    hsv_params=hsv_params,
                    sections=VALIDATION_CLIP_SECTIONS,
                    section_frames=VALIDATION_CLIP_SECTION_FRAMES,
                    output_scale=VALIDATION_CLIP_OUTPUT_SCALE,
                    stills_per_section=VALIDATION_CLIP_STILLS_PER_SECTION,
                    n_jobs=VALIDATION_CLIP_CPU_WORKERS if ENABLE_CPU_PARALLEL else 1,
                    batch_frames=VALIDATION_CLIP_BATCH_FRAMES,
                    write_video=VALIDATION_CLIP_WRITE_VIDEO,
                )
        else:
            warnings.warn(
                "Lightweight validation clips currently require video input "
                "with SEGMENTATION_PROFILE='hsv_circle'."
            )
    if RUN_RELIABILITY_VALIDATION:
        if INPUT_MODE == "video" and SEGMENTATION_PROFILE == "hsv_circle":
            validate_segmentation_reliability(
                input_path,
                metrics_df=df,
                out_dir=out_dir,
                dish_center=DISH_CENTER,
                dish_radius=DISH_RADIUS,
                hsv_params=hsv_params,
                vein_params=vein_params,
                pixel_size=PIXEL_SIZE,
                sample_count=RELIABILITY_SAMPLE_COUNT,
                top_area_jump_count=RELIABILITY_TOP_AREA_JUMPS,
                validate_veins=RELIABILITY_VALIDATE_VEINS,
            )
        else:
            warnings.warn(
                "Reliability validation currently requires video input with "
                "SEGMENTATION_PROFILE='hsv_circle'."
            )

    summary_plot = out_dir / "plasmodium_summary.png"
    plot_summary(df, out_prefix=summary_plot)
    plt.show()
    plt.close("all")
    gc.collect()
    print(f"Saved summary plot: {summary_plot}")

    debug_out = out_dir / "debug_examples"
    if len(debug_data) > 0:
        if len(debug_data) <= 5:
            debug_indices_to_save = list(range(len(debug_data)))
        else:
            debug_indices_to_save = sorted(
                set([0, len(debug_data) // 4, len(debug_data) // 2, 3 * len(debug_data) // 4, len(debug_data) - 1])
            )
        save_debug_examples(debug_data, debug_out, debug_indices_to_save)
    print(f"Saved debug examples: {debug_out}")

    if RUN_QC_VALIDATION:
        qc_df = validate_analysis_qc(
            df,
            debug_data,
            out_dir=out_dir,
            analysis_roi=None,
            area_jump_threshold=0.35,
            save_review_images=True,
        )
        if MANUAL_MASK_DIR is not None:
            validate_against_manual_masks(
                debug_data,
                manual_mask_dir=MANUAL_MASK_DIR,
                out_csv=out_dir / "manual_mask_validation.csv",
            )

    summary_df = summarize_run(df, label=input_path.stem)
    summary_df.insert(0, "run_mode", RUN_MODE)
    summary_df.insert(1, "validation_sections", ";".join(VALIDATION_CLIP_SECTIONS))
    summary_df["osc_dominant_freq"] = osc.get("dominant_freq", np.nan)
    summary_df["osc_period"] = osc.get("period", np.nan)
    summary_df["osc_n_peaks"] = osc.get("n_peaks", 0)

    summary_csv = out_dir / "plasmodium_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary CSV: {summary_csv}")
    print(summary_df.to_string(index=False))
