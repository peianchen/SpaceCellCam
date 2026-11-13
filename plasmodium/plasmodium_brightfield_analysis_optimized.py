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
from concurrent.futures.process import BrokenProcessPool
import gc
import os
import re
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


def _as_rgb_uint8(img):
    """Convert an RGB/grayscale image array to 3-channel RGB uint8."""
    if img is None:
        raise ValueError("Input image is None.")

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    if hi <= lo:
        lo = np.nanmin(arr[finite])
        hi = np.nanmax(arr[finite])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return np.ascontiguousarray((arr * 255).astype(np.uint8))


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


CONTRACTION_PROXY_NOTE = (
    "The contraction signal is an optical intensity proxy associated with "
    "local thickness, density, flow, or light-scattering variation. It is not "
    "a direct measurement of contraction force or cytoplasmic flow."
)


def parse_contraction_rois(rois):
    """
    Normalize contraction ROI definitions.

    Accepted formats are (name, x, y, w, h) and (x, y, w, h).
    Already-normalized dictionaries are also accepted for internal call paths.
    """
    if rois is None:
        return []

    parsed = []
    used_names = set()
    for index, roi in enumerate(rois):
        if isinstance(roi, dict):
            missing = {"name", "x", "y", "w", "h"} - set(roi)
            if missing:
                raise ValueError(
                    f"Contraction ROI {index} is missing fields: {sorted(missing)}"
                )
            raw_name = roi["name"]
            coords = (roi["x"], roi["y"], roi["w"], roi["h"])
        elif isinstance(roi, (tuple, list)) and len(roi) == 4:
            raw_name = f"roi_{index}"
            coords = roi
        elif isinstance(roi, (tuple, list)) and len(roi) == 5:
            raw_name = roi[0]
            coords = roi[1:]
        else:
            raise ValueError(
                f"Contraction ROI {index} must be (x, y, w, h) or "
                f"(name, x, y, w, h); received {roi!r}."
            )

        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw_name)).strip("_")
        if not safe_name:
            raise ValueError(f"Contraction ROI {index} has an empty safe name.")
        if safe_name in used_names:
            raise ValueError(f"Duplicate contraction ROI name: {safe_name!r}.")

        try:
            x, y, w, h = [int(value) for value in coords]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Contraction ROI {safe_name!r} coordinates must be integers."
            ) from exc
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Contraction ROI {safe_name!r} must have positive width and height."
            )

        parsed.append({"name": safe_name, "x": x, "y": y, "w": w, "h": h})
        used_names.add(safe_name)
    return parsed


def _safe_contraction_roi_slices(shape, roi_def):
    """Clip a parsed contraction ROI to an image, or return (None, None)."""
    height, width = int(shape[0]), int(shape[1])
    x = int(roi_def["x"])
    y = int(roi_def["y"])
    w = int(roi_def["w"])
    h = int(roi_def["h"])
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None, None
    return slice(y0, y1), slice(x0, x1)


def contraction_roi_intensity_stats(
    image,
    roi_def,
    body_mask=None,
    min_pixels=50,
):
    """Calculate scalar intensity statistics inside one contraction ROI."""
    gray = _as_gray_uint8(image)
    ys, xs = _safe_contraction_roi_slices(gray.shape, roi_def)
    empty = {
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "p05": np.nan,
        "p95": np.nan,
        "n_pixels": 0,
        "valid": False,
    }
    if ys is None:
        return empty

    roi_values = gray[ys, xs]
    if body_mask is not None:
        body_mask = np.asarray(body_mask, dtype=bool)
        if body_mask.shape != gray.shape:
            raise ValueError(
                "Contraction body mask shape must match the intensity image."
            )
        roi_values = roi_values[body_mask[ys, xs]]
    else:
        roi_values = roi_values.reshape(-1)

    n_pixels = int(roi_values.size)
    if n_pixels < max(1, int(min_pixels)):
        return {**empty, "n_pixels": n_pixels}

    values = roi_values.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "n_pixels": n_pixels,
        "valid": True,
    }


def select_contraction_intensity_image(source, gray, enhanced, pre_debug):
    """
    Select the contraction intensity image.

    Gray is the reproducible default. Enhanced images may include normalization
    or CLAHE, so their absolute amplitudes should not be compared across videos.
    Corrected-image meaning varies with the segmentation profile.
    """
    source = str(source).lower()
    if source == "gray":
        return gray
    if source == "enhanced":
        return enhanced
    if source == "corrected":
        return pre_debug.get("corrected", gray)
    raise ValueError(
        "contraction_intensity_source must be 'gray', 'enhanced', or 'corrected'."
    )


def select_contraction_mask(
    mask_layer,
    clean_mask,
    occupied_mask,
    envelope_mask,
):
    """Select the body layer used to constrain contraction intensity."""
    layers = {
        "clean": clean_mask,
        "occupied": occupied_mask,
        "envelope": envelope_mask,
        "none": None,
    }
    key = str(mask_layer).lower()
    if key not in layers:
        raise ValueError(
            "contraction_mask_layer must be 'clean', 'occupied', 'envelope', or 'none'."
        )
    return layers[key]


def add_contraction_metrics_to_row(
    row,
    intensity_image,
    contraction_rois,
    contraction_mask=None,
    min_pixels=50,
):
    """Add scalar per-ROI contraction intensity values to one frame row."""
    for roi_def in contraction_rois:
        stats = contraction_roi_intensity_stats(
            intensity_image,
            roi_def,
            body_mask=contraction_mask,
            min_pixels=min_pixels,
        )
        prefix = f"contraction_{roi_def['name']}"
        for field in ("mean", "median", "std", "p05", "p95", "n_pixels", "valid"):
            row[f"{prefix}_{field}"] = stats[field]
    return row


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
    """Recover a conservative vesselness mask near an existing body mask."""
    body_mask = np.asarray(body_mask, dtype=bool)
    empty = np.zeros_like(body_mask, dtype=bool)
    if frame is None or not np.any(body_mask):
        return empty, empty, np.nan, np.nan
    if frame.ndim == 2:
        bgr = cv2.cvtColor(_as_gray_uint8(frame), cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame

    dish_mask = circular_roi_mask(body_mask.shape, dish_center, dish_radius)
    radius = max(1, int(connect_radius))
    near_body = morphology.binary_dilation(body_mask, morphology.disk(radius)) & dish_mask
    search_mask = near_body & (~body_mask)
    if not np.any(search_mask):
        return empty, empty, np.nan, np.nan

    ys, xs = np.where(near_body)
    sigma_min = max(float(sigma_min), 0.5)
    sigma_max = max(float(sigma_max), sigma_min)
    pad = max(12, int(np.ceil(sigma_max * 5)))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(body_mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(body_mask.shape[1], int(xs.max()) + pad + 1)

    gray_crop = _as_gray_uint8(bgr[y0:y1, x0:x1]).astype(np.float32) / 255.0
    near_crop = near_body[y0:y1, x0:x1]
    work = exposure.equalize_adapthist(gray_crop, clip_limit=0.02)
    fill_value = float(np.nanmedian(work[near_crop])) if np.any(near_crop) else float(np.nanmedian(work))
    work[~near_crop] = fill_value
    sigmas = np.linspace(
        sigma_min,
        sigma_max,
        num=max(2, int(np.ceil(sigma_max - sigma_min)) + 1),
    )
    response_crop = filters.frangi(
        work,
        sigmas=sigmas,
        black_ridges=bool(black_ridges),
    )
    response_crop = np.nan_to_num(response_crop, nan=0.0, posinf=0.0, neginf=0.0)
    response = np.zeros(body_mask.shape, dtype=float)
    response[y0:y1, x0:x1] = response_crop

    threshold = float(np.nanpercentile(response[search_mask], float(percentile)))
    threshold = max(threshold, float(min_response))
    response_mean = float(np.nanmean(response[search_mask]))

    blue, green, red = cv2.split(bgr.astype(np.int16))
    yellow_chroma = np.minimum(red - blue, green - blue)
    min_rgb = np.minimum(np.minimum(red, green), blue)
    max_rgb = np.maximum(np.maximum(red, green), blue)
    color_gate = (
        (yellow_chroma >= int(chroma_min))
        & (min_rgb >= int(min_channel))
        & (max_rgb <= int(max_value))
        & dish_mask
    )
    candidate = (response >= threshold) & search_mask & color_gate
    if min_obj_size and int(min_obj_size) > 0:
        candidate = morphology.remove_small_objects(
            candidate.astype(bool),
            min_size=int(min_obj_size),
        )

    max_added = int(round(np.sum(body_mask) * float(max_added_fraction)))
    if max_added > 0 and int(np.sum(candidate)) > max_added:
        labeled, n_labels = ndi.label(candidate)
        component_scores = []
        for label_id in range(1, n_labels + 1):
            component = labeled == label_id
            component_scores.append(
                (float(np.nanmean(response[component])), int(np.sum(component)), label_id)
            )
        component_scores.sort(reverse=True)
        capped = np.zeros_like(candidate, dtype=bool)
        total_added = 0
        for _score, size, label_id in component_scores:
            if total_added > 0 and total_added + size > max_added:
                continue
            component = labeled == label_id
            capped |= component
            total_added += size
            if total_added >= max_added:
                break
        candidate = capped

    added = candidate & (~body_mask)
    return added.astype(bool), candidate.astype(bool), threshold, response_mean
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


    vesselness_candidate_mask = np.zeros_like(seed_mask, dtype=bool)
    vesselness_added_mask = np.zeros_like(seed_mask, dtype=bool)
    vesselness_threshold = np.nan
    vesselness_response_mean = np.nan
    if use_vesselness_recovery and np.any(occupied_mask):
        (
            vesselness_added_mask,
            vesselness_candidate_mask,
            vesselness_threshold,
            vesselness_response_mean,
        ) = recover_vesselness_near_body(
            bgr,
            occupied_mask,
            dish_center=dish_center,
            dish_radius=dish_radius,
            percentile=vesselness_percentile,
            min_response=vesselness_min_response,
            connect_radius=vesselness_connect_radius,
            min_obj_size=vesselness_min_obj_size,
            max_added_fraction=vesselness_max_added_fraction,
            chroma_min=vesselness_chroma_min,
            min_channel=vesselness_min_channel,
            max_value=vesselness_max_value,
            sigma_min=vesselness_sigma_min,
            sigma_max=vesselness_sigma_max,
            black_ridges=vesselness_black_ridges,
        )
        occupied_mask = (occupied_mask | vesselness_added_mask).astype(bool)
    if keep_largest:
        occupied_mask = keep_largest_component(occupied_mask)
        seed_mask = seed_mask & occupied_mask
        yellow_chroma_added_mask = yellow_chroma_added_mask & occupied_mask
        yellow_chroma_candidate = yellow_chroma_candidate & occupied_mask
        relaxed_added_mask = relaxed_added_mask & occupied_mask
        relaxed_candidate = relaxed_candidate & occupied_mask
        vesselness_added_mask = vesselness_added_mask & occupied_mask
        vesselness_candidate_mask = vesselness_candidate_mask & occupied_mask

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
    vesselness_added_area_px = int(np.sum(vesselness_added_mask))
    occupied_area_px = int(np.sum(occupied_mask))
    yellow_chroma_added_fraction = float(
        yellow_chroma_added_area_px / max(occupied_area_px, 1)
    )
    relaxed_added_fraction = float(relaxed_added_area_px / max(occupied_area_px, 1))
    vesselness_added_fraction = float(vesselness_added_area_px / max(occupied_area_px, 1))

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
        "vesselness_candidate_mask": vesselness_candidate_mask,
        "vesselness_added_mask": vesselness_added_mask,
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
        "vesselness_added_area_px": vesselness_added_area_px,
        "vesselness_added_fraction": vesselness_added_fraction,
        "use_vesselness_recovery": bool(use_vesselness_recovery),
        "vesselness_threshold": vesselness_threshold,
        "vesselness_response_mean": vesselness_response_mean,
        "vesselness_percentile": float(vesselness_percentile),
        "vesselness_connect_radius": int(vesselness_connect_radius),
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


def _interpolate_short_nan_gaps(signal, max_gap):
    """Interpolate only bounded NaN runs no longer than max_gap."""
    values = np.asarray(signal, dtype=np.float64).copy()
    max_gap = max(0, int(max_gap or 0))
    if max_gap == 0 or values.size < 3:
        return values

    index = 0
    while index < values.size:
        if np.isfinite(values[index]):
            index += 1
            continue
        start = index
        while index < values.size and not np.isfinite(values[index]):
            index += 1
        stop = index
        gap = stop - start
        if (
            gap <= max_gap
            and start > 0
            and stop < values.size
            and np.isfinite(values[start - 1])
            and np.isfinite(values[stop])
        ):
            values[start:stop] = np.interp(
                np.arange(start, stop),
                [start - 1, stop],
                [values[start - 1], values[stop]],
            )
    return values


def _longest_finite_segment(values):
    """Return start and stop indices of the longest contiguous finite segment."""
    finite = np.isfinite(values)
    best_start = best_stop = 0
    index = 0
    while index < finite.size:
        if not finite[index]:
            index += 1
            continue
        start = index
        while index < finite.size and finite[index]:
            index += 1
        if index - start > best_stop - best_start:
            best_start, best_stop = start, index
    return best_start, best_stop


def analyze_oscillation(
    signal,
    dt,
    min_period_sec=None,
    max_period_sec=None,
    interpolate_short_gaps=True,
    max_interpolation_gap=3,
):
    """Analyze a detrended intensity signal within an optional period range."""
    sig = np.asarray(signal, dtype=np.float64)
    original_valid = np.isfinite(sig)
    valid_count = int(np.sum(original_valid))
    valid_fraction = float(valid_count / sig.size) if sig.size else 0.0
    empty = {
        "dominant_freq": np.nan,
        "period": np.nan,
        "n_peaks": 0,
        "detrended_signal": np.full(sig.shape, np.nan, dtype=np.float64),
        "fft_freq": np.array([], dtype=np.float64),
        "fft_power": np.array([], dtype=np.float64),
        "amplitude_peak_to_peak": np.nan,
        "amplitude_std": np.nan,
        "valid_sample_count": valid_count,
        "valid_fraction": valid_fraction,
        "analyzed_sample_count": 0,
        "quality_flag": "insufficient_valid_data",
    }
    if dt is None or not np.isfinite(dt) or dt <= 0 or valid_count < 4:
        return empty

    prepared = (
        _interpolate_short_nan_gaps(sig, max_interpolation_gap)
        if interpolate_short_gaps
        else sig.copy()
    )
    start, stop = _longest_finite_segment(prepared)
    if stop - start < 4:
        return empty

    segment = prepared[start:stop]
    segment_detrended = detrend(segment)
    detrended_full = np.full(sig.shape, np.nan, dtype=np.float64)
    detrended_full[start:stop] = segment_detrended

    fft_values = np.fft.rfft(segment_detrended)
    frequencies = np.fft.rfftfreq(segment_detrended.size, d=float(dt))
    power = np.abs(fft_values) ** 2

    frequency_ok = frequencies > 0
    if max_period_sec is not None:
        if not np.isfinite(max_period_sec) or max_period_sec <= 0:
            raise ValueError("max_period_sec must be positive.")
        frequency_ok &= frequencies >= 1.0 / float(max_period_sec)
    if min_period_sec is not None:
        if not np.isfinite(min_period_sec) or min_period_sec <= 0:
            raise ValueError("min_period_sec must be positive.")
        frequency_ok &= frequencies <= 1.0 / float(min_period_sec)

    if np.any(frequency_ok):
        candidate_indices = np.flatnonzero(frequency_ok)
        dominant_index = candidate_indices[int(np.argmax(power[frequency_ok]))]
        dominant_freq = float(frequencies[dominant_index])
        period = float(1.0 / dominant_freq)
    else:
        dominant_freq = np.nan
        period = np.nan

    prominence = float(np.std(segment_detrended) * 0.5)
    peak_kwargs = {}
    if np.isfinite(prominence) and prominence > 0:
        peak_kwargs["prominence"] = prominence
    if min_period_sec is not None:
        peak_kwargs["distance"] = max(
            1, int(np.ceil(float(min_period_sec) / float(dt)))
        )
    peaks, _ = find_peaks(segment_detrended, **peak_kwargs)

    duration_sec = float(max(0, segment.size - 1) * float(dt))
    insufficient_duration = (
        max_period_sec is not None
        and duration_sec < 2.0 * float(max_period_sec)
    )
    quality_flag = (
        "insufficient_duration_for_two_max_period_cycles"
        if insufficient_duration
        else "ok"
    )

    return {
        "dominant_freq": dominant_freq,
        "period": period,
        "n_peaks": int(len(peaks)),
        "detrended_signal": detrended_full,
        "fft_freq": frequencies,
        "fft_power": power,
        "amplitude_peak_to_peak": float(
            np.percentile(segment_detrended, 95)
            - np.percentile(segment_detrended, 5)
        ),
        "amplitude_std": float(np.std(segment_detrended)),
        "valid_sample_count": valid_count,
        "valid_fraction": valid_fraction,
        "analyzed_sample_count": int(segment.size),
        "quality_flag": quality_flag,
    }

def _plot_contraction_roi(
    roi_name,
    time_values,
    raw_signal,
    oscillation,
    summary_row,
    out_path,
):
    """Save raw, detrended, and FFT panels for one contraction ROI."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(time_values, raw_signal, color="tab:blue", linewidth=1.0)
    axes[0].set_ylabel("Mean intensity")
    axes[0].set_title(f"Contraction optical proxy: {roi_name}")
    axes[0].grid(alpha=0.25)

    detrended_signal = oscillation["detrended_signal"]
    if detrended_signal.size == time_values.size:
        axes[1].plot(
            time_values,
            detrended_signal,
            color="tab:orange",
            linewidth=1.0,
        )
    axes[1].set_ylabel("Detrended intensity")
    axes[1].grid(alpha=0.25)

    frequencies = oscillation["fft_freq"]
    power = oscillation["fft_power"]
    positive = frequencies > 0 if frequencies.size else np.array([], dtype=bool)
    if frequencies.size and np.any(positive):
        axes[2].plot(frequencies[positive], power[positive], color="tab:green")
    dominant_freq = summary_row["dominant_freq_hz"]
    if np.isfinite(dominant_freq):
        axes[2].axvline(
            dominant_freq,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
        )
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("FFT power")
    axes[2].grid(alpha=0.25)

    annotation = (
        f"period={summary_row['period_sec']:.3g} s, "
        f"frequency={summary_row['dominant_freq_hz']:.3g} Hz\n"
        f"valid={summary_row['valid']}, n_valid={summary_row['n_valid']}, "
        f"cycles={summary_row['estimated_cycles']:.3g}\n"
        f"reason={summary_row['invalid_reason']}"
    )
    fig.text(0.02, 0.01, annotation, fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze_contraction_rois(
    df,
    contraction_rois,
    dt,
    min_period_sec=30.0,
    max_period_sec=600.0,
    min_valid_samples=20,
    interpolate_short_gaps=True,
    max_interpolation_gap=3,
    out_dir=None,
    save_plots=True,
):
    """
    Summarize multi-ROI bright-field intensity oscillations.

    These values are optical proxies associated with local thickness, density,
    flow, or light scattering. They are not direct force or flow measurements.
    """
    contraction_rois = parse_contraction_rois(contraction_rois)
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("A positive acquisition interval dt is required.")

    n_frames = int(len(df))
    if "time" in df.columns:
        time_values = pd.to_numeric(df["time"], errors="coerce").to_numpy(
            dtype=float
        )
    else:
        time_values = np.arange(n_frames, dtype=float) * float(dt)
    observation_duration = float(max(0, n_frames - 1) * float(dt))
    rows = []

    for roi_def in contraction_rois:
        roi_name = roi_def["name"]
        signal_col = f"contraction_{roi_name}_mean"
        pixels_col = f"contraction_{roi_name}_n_pixels"
        missing_column = signal_col not in df.columns

        if missing_column:
            raw_signal = np.full(n_frames, np.nan, dtype=float)
        else:
            raw_signal = pd.to_numeric(
                df[signal_col], errors="coerce"
            ).to_numpy(dtype=float)

        oscillation = analyze_oscillation(
            raw_signal,
            dt,
            min_period_sec=min_period_sec,
            max_period_sec=max_period_sec,
            interpolate_short_gaps=interpolate_short_gaps,
            max_interpolation_gap=max_interpolation_gap,
        )
        finite = np.isfinite(raw_signal)
        n_valid = int(np.sum(finite))
        valid_fraction = float(n_valid / n_frames) if n_frames else 0.0
        mean_intensity = (
            float(np.mean(raw_signal[finite])) if np.any(finite) else np.nan
        )
        std_intensity = (
            float(np.std(raw_signal[finite])) if np.any(finite) else np.nan
        )
        if pixels_col in df.columns:
            pixel_values = pd.to_numeric(
                df[pixels_col], errors="coerce"
            ).to_numpy(dtype=float)
            mean_valid_pixels = (
                float(np.nanmean(pixel_values))
                if np.isfinite(pixel_values).any()
                else np.nan
            )
        else:
            mean_valid_pixels = np.nan

        period_sec = oscillation["period"]
        dominant_freq = oscillation["dominant_freq"]
        estimated_cycles = (
            float(observation_duration / period_sec)
            if np.isfinite(period_sec) and period_sec > 0
            else np.nan
        )
        amplitude = oscillation["amplitude_peak_to_peak"]

        if missing_column:
            invalid_reason = "missing_column"
        elif n_valid < int(min_valid_samples):
            invalid_reason = "too_few_valid_samples"
        elif valid_fraction < 0.8:
            invalid_reason = "low_valid_fraction"
        elif not np.isfinite(dominant_freq):
            invalid_reason = "no_frequency_in_range"
        elif (
            period_sec < float(min_period_sec)
            or period_sec > float(max_period_sec)
        ):
            invalid_reason = "no_frequency_in_range"
        elif not np.isfinite(amplitude) or amplitude <= 0:
            invalid_reason = "zero_amplitude"
        elif not np.isfinite(estimated_cycles) or estimated_cycles < 2.0:
            invalid_reason = "too_few_cycles"
        else:
            invalid_reason = "ok"

        summary_row = {
            "roi_name": roi_name,
            "n_frames": n_frames,
            "n_valid": n_valid,
            "valid_fraction": valid_fraction,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "mean_valid_pixels": mean_valid_pixels,
            "amplitude_peak_to_peak": amplitude,
            "amplitude_std": oscillation["amplitude_std"],
            "dominant_freq_hz": dominant_freq,
            "period_sec": period_sec,
            "period_min": (
                float(period_sec / 60.0) if np.isfinite(period_sec) else np.nan
            ),
            "n_peaks": oscillation["n_peaks"],
            "sampling_interval_sec": float(dt),
            "observation_duration_sec": observation_duration,
            "estimated_cycles": estimated_cycles,
            "valid": invalid_reason == "ok",
            "invalid_reason": invalid_reason,
            "frequency_quality_flag": oscillation["quality_flag"],
            "signal_interpretation": CONTRACTION_PROXY_NOTE,
        }
        rows.append(summary_row)

        if out_dir is not None and save_plots:
            _plot_contraction_roi(
                roi_name,
                time_values,
                raw_signal,
                oscillation,
                summary_row,
                Path(out_dir)
                / "contraction_plots"
                / f"contraction_{roi_name}.png",
            )

    summary = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "contraction_summary.csv", index=False)
    return summary


def plot_contraction_summary(contraction_summary, out_path):
    """Plot ROI period and optical-proxy amplitude with validity status."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    if contraction_summary is None or contraction_summary.empty:
        for axis in axes:
            axis.axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No contraction ROIs were analyzed.",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    else:
        names = contraction_summary["roi_name"].astype(str).tolist()
        valid = contraction_summary["valid"].astype(bool).to_numpy()
        colors = np.where(valid, "tab:green", "tab:gray")
        periods = pd.to_numeric(
            contraction_summary["period_sec"], errors="coerce"
        ).to_numpy(dtype=float)
        amplitudes = pd.to_numeric(
            contraction_summary["amplitude_peak_to_peak"], errors="coerce"
        ).to_numpy(dtype=float)
        x_values = np.arange(len(names))

        axes[0].bar(x_values, periods, color=colors)
        axes[0].set_ylabel("Dominant period (s)")
        axes[0].set_xticks(x_values, names, rotation=30, ha="right")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(x_values, amplitudes, color=colors)
        axes[1].set_ylabel("Optical amplitude (gray level)")
        axes[1].set_xticks(x_values, names, rotation=30, ha="right")
        axes[1].grid(axis="y", alpha=0.25)
        axes[0].set_title("Green=valid, gray=invalid")

    fig.suptitle("Contraction intensity proxy summary")
    fig.text(
        0.01,
        0.005,
        "Optical proxy only; not a direct measurement of force or cytoplasmic flow.",
        fontsize=8,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def _draw_contraction_rois(image, contraction_rois, color, thickness=2):
    """Draw clipped contraction ROI rectangles and names in-place."""
    for roi_def in parse_contraction_rois(contraction_rois):
        ys, xs = _safe_contraction_roi_slices(image.shape, roi_def)
        if ys is None:
            continue
        x0, x1 = int(xs.start), int(xs.stop)
        y0, y1 = int(ys.start), int(ys.stop)
        cv2.rectangle(
            image,
            (x0, y0),
            (max(x0, x1 - 1), max(y0, y1 - 1)),
            color,
            int(thickness),
        )
        text_y = min(image.shape[0] - 3, max(12, y0 + 14))
        cv2.putText(
            image,
            roi_def["name"],
            (x0 + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return image

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
    contraction_rois=None,
    draw_contraction_rois=False,
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

    output = np.clip(overlay, 0, 255).astype(np.uint8)
    if draw_contraction_rois and contraction_rois:
        _draw_contraction_rois(
            output,
            contraction_rois,
            color=(215, 170, 255),
            thickness=2,
        )
    return output


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
        compute_contraction,
        contraction_rois,
        contraction_intensity_source,
        contraction_mask_layer,
        contraction_min_body_pixels,
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

    if compute_contraction and contraction_rois:
        contraction_image = select_contraction_intensity_image(
            contraction_intensity_source,
            gray,
            enhanced,
            pre_debug,
        )
        contraction_mask = select_contraction_mask(
            contraction_mask_layer,
            clean_mask=mask,
            occupied_mask=occupied_mask,
            envelope_mask=envelope_mask,
        )
        add_contraction_metrics_to_row(
            row,
            contraction_image,
            contraction_rois,
            contraction_mask=contraction_mask,
            min_pixels=contraction_min_body_pixels,
        )
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
    compute_contraction=False,
    contraction_rois=None,
    contraction_intensity_source="gray",
    contraction_mask_layer="clean",
    contraction_min_body_pixels=50,
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
    contraction_rois = parse_contraction_rois(contraction_rois)

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
                compute_contraction,
                contraction_rois,
                contraction_intensity_source,
                contraction_mask_layer,
                contraction_min_body_pixels,
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

        if compute_contraction and contraction_rois:
            contraction_image = select_contraction_intensity_image(
                contraction_intensity_source,
                gray,
                enhanced,
                pre_debug,
            )
            contraction_mask = select_contraction_mask(
                contraction_mask_layer,
                clean_mask=mask,
                occupied_mask=occupied_mask,
                envelope_mask=envelope_mask,
            )
            add_contraction_metrics_to_row(
                row,
                contraction_image,
                contraction_rois,
                contraction_mask=contraction_mask,
                min_pixels=contraction_min_body_pixels,
            )
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
    compute_contraction=False,
    contraction_rois=None,
    contraction_intensity_source="gray",
    contraction_mask_layer="clean",
    contraction_min_body_pixels=50,
    draw_contraction_rois=False,
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
    contraction_rois = parse_contraction_rois(contraction_rois)

    writer = None
    if overlay_video_path is not None:
        overlay_video_path = Path(overlay_video_path)
        overlay_video_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    debug_data = []
    prev_mask = None
    prev_area = np.nan
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

            if compute_contraction and contraction_rois:
                contraction_image = select_contraction_intensity_image(
                    contraction_intensity_source,
                    gray,
                    enhanced,
                    pre_debug,
                )
                contraction_mask = select_contraction_mask(
                    contraction_mask_layer,
                    clean_mask=mask,
                    occupied_mask=occupied_mask,
                    envelope_mask=envelope_mask,
                )
                add_contraction_metrics_to_row(
                    row,
                    contraction_image,
                    contraction_rois,
                    contraction_mask=contraction_mask,
                    min_pixels=contraction_min_body_pixels,
                )
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
                    contraction_rois=contraction_rois,
                    draw_contraction_rois=draw_contraction_rois,
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
    compute_contraction=False,
    contraction_rois=None,
    contraction_intensity_source="gray",
    contraction_mask_layer="clean",
    contraction_min_body_pixels=50,
    draw_contraction_rois=False,
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
    contraction_rois = parse_contraction_rois(contraction_rois)
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
                                compute_contraction,
                                contraction_rois,
                                contraction_intensity_source,
                                contraction_mask_layer,
                                contraction_min_body_pixels,
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
                                contraction_rois=contraction_rois,
                                draw_contraction_rois=draw_contraction_rois,
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




def _safe_overlay_output_scale(output_scale):
    """Return a safe overlay video scale in the 0.1--1.0 range."""
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
    return float(np.clip(output_scale, 0.1, 1.0))




def _safe_formal_frame_scale(frame_scale):
    """Return a safe formal-analysis frame scale in the 0.1--1.0 range."""
    try:
        frame_scale = float(frame_scale)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid frame_scale={frame_scale!r}; falling back to 1.0."
        )
        frame_scale = 1.0
    if not np.isfinite(frame_scale):
        warnings.warn(
            f"Invalid frame_scale={frame_scale!r}; falling back to 1.0."
        )
        frame_scale = 1.0
    return float(np.clip(frame_scale, 0.1, 1.0))


def _resize_frame_for_formal_analysis(frame, frame_scale):
    """Resize one frame for faster formal analysis."""
    frame_scale = _safe_formal_frame_scale(frame_scale)
    if frame_scale >= 0.999:
        return frame
    h, w = frame.shape[:2]
    out_w = max(1, int(round(w * frame_scale)))
    out_h = max(1, int(round(h * frame_scale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _scale_point_for_formal_analysis(point, frame_scale):
    """Scale an x, y point for resized formal-analysis frames."""
    if point is None:
        return None
    frame_scale = _safe_formal_frame_scale(frame_scale)
    x, y = point
    return (int(round(float(x) * frame_scale)), int(round(float(y) * frame_scale)))


def _scale_radius_for_formal_analysis(radius, frame_scale):
    """Scale a circular ROI radius for resized formal-analysis frames."""
    if radius is None:
        return None
    frame_scale = _safe_formal_frame_scale(frame_scale)
    return max(1, int(round(float(radius) * frame_scale)))


def _scale_roi_for_formal_analysis(roi, frame_scale):
    """Scale an x, y, w, h ROI for resized formal-analysis frames."""
    if roi is None:
        return None
    frame_scale = _safe_formal_frame_scale(frame_scale)
    x, y, w, h = roi
    return (
        int(round(float(x) * frame_scale)),
        int(round(float(y) * frame_scale)),
        max(1, int(round(float(w) * frame_scale))),
        max(1, int(round(float(h) * frame_scale))),
    )


def _scale_contraction_rois_for_formal_analysis(contraction_rois, frame_scale):
    """Scale parsed contraction ROIs for resized formal-analysis frames."""
    frame_scale = _safe_formal_frame_scale(frame_scale)
    scaled = []
    for roi_def in parse_contraction_rois(contraction_rois):
        scaled.append(
            {
                "name": roi_def["name"],
                "x": int(round(float(roi_def["x"]) * frame_scale)),
                "y": int(round(float(roi_def["y"]) * frame_scale)),
                "w": max(1, int(round(float(roi_def["w"]) * frame_scale))),
                "h": max(1, int(round(float(roi_def["h"]) * frame_scale))),
            }
        )
    return scaled

def _effective_pixel_size_for_formal_scale(pixel_size, frame_scale):
    """
    Keep physical/px-based units comparable after image resizing.

    If the image is resized to 0.5 width and height, each processed pixel
    represents 2 original pixels in each axis, so pixel_size is divided by 0.5.
    With PIXEL_SIZE=1.0, area remains approximately in original px^2.
    """
    frame_scale = _safe_formal_frame_scale(frame_scale)
    if pixel_size is None:
        return pixel_size
    return float(pixel_size) / frame_scale



def _scale_radius_param_for_formal_analysis(value, frame_scale):
    """Scale a pixel-radius parameter when formal frames are resized."""
    if value is None:
        return None
    frame_scale = _safe_formal_frame_scale(frame_scale)
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(value_float) or value_float <= 0:
        return int(round(value_float)) if np.isfinite(value_float) else value
    return max(1, int(round(value_float * frame_scale)))


def _scale_area_param_for_formal_analysis(value, frame_scale):
    """Scale a pixel-area parameter when formal frames are resized."""
    if value is None:
        return None
    frame_scale = _safe_formal_frame_scale(frame_scale)
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(value_float) or value_float <= 0:
        return int(round(value_float)) if np.isfinite(value_float) else value
    return max(1, int(round(value_float * frame_scale * frame_scale)))


def _scale_hsv_params_for_formal_analysis(hsv_params, frame_scale):
    """Scale pixel-based HSV segmentation parameters for resized formal frames."""
    scaled = dict(hsv_params or {})
    radius_keys = {
        "morph_open_size",
        "morph_close_size",
        "yellow_chroma_rim_exclusion_px",
        "yellow_chroma_connect_radius",
        "relaxed_connect_radius",
        "rim_exclusion_px",
        "vesselness_connect_radius",
    }
    area_keys = {
        "min_obj_size",
        "min_hole_size",
        "body_min_hole_size",
        "body_max_hole_size",
        "relaxed_min_obj_size",
        "vesselness_min_obj_size",
    }
    for key in radius_keys:
        if key in scaled:
            scaled[key] = _scale_radius_param_for_formal_analysis(scaled[key], frame_scale)
    for key in area_keys:
        if key in scaled:
            scaled[key] = _scale_area_param_for_formal_analysis(scaled[key], frame_scale)
    return scaled


def _prepare_formal_temporal_item(
    frame_idx,
    frame,
    name,
    dish_center,
    dish_radius,
    hsv_params,
    temporal_chroma_min=10,
    temporal_min_channel=35,
    temporal_max_value=240,
    temporal_rim_exclusion_px=30,
    temporal_min_obj_size=20,
    frame_scale=1.0,
):
    """Segment one resized formal-analysis frame and prepare candidate pixels for temporal recovery."""
    frame_scale = _safe_formal_frame_scale(frame_scale)
    frame = _resize_frame_for_formal_analysis(frame, frame_scale)
    scaled_dish_center = _scale_point_for_formal_analysis(dish_center, frame_scale)
    scaled_dish_radius = _scale_radius_for_formal_analysis(dish_radius, frame_scale)
    scaled_hsv_params = _scale_hsv_params_for_formal_analysis(hsv_params or {}, frame_scale)
    enhanced, strict_mask, pre_debug = segment_hsv_circle(
        frame,
        dish_center=scaled_dish_center,
        dish_radius=scaled_dish_radius,
        **scaled_hsv_params,
    )
    candidate = temporal_pale_branch_candidate(
        frame,
        dish_center=scaled_dish_center,
        dish_radius=scaled_dish_radius,
        chroma_min=temporal_chroma_min,
        min_channel=temporal_min_channel,
        max_value=temporal_max_value,
        rim_exclusion_px=_scale_radius_param_for_formal_analysis(temporal_rim_exclusion_px, frame_scale),
        min_obj_size=_scale_area_param_for_formal_analysis(temporal_min_obj_size, frame_scale),
    )
    return {
        "frame_idx": int(frame_idx),
        "name": name,
        "frame": frame,
        "gray": _as_gray_uint8(frame),
        "enhanced": enhanced,
        "mask": strict_mask.astype(bool),
        "candidate": candidate.astype(bool),
        "pre_debug": pre_debug,
    }



def _prepare_formal_temporal_item_for_parallel(args):
    """Worker wrapper for formal temporal frame preparation."""
    (
        frame_idx,
        frame,
        name,
        dish_center,
        dish_radius,
        hsv_params,
        temporal_chroma_min,
        temporal_min_channel,
        temporal_max_value,
        temporal_rim_exclusion_px,
        temporal_min_obj_size,
        frame_scale,
    ) = args
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    return _prepare_formal_temporal_item(
        frame_idx,
        frame,
        name,
        dish_center=dish_center,
        dish_radius=dish_radius,
        hsv_params=hsv_params,
        temporal_chroma_min=temporal_chroma_min,
        temporal_min_channel=temporal_min_channel,
        temporal_max_value=temporal_max_value,
        temporal_rim_exclusion_px=temporal_rim_exclusion_px,
        temporal_min_obj_size=temporal_min_obj_size,
        frame_scale=frame_scale,
    )

def _build_temporal_formal_row_debug(
    item,
    dt,
    pixel_size,
    roi,
    analysis_roi,
    vein_params,
    vein_every_n_frames,
    compute_body_skeleton,
    compute_front_speed,
    compute_contraction,
    contraction_rois,
    contraction_intensity_source,
    contraction_mask_layer,
    contraction_min_body_pixels,
    draw_contraction_rois,
    prev_mask,
    prev_area,
):
    """Build one metrics row and optional debug record from a temporal-recovered item."""
    frame_idx = int(item["frame_idx"])
    frame = item["frame"]
    gray = _as_gray_uint8(frame)
    pre_debug = item.get("pre_debug", {})
    strict_mask = item.get("strict_mask", item.get("mask")).astype(bool)
    temporal_mask = item["mask"].astype(bool)
    temporal_added_mask = item.get(
        "temporal_added_mask", np.zeros_like(temporal_mask, dtype=bool)
    ).astype(bool)
    candidate_mask = item.get(
        "candidate", np.zeros_like(temporal_mask, dtype=bool)
    ).astype(bool)

    occupied_mask = temporal_mask
    clean_mask = temporal_mask
    envelope_mask = ndi.binary_fill_holes(temporal_mask).astype(bool)
    body_layers = body_mask_layer_metrics(
        occupied_mask,
        clean_mask,
        envelope_mask,
        pixel_size=pixel_size,
    )
    body_layers["analysis_mask_type"] = "temporal_body"
    body_layers["formal_segmentation_mask"] = "temporal_recovered_hsv_mask"
    body_layers["use_temporal_body_recovery_formal"] = True
    body_layers["strict_hsv_area_px"] = int(np.sum(strict_mask))
    body_layers["strict_hsv_area"] = float(np.sum(strict_mask) * pixel_size**2)
    body_layers["temporal_candidate_area_px"] = int(np.sum(candidate_mask))
    body_layers["temporal_recoverable_candidate_area_px"] = int(
        np.sum(candidate_mask & (~strict_mask))
    )
    body_layers["temporal_added_area_px"] = int(np.sum(temporal_added_mask))
    body_layers["temporal_added_area"] = float(
        np.sum(temporal_added_mask) * pixel_size**2
    )
    body_layers["temporal_added_fraction"] = float(
        np.sum(temporal_added_mask) / max(np.sum(temporal_mask), 1)
    )

    focus = focus_score_laplacian(gray)
    reg = region_metrics(temporal_mask, pixel_size=pixel_size)
    mask_qc = mask_qc_metrics(temporal_mask, analysis_roi=analysis_roi)
    skel = (
        skeleton_metrics(temporal_mask, pixel_size=pixel_size)
        if compute_body_skeleton
        else empty_skeleton_metrics_like(temporal_mask)
    )
    compute_vein = vein_every_n_frames > 0 and frame_idx % int(vein_every_n_frames) == 0
    vein = (
        vein_network_metrics(
            frame,
            dish_center=pre_debug.get("dish_center"),
            dish_radius=pre_debug.get("dish_radius"),
            body_mask=temporal_mask,
            pixel_size=pixel_size,
            **(vein_params or {}),
        )
        if compute_vein
        else None
    )
    dyn = expansion_retraction_metrics(
        prev_mask,
        temporal_mask,
        dt=dt,
        pixel_size=pixel_size,
        compute_front_speed=compute_front_speed,
    )

    area = reg["area"]
    if frame_idx == 0 or not np.isfinite(prev_area) or dt is None or dt <= 0:
        dA_dt = np.nan
    else:
        dA_dt = float((area - prev_area) / dt)

    row = {
        "frame": frame_idx,
        "name": item.get("name", f"frame_{frame_idx:06d}"),
        "time": float(frame_idx * dt) if dt is not None else np.nan,
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

    if compute_contraction and contraction_rois:
        contraction_image = select_contraction_intensity_image(
            contraction_intensity_source,
            gray,
            item.get("enhanced", pre_debug.get("enhanced", gray)),
            pre_debug,
        )
        contraction_mask = select_contraction_mask(
            contraction_mask_layer,
            clean_mask=clean_mask,
            occupied_mask=occupied_mask,
            envelope_mask=envelope_mask,
        )
        add_contraction_metrics_to_row(
            row,
            contraction_image,
            contraction_rois,
            contraction_mask=contraction_mask,
            min_pixels=contraction_min_body_pixels,
        )
    debug_item = {
        "frame": frame_idx,
        "name": row["name"],
        "gray": gray,
        "background": pre_debug.get("background", np.zeros_like(gray, dtype=np.uint8)),
        "corrected": pre_debug.get("corrected", np.zeros_like(gray, dtype=np.uint8)),
        "enhanced": item.get("enhanced", pre_debug.get("enhanced", gray)),
        "mask": temporal_mask,
        "occupied_mask": occupied_mask,
        "strict_mask": strict_mask,
        "seed_mask": pre_debug.get("seed_mask"),
        "temporal_candidate_mask": candidate_mask,
        "temporal_added_mask": temporal_added_mask,
        # Reuse the existing cyan overlay channel to make temporal additions visible.
        "relaxed_added_mask": temporal_added_mask,
        "envelope_mask": envelope_mask,
        "skeleton": skel["skeleton"],
        "endpoint_mask": skel["endpoint_mask"],
        "branch_mask": skel["branch_mask"],
        "vein_mask": vein["vein_mask"] if vein is not None else None,
        "vein_skeleton": vein["vein_skeleton"] if vein is not None else None,
        "new_region": dyn["new_region"],
        "retracted_region": dyn["retracted_region"],
        "contraction_rois": contraction_rois,
        "draw_contraction_rois": bool(draw_contraction_rois),
    }
    return row, debug_item, temporal_mask, area


def _finalize_temporal_formal_dataframe(rows, roi, dt):
    """Apply post-processing steps shared by formal temporal analysis paths."""
    df = pd.DataFrame(rows)
    df = apply_vein_confidence(df)
    df = apply_body_mask_confidence(df)
    df = add_body_area_derivatives(df, dt)
    if roi is not None and "roi_mean_intensity" in df.columns:
        osc = analyze_oscillation(df["roi_mean_intensity"].to_numpy(), dt)
    else:
        osc = {"dominant_freq": np.nan, "period": np.nan, "n_peaks": 0}
    return df, osc


def analyze_timelapse_formal_temporal(
    frames,
    names=None,
    dt=30.0,
    pixel_size=1.0,
    roi=None,
    analysis_roi=None,
    dish_center=None,
    dish_radius=None,
    hsv_params=None,
    vein_params=None,
    vein_every_n_frames=1,
    compute_body_skeleton=True,
    compute_front_speed=True,
    compute_contraction=False,
    contraction_rois=None,
    contraction_intensity_source="gray",
    contraction_mask_layer="clean",
    contraction_min_body_pixels=50,
    draw_contraction_rois=False,
    keep_debug=True,
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
    formal_frame_scale=1.0,
    n_jobs=1,
):
    """Formal full-frame analysis where metrics are computed from temporal-recovered masks."""
    formal_frame_scale = _safe_formal_frame_scale(formal_frame_scale)
    analysis_pixel_size = _effective_pixel_size_for_formal_scale(pixel_size, formal_frame_scale)
    analysis_roi_scaled = _scale_roi_for_formal_analysis(analysis_roi, formal_frame_scale)
    roi_scaled = _scale_roi_for_formal_analysis(roi, formal_frame_scale)
    contraction_rois_scaled = _scale_contraction_rois_for_formal_analysis(
        contraction_rois, formal_frame_scale
    )
    contraction_min_body_pixels_scaled = _scale_area_param_for_formal_analysis(
        contraction_min_body_pixels, formal_frame_scale
    )
    if formal_frame_scale < 0.999:
        print(
            "Formal analysis frame scaling enabled: "
            f"scale={formal_frame_scale:.3f}. Metrics use effective pixel_size={analysis_pixel_size}."
        )
    temporal_connect_radius = _scale_radius_param_for_formal_analysis(
        temporal_connect_radius, formal_frame_scale
    )
    temporal_block_hole_max_area = _scale_area_param_for_formal_analysis(
        temporal_block_hole_max_area, formal_frame_scale
    )
    if frames is None or len(frames) == 0:
        raise ValueError("No frames were provided for temporal formal analysis.")
    if names is None:
        names = [f"frame_{i:06d}" for i in range(len(frames))]

    n_jobs = max(1, int(n_jobs or 1))
    prepare_args = [
        (
            i,
            frame,
            names[i] if i < len(names) else f"frame_{i:06d}",
        )
        for i, frame in enumerate(frames)
    ]
    if n_jobs > 1:
        print(f"Preparing temporal formal frames with {n_jobs} CPU workers")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _prepare_formal_temporal_item,
                    i,
                    frame,
                    name,
                    dish_center=dish_center,
                    dish_radius=dish_radius,
                    hsv_params=hsv_params,
                    temporal_chroma_min=temporal_chroma_min,
                    temporal_min_channel=temporal_min_channel,
                    temporal_max_value=temporal_max_value,
                    temporal_rim_exclusion_px=temporal_rim_exclusion_px,
                    temporal_min_obj_size=temporal_min_obj_size,
                    frame_scale=formal_frame_scale,
                )
                for i, frame, name in prepare_args
            ]
            items = [
                future.result()
                for future in tqdm(futures, desc=f"Preparing temporal formal frames with {n_jobs} CPU workers")
            ]
    else:
        items = []
        for i, frame, name in tqdm(prepare_args, desc="Preparing temporal formal frames"):
            items.append(
                _prepare_formal_temporal_item(
                    i,
                    frame,
                    name,
                    dish_center=dish_center,
                    dish_radius=dish_radius,
                    hsv_params=hsv_params,
                    temporal_chroma_min=temporal_chroma_min,
                    temporal_min_channel=temporal_min_channel,
                    temporal_max_value=temporal_max_value,
                    temporal_rim_exclusion_px=temporal_rim_exclusion_px,
                    temporal_min_obj_size=temporal_min_obj_size,
                    frame_scale=formal_frame_scale,
                )
            )

    recovered = apply_temporal_body_recovery_to_sequence(
        items,
        window=temporal_window,
        min_hits=temporal_min_hits,
        connect_radius=temporal_connect_radius,
        exterior_only=temporal_exterior_only,
        block_hole_max_area=temporal_block_hole_max_area,
        final_smooth_window=temporal_final_smooth_window,
        final_smooth_min_hits=temporal_final_smooth_min_hits,
    )

    rows = []
    debug_data = []
    prev_mask = None
    prev_area = np.nan
    for item in tqdm(recovered, desc="Computing temporal formal metrics"):
        row, debug_item, prev_mask_next, prev_area_next = _build_temporal_formal_row_debug(
            item,
            dt=dt,
            pixel_size=analysis_pixel_size,
            roi=roi_scaled,
            analysis_roi=analysis_roi_scaled,
            vein_params=vein_params,
            vein_every_n_frames=vein_every_n_frames,
            compute_body_skeleton=compute_body_skeleton,
            compute_front_speed=compute_front_speed,
            compute_contraction=compute_contraction,
            contraction_rois=contraction_rois_scaled,
            contraction_intensity_source=contraction_intensity_source,
            contraction_mask_layer=contraction_mask_layer,
            contraction_min_body_pixels=contraction_min_body_pixels_scaled,
            draw_contraction_rois=draw_contraction_rois,
            prev_mask=prev_mask,
            prev_area=prev_area,
        )
        rows.append(row)
        if keep_debug:
            debug_data.append(debug_item)
        prev_mask = prev_mask_next
        prev_area = prev_area_next

    df, osc = _finalize_temporal_formal_dataframe(rows, roi, dt)
    return df, debug_data, osc


def analyze_video_stream_formal_temporal(
    video_path,
    dt=30.0,
    pixel_size=1.0,
    roi=None,
    max_frames=None,
    start_frame=0,
    debug_frame_indices=None,
    dish_center=None,
    dish_radius=None,
    hsv_params=None,
    vein_params=None,
    vein_every_n_frames=1,
    compute_body_skeleton=True,
    compute_front_speed=True,
    analysis_roi=None,
    compute_contraction=False,
    contraction_rois=None,
    contraction_intensity_source="gray",
    contraction_mask_layer="clean",
    contraction_min_body_pixels=50,
    draw_contraction_rois=False,
    overlay_video_path=None,
    overlay_video_fps=30,
    overlay_video_output_scale=1.0,
    overlay_style="formal",
    overlay_label=None,
    overlay_total_frames=None,
    overlay_alpha=0.40,
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
    batch_frames=48,
    edge_skip=0,
    formal_frame_scale=1.0,
    n_jobs=1,
):
    """
    Low-memory formal analysis from a video using temporal-recovered masks.

    Frames are processed in core chunks with temporal halo frames around each
    chunk, so the symmetric temporal window can be used without loading the
    entire movie into memory.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(start_frame or 0))
    if total > 0 and start_frame >= total:
        cap.release()
        raise ValueError(
            f"start_frame={start_frame} is outside the video length ({total})."
        )
    available_frames = max(0, total - start_frame) if total > 0 else int(max_frames or 0)
    if max_frames is not None:
        total_to_process = min(available_frames, int(max_frames)) if available_frames > 0 else int(max_frames)
    else:
        total_to_process = available_frames
    if total_to_process <= 0:
        cap.release()
        raise ValueError(f"No frames could be read from video: {video_path}")

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
    temporal_halo = max(temporal_window // 2, temporal_final_smooth_window // 2)
    batch_frames = max(1, int(batch_frames or 1))
    n_jobs = max(1, int(n_jobs or 1))
    if n_jobs > 1:
        batch_frames = max(batch_frames, n_jobs)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
    debug_frame_indices = set(debug_frame_indices or [])
    overlay_video_output_scale = _safe_overlay_output_scale(overlay_video_output_scale)
    overlay_style = str(overlay_style or "formal").strip().lower()
    if overlay_style not in {"formal", "validation_temporal"}:
        raise ValueError(f"Unknown overlay_style: {overlay_style!r}")
    overlay_label = str(overlay_label or "validation")
    overlay_total_frames = int(overlay_total_frames or total)
    formal_frame_scale = _safe_formal_frame_scale(formal_frame_scale)
    analysis_pixel_size = _effective_pixel_size_for_formal_scale(pixel_size, formal_frame_scale)
    analysis_roi_scaled = _scale_roi_for_formal_analysis(analysis_roi, formal_frame_scale)
    roi_scaled = _scale_roi_for_formal_analysis(roi, formal_frame_scale)
    contraction_rois_scaled = _scale_contraction_rois_for_formal_analysis(
        contraction_rois, formal_frame_scale
    )
    contraction_min_body_pixels_scaled = _scale_area_param_for_formal_analysis(
        contraction_min_body_pixels, formal_frame_scale
    )
    # Frames are already resized before analysis. Avoid double-shrinking overlay output.
    if formal_frame_scale < 0.999:
        overlay_video_output_scale = 1.0
        print(
            "Formal analysis frame scaling enabled: "
            f"scale={formal_frame_scale:.3f}. Metrics use effective pixel_size={analysis_pixel_size}. "
            "Formal overlay is written at the same processed resolution."
        )
    temporal_connect_radius = _scale_radius_param_for_formal_analysis(
        temporal_connect_radius, formal_frame_scale
    )
    temporal_block_hole_max_area = _scale_area_param_for_formal_analysis(
        temporal_block_hole_max_area, formal_frame_scale
    )

    writer = None
    writer_size = None
    if overlay_video_path is not None:
        overlay_video_path = Path(overlay_video_path)
        overlay_video_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    debug_data = []
    prev_mask = None
    prev_area = np.nan
    core_start = start_frame
    core_limit = start_frame + total_to_process
    source_last_frame = total - 1 if total > 0 else core_limit - 1

    if overlay_style == "validation_temporal":
        print(
            "Formal temporal recovery enabled: metrics and summary use the "
            "temporal-recovered mask; validation clip uses the lightweight "
            "color overlay style."
        )
    else:
        print(
            "Formal temporal recovery enabled: metrics, summary, and formal overlay "
            "will use the temporal-recovered mask."
        )

    try:
        with tqdm(total=total_to_process, desc="Analyzing video with formal temporal recovery") as pbar:
            while core_start < core_limit:
                core_end = min(core_limit - 1, core_start + batch_frames - 1)
                # Read temporal context outside the requested output window when
                # the source video contains it. core_items below still limits
                # CSV, plots, debug images, and overlay video to the output range.
                read_start = max(0, core_start - temporal_halo)
                read_end = min(source_last_frame, core_end + temporal_halo)

                cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)
                prepare_args = []
                for frame_idx in range(read_start, read_end + 1):
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    prepare_args.append(
                        (
                            frame_idx,
                            frame_bgr,
                            f"frame_{frame_idx:06d}",
                            dish_center,
                            dish_radius,
                            hsv_params,
                            temporal_chroma_min,
                            temporal_min_channel,
                            temporal_max_value,
                            temporal_rim_exclusion_px,
                            temporal_min_obj_size,
                            formal_frame_scale,
                        )
                    )

                if not prepare_args:
                    break

                if n_jobs > 1:
                    try:
                        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                            items = list(
                                executor.map(
                                    _prepare_formal_temporal_item_for_parallel,
                                    prepare_args,
                                )
                            )
                    except BrokenProcessPool:
                        warnings.warn(
                            "ProcessPool crashed during formal temporal preparation; "
                            "falling back to single-process preparation for this batch. "
                            "Use fewer VALIDATION_WORKERS_PER_SECTION or smaller "
                            "VALIDATION_CLIP_BATCH_FRAMES if this repeats."
                        )
                        items = [
                            _prepare_formal_temporal_item_for_parallel(args)
                            for args in prepare_args
                        ]
                else:
                    items = [
                        _prepare_formal_temporal_item_for_parallel(args)
                        for args in prepare_args
                    ]
                if not items:
                    break

                recovered = apply_temporal_body_recovery_to_sequence(
                    items,
                    window=temporal_window,
                    min_hits=temporal_min_hits,
                    connect_radius=temporal_connect_radius,
                    exterior_only=temporal_exterior_only,
                    block_hole_max_area=temporal_block_hole_max_area,
                    final_smooth_window=temporal_final_smooth_window,
                    final_smooth_min_hits=temporal_final_smooth_min_hits,
                )

                core_items = [
                    item for item in recovered
                    if core_start <= int(item["frame_idx"]) <= core_end
                ]
                for item in core_items:
                    row, debug_item, prev_mask_next, prev_area_next = _build_temporal_formal_row_debug(
                        item,
                        dt=dt,
                        pixel_size=analysis_pixel_size,
                        roi=roi_scaled,
                        analysis_roi=analysis_roi_scaled,
                        vein_params=vein_params,
                        vein_every_n_frames=vein_every_n_frames,
                        compute_body_skeleton=compute_body_skeleton,
                        compute_front_speed=compute_front_speed,
                        compute_contraction=compute_contraction,
                        contraction_rois=contraction_rois_scaled,
                        contraction_intensity_source=contraction_intensity_source,
                        contraction_mask_layer=contraction_mask_layer,
                        contraction_min_body_pixels=contraction_min_body_pixels_scaled,
                        draw_contraction_rois=draw_contraction_rois,
                        prev_mask=prev_mask,
                        prev_area=prev_area,
                    )
                    frame_idx = int(item["frame_idx"])
                    rows.append(row)

                    if frame_idx in debug_frame_indices:
                        debug_data.append(debug_item)

                    if overlay_video_path is not None:
                        if overlay_style == "validation_temporal":
                            overlay_bgr = _render_temporal_validation_overlay(
                                item["frame"],
                                item["mask"],
                                item.get("temporal_added_mask"),
                                overlay_label,
                                frame_idx,
                                overlay_total_frames,
                                overlay_alpha,
                                contraction_rois=contraction_rois_scaled,
                                draw_contraction_rois=draw_contraction_rois,
                            )
                        else:
                            overlay_rgb = overlay_mask_skeleton(
                                debug_item["gray"],
                                mask=debug_item["mask"],
                                skeleton=debug_item["skeleton"],
                                endpoint_mask=debug_item["endpoint_mask"],
                                branch_mask=debug_item["branch_mask"],
                                vein_mask=debug_item.get("vein_mask"),
                                vein_skeleton=debug_item.get("vein_skeleton"),
                                relaxed_added_mask=debug_item.get("temporal_added_mask"),
                                new_region=debug_item["new_region"],
                                retracted_region=debug_item["retracted_region"],
                                contraction_rois=contraction_rois_scaled,
                                draw_contraction_rois=draw_contraction_rois,
                            )
                            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                        if writer is None:
                            height, width = overlay_bgr.shape[:2]
                            out_width = max(1, int(round(width * overlay_video_output_scale)))
                            out_height = max(1, int(round(height * overlay_video_output_scale)))
                            writer_size = (out_width, out_height)
                            writer = cv2.VideoWriter(
                                str(overlay_video_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                float(overlay_video_fps),
                                writer_size,
                                True,
                            )
                            if not writer.isOpened():
                                raise ValueError(
                                    f"Could not open video writer for: {overlay_video_path}"
                                )
                            pbar.write(
                                ("Writing temporal validation overlay video " if overlay_style == "validation_temporal" else "Writing formal temporal overlay video ") +
                                f"at scale {overlay_video_output_scale:.3f}: "
                                f"{width}x{height} -> {out_width}x{out_height}"
                            )
                        if writer_size is not None and overlay_bgr.shape[1::-1] != writer_size:
                            overlay_bgr = cv2.resize(
                                overlay_bgr,
                                writer_size,
                                interpolation=cv2.INTER_AREA,
                            )
                        writer.write(overlay_bgr)

                    prev_mask = prev_mask_next
                    prev_area = prev_area_next

                processed = len(core_items)
                pbar.update(processed)
                core_start = core_end + 1
                del items, recovered, core_items
                gc.collect()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not rows:
        raise ValueError(f"No frames could be read from video: {video_path}")

    df, osc = _finalize_temporal_formal_dataframe(rows, roi, dt)
    return df, debug_data, osc

def validation_section_frame_ranges(total_frames, section_frames, sections=("front", "middle", "rear"), edge_skip=0):
    """Return front/middle/rear frame ranges while avoiding unreliable video edges."""
    total_frames = int(total_frames or 0)
    if total_frames <= 0:
        raise ValueError("Video frame count is unavailable; cannot define validation sections.")
    edge_skip = max(0, int(edge_skip or 0))
    usable_start = min(edge_skip, max(0, total_frames - 1))
    usable_end = max(usable_start, total_frames - edge_skip - 1)
    usable_count = max(1, usable_end - usable_start + 1)
    section_frames = max(1, min(int(section_frames), usable_count))
    starts = {
        "front": usable_start,
        "middle": usable_start + max(0, (usable_count - section_frames) // 2),
        "rear": usable_end - section_frames + 1,
    }
    ranges = []
    for section in sections:
        if section not in starts:
            raise ValueError(f"Unknown validation section: {section!r}")
        start = int(starts[section])
        end = min(usable_end, start + section_frames - 1)
        ranges.append((section, start, end))
    return ranges

def analyze_validation_sections_formal_outputs(
    video_path,
    out_dir,
    sections,
    section_frames,
    dt,
    pixel_size,
    roi,
    dish_center,
    dish_radius,
    hsv_params,
    vein_params,
    vein_every_n_frames,
    compute_body_skeleton,
    compute_front_speed,
    compute_contraction,
    contraction_rois,
    contraction_intensity_source,
    contraction_mask_layer,
    contraction_min_body_pixels,
    contraction_min_period_sec,
    contraction_max_period_sec,
    contraction_min_valid_samples,
    contraction_interpolate_short_gaps,
    contraction_max_interpolation_gap,
    save_contraction_plots,
    draw_contraction_rois,
    formal_frame_scale,
    temporal_kwargs,
    validation_output_scale=0.5,
    validation_stills_per_section=6,
    validation_write_video=True,
    edge_skip=0,
    batch_frames=48,
    n_jobs=1,
):
    """
    Run formal-style metrics and plots on front/middle/rear validation windows.

    This uses the same formal temporal body-mask analysis path as the full-video
    formal run, but writes outputs under validation_analysis/<section>/.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    root = out_dir / "validation_analysis"
    validation_clip_root = out_dir / "validation_clips"
    root.mkdir(parents=True, exist_ok=True)
    section_summaries = []
    ranges = validation_section_frame_ranges(total_frames, section_frames, sections, edge_skip=edge_skip)

    for section, start_frame, end_frame in ranges:
        n_frames = int(end_frame - start_frame + 1)
        section_dir = root / section
        section_dir.mkdir(parents=True, exist_ok=True)
        still_frames = sorted(
            set(
                np.linspace(
                    start_frame,
                    end_frame,
                    min(int(validation_stills_per_section), n_frames),
                    dtype=int,
                ).tolist()
            )
        )
        debug_indices = still_frames
        validation_section_dir = validation_clip_root / section
        validation_still_dir = validation_section_dir / "stills"
        validation_still_dir.mkdir(parents=True, exist_ok=True)
        validation_overlay_video = (
            validation_section_dir / f"{section}_color_mask_overlay.mp4"
            if validation_write_video else None
        )
        print(
            f"Validation data analysis [{section}]: "
            f"frames {start_frame}-{end_frame} ({n_frames} frames)."
        )
        df, debug_data, osc = analyze_video_stream_formal_temporal(
            video_path,
            dt=dt,
            pixel_size=pixel_size,
            roi=roi,
            max_frames=n_frames,
            start_frame=start_frame,
            debug_frame_indices=debug_indices,
            dish_center=dish_center,
            dish_radius=dish_radius,
            hsv_params=hsv_params,
            vein_params=vein_params,
            vein_every_n_frames=vein_every_n_frames,
            compute_body_skeleton=compute_body_skeleton,
            compute_front_speed=compute_front_speed,
            compute_contraction=compute_contraction,
            contraction_rois=contraction_rois,
            contraction_intensity_source=contraction_intensity_source,
            contraction_mask_layer=contraction_mask_layer,
            contraction_min_body_pixels=contraction_min_body_pixels,
            draw_contraction_rois=draw_contraction_rois,
            overlay_video_path=validation_overlay_video,
            overlay_video_output_scale=validation_output_scale,
            overlay_style="validation_temporal",
            overlay_label=section,
            overlay_total_frames=total_frames,
            overlay_alpha=0.40,
            formal_frame_scale=formal_frame_scale,
            batch_frames=batch_frames,
            n_jobs=n_jobs,
            **dict(temporal_kwargs or {}),
        )
        df.insert(0, "validation_section", section)
        df.insert(1, "section_start_frame", start_frame)
        df.insert(2, "section_end_frame", end_frame)
        metrics_csv = section_dir / f"{section}_formal_metrics.csv"
        df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

        summary_plot = section_dir / f"{section}_plasmodium_summary.png"
        plot_summary(df, out_prefix=summary_plot)
        plt.close("all")

        contraction_summary = pd.DataFrame()
        if compute_contraction and parse_contraction_rois(contraction_rois):
            contraction_summary = analyze_contraction_rois(
                df=df,
                contraction_rois=contraction_rois,
                dt=dt,
                min_period_sec=contraction_min_period_sec,
                max_period_sec=contraction_max_period_sec,
                min_valid_samples=contraction_min_valid_samples,
                interpolate_short_gaps=contraction_interpolate_short_gaps,
                max_interpolation_gap=contraction_max_interpolation_gap,
                out_dir=section_dir,
                save_plots=save_contraction_plots,
            )
            plot_contraction_summary(
                contraction_summary,
                section_dir / f"{section}_plasmodium_summary_contraction.png",
            )
            plt.close("all")

        if debug_data:
            debug_frame_numbers = [int(item["frame"]) for item in debug_data]
            save_debug_examples(
                debug_data,
                section_dir / "debug_examples",
                debug_frame_numbers,
                use_frame_number=True,
                contraction_rois=contraction_rois,
                draw_contraction_rois=draw_contraction_rois,
            )
            save_debug_examples(
                debug_data,
                validation_still_dir,
                debug_frame_numbers,
                use_frame_number=True,
                contraction_rois=contraction_rois,
                draw_contraction_rois=draw_contraction_rois,
            )
            print(f"Saved validation section stills: {validation_still_dir}")

        summary_df = summarize_run(df, label=f"validation_{section}")
        summary_df.insert(0, "validation_section", section)
        summary_df.insert(1, "section_start_frame", start_frame)
        summary_df.insert(2, "section_end_frame", end_frame)
        summary_df["osc_dominant_freq"] = osc.get("dominant_freq", np.nan)
        summary_df["osc_period"] = osc.get("period", np.nan)
        summary_df["osc_n_peaks"] = osc.get("n_peaks", 0)
        summary_df["contraction_roi_count"] = int(len(contraction_summary))
        summary_csv = section_dir / f"{section}_summary_table.csv"
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        section_summaries.append(summary_df)
        print(f"Saved validation section metrics: {metrics_csv}")
        print(f"Saved validation section summary plot: {summary_plot}")
        gc.collect()

    combined = pd.concat(section_summaries, ignore_index=True) if section_summaries else pd.DataFrame()
    combined_csv = root / "validation_analysis_summary.csv"
    combined.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    print(f"Saved validation analysis summary: {combined_csv}")
    return combined

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
    contraction_rois=None,
    draw_contraction_rois=False,
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
            contraction_rois=item.get("contraction_rois", contraction_rois),
            draw_contraction_rois=item.get(
                "draw_contraction_rois", draw_contraction_rois
            ),
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
        if item.get("temporal_added_mask") is not None:
            cv2.imwrite(
                str(out_dir / f"{stem}_temporal_added_mask.png"),
                item["temporal_added_mask"].astype(np.uint8) * 255,
            )
        if item.get("temporal_candidate_mask") is not None:
            cv2.imwrite(
                str(out_dir / f"{stem}_temporal_candidate_mask.png"),
                item["temporal_candidate_mask"].astype(np.uint8) * 255,
            )
        if item.get("strict_mask") is not None:
            cv2.imwrite(
                str(out_dir / f"{stem}_strict_hsv_mask.png"),
                item["strict_mask"].astype(np.uint8) * 255,
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
    contraction_rois=None,
    draw_contraction_rois=False,
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
                relaxed_added_mask=item.get("relaxed_added_mask"),
                contraction_rois=item.get("contraction_rois", contraction_rois),
                draw_contraction_rois=item.get(
                    "draw_contraction_rois", draw_contraction_rois
                ),
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
        contraction_rois,
        draw_contraction_rois,
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
    if draw_contraction_rois and contraction_rois:
        _draw_contraction_rois(
            overlay, contraction_rois, color=(255, 170, 215), thickness=2
        )
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
    contraction_rois=None,
    draw_contraction_rois=False,
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
    if draw_contraction_rois and contraction_rois:
        _draw_contraction_rois(
            overlay, contraction_rois, color=(255, 170, 215), thickness=2
        )
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
    contraction_rois=None,
    draw_contraction_rois=False,
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
    scaled_contraction_rois = _scale_contraction_rois_for_formal_analysis(
        contraction_rois, output_scale
    )

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
                    # Keep the validation output range unchanged, but allow its
                    # boundary frames to use temporal context from the source.
                    read_start = max(0, core_start - temporal_halo)
                    read_end = min(total_frames - 1, core_end + temporal_halo)
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
                            contraction_rois=scaled_contraction_rois,
                            draw_contraction_rois=draw_contraction_rois,
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
    contraction_rois=None,
    draw_contraction_rois=False,
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
    contraction_rois = parse_contraction_rois(contraction_rois)
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
                                contraction_rois,
                                draw_contraction_rois,
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


def _show_segmentation_overlay(
    gray,
    mask,
    contraction_rois=None,
    draw_contraction_rois=False,
):
    """Display first-frame segmentation and skeleton overlay."""
    skel = skeleton_metrics(mask)
    overlay = overlay_mask_skeleton(
        gray,
        mask=mask,
        skeleton=skel["skeleton"],
        endpoint_mask=skel["endpoint_mask"],
        branch_mask=skel["branch_mask"],
        contraction_rois=contraction_rois,
        draw_contraction_rois=draw_contraction_rois,
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
    contraction_rois=None,
    draw_contraction_rois=False,
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
    _show_segmentation_overlay(
        first_gray,
        first_mask,
        contraction_rois=contraction_rois,
        draw_contraction_rois=draw_contraction_rois,
    )
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
    contraction_rois=None,
    draw_contraction_rois=False,
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
                contraction_rois=contraction_rois,
                draw_contraction_rois=draw_contraction_rois,
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



def _sampled_segment_ranges(total_frames, frames_per_segment=300):
    """Return early/middle/late inclusive-exclusive frame ranges."""
    total_frames = int(total_frames)
    n = max(1, int(frames_per_segment))
    if total_frames <= 0:
        raise ValueError("Video frame count must be positive.")
    n = min(n, total_frames)
    early = (0, n)
    middle_start = max(0, min(total_frames - n, total_frames // 2 - n // 2))
    late_start = max(0, total_frames - n)
    return {
        "early": early,
        "middle": (middle_start, middle_start + n),
        "late": (late_start, late_start + n),
    }


def _read_video_frame_range(video_path, start_frame, end_frame):
    """
    Read [start_frame, end_frame) from a video as BGR frames.

    OpenCV VideoCapture returns BGR. Keep that format because the segmentation
    pipeline uses cv2.COLOR_BGR2HSV and cv2.COLOR_BGR2GRAY. Converting these
    frames to RGB here would swap red/blue channels and cause incorrect HSV
    masks, especially large false-positive dish/background regions.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    frames = []
    names = []
    idx = int(start_frame)
    try:
        while idx < int(end_frame):
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
            names.append(f"frame_{idx:06d}")
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(
            f"No frames could be read from range [{start_frame}, {end_frame})."
        )
    return frames, names


def _read_video_frame_at(video_path, frame_index):
    """Read one RGB frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index}.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _safe_ui_roi_name(name, fallback):
    name = str(name).strip() or fallback
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return name or fallback


def select_contraction_rois_ui(
    rgb_frame,
    window_title="Select contraction ROIs",
    default_prefix="roi",
):
    """
    Let the user draw multiple ROIs with OpenCV.

    Draw a box, press Enter/Space to accept it, and press Esc when finished.
    ROI names are entered in the console after the window closes.
    """
    bgr = cv2.cvtColor(_as_rgb_uint8(rgb_frame), cv2.COLOR_RGB2BGR)
    display = bgr.copy()
    instructions = [
        "Drag ROI; Enter/Space = accept",
        "Esc = finish selection",
    ]
    y = 28
    for msg in instructions:
        cv2.putText(
            display, msg, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 28
    try:
        boxes = cv2.selectROIs(
            window_title,
            display,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    boxes = np.asarray(boxes, dtype=int).reshape(-1, 4)
    rois = []
    used = set()
    for i, (x, y, w, h) in enumerate(boxes):
        if w <= 0 or h <= 0:
            continue
        fallback = f"{default_prefix}_{i}"
        entered = input(f"Name for ROI {i + 1} [{fallback}]: ").strip()
        name = _safe_ui_roi_name(entered, fallback)
        base = name
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        rois.append((name, int(x), int(y), int(w), int(h)))
    return parse_contraction_rois(rois)


def _save_sampled_roi_preview(rgb_frame, rois, out_path, title=None):
    preview = _as_rgb_uint8(rgb_frame).copy()
    h, w = preview.shape[:2]
    for roi in parse_contraction_rois(rois):
        x0 = max(0, min(w - 1, int(roi["x"])))
        y0 = max(0, min(h - 1, int(roi["y"])))
        x1 = max(0, min(w - 1, int(roi["x"] + roi["w"])))
        y1 = max(0, min(h - 1, int(roi["y"] + roi["h"])))
        cv2.rectangle(preview, (x0, y0), (x1, y1), (220, 140, 255), 2)
        cv2.putText(
            preview, roi["name"], (x0, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 140, 255), 2, cv2.LINE_AA
        )
    if title:
        cv2.putText(
            preview, str(title), (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))



def _as_tracker_gray(frame):
    """Return contrast-normalized gray image for tracker-based contraction."""
    gray = _as_gray_uint8(frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _clip_tracker_box(x, y, w, h, image_shape):
    height, width = image_shape[:2]
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x = int(round(x))
    y = int(round(y))
    x = max(0, min(max(0, width - w), x))
    y = max(0, min(max(0, height - h), y))
    return x, y, w, h


def _match_tracker_template(gray, template, prev_box, search_radius):
    x, y, w, h = prev_box
    radius = max(0, int(round(search_radius)))
    height, width = gray.shape[:2]
    sx0 = max(0, x - radius)
    sy0 = max(0, y - radius)
    sx1 = min(width, x + w + radius)
    sy1 = min(height, y + h + radius)
    search = gray[sy0:sy1, sx0:sx1]
    if search.shape[0] < h or search.shape[1] < w:
        return prev_box, np.nan
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, max_loc = cv2.minMaxLoc(result)
    next_x = sx0 + int(max_loc[0])
    next_y = sy0 + int(max_loc[1])
    return _clip_tracker_box(next_x, next_y, w, h, gray.shape), float(score)


def _tracker_internal_flow(prev_crop, crop):
    empty = {
        "flow_mag_mean": np.nan,
        "flow_mag_p95": np.nan,
        "flow_dx_median": np.nan,
        "flow_dy_median": np.nan,
        "diff_mean": np.nan,
    }
    if prev_crop is None or crop is None or prev_crop.shape != crop.shape:
        return empty
    if min(crop.shape[:2]) < 8:
        return empty
    prev_u8 = prev_crop.astype(np.uint8, copy=False)
    crop_u8 = crop.astype(np.uint8, copy=False)
    flow = cv2.calcOpticalFlowFarneback(
        prev_u8,
        crop_u8,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    diff = cv2.absdiff(prev_u8, crop_u8)
    return {
        "flow_mag_mean": float(np.mean(mag)),
        "flow_mag_p95": float(np.percentile(mag, 95)),
        "flow_dx_median": float(np.median(flow[..., 0])),
        "flow_dy_median": float(np.median(flow[..., 1])),
        "diff_mean": float(np.mean(diff)),
    }


def _principal_tracker_displacement(xs, ys):
    coords = np.column_stack([xs, ys]).astype(float)
    valid = np.isfinite(coords).all(axis=1)
    signal = np.full(len(coords), np.nan, dtype=float)
    if np.count_nonzero(valid) < 3:
        return signal
    centered = coords[valid] - np.nanmean(coords[valid], axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
    except np.linalg.LinAlgError:
        direction = np.array([1.0, 0.0], dtype=float)
    signal[valid] = centered @ direction
    return signal


def _plot_tracker_contraction_signal(
    roi_name,
    signal_name,
    time_values,
    signal,
    match_scores,
    oscillation,
    summary_row,
    out_path,
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(time_values, signal, color="tab:blue", linewidth=1.0)
    axes[0].set_title(f"Tracker-like contraction proxy: {roi_name} / {signal_name}")
    axes[0].set_ylabel(signal_name)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_values, match_scores, color="tab:purple", linewidth=1.0)
    axes[1].set_ylabel("match score")
    axes[1].grid(alpha=0.25)

    frequencies = np.asarray(oscillation.get("fft_freq", []), dtype=float)
    power = np.asarray(oscillation.get("fft_power", []), dtype=float)
    positive = frequencies > 0 if frequencies.size else np.array([], dtype=bool)
    if frequencies.size and np.any(positive):
        axes[2].plot(frequencies[positive], power[positive], color="tab:green")
    dominant_freq = summary_row["dominant_freq_hz"]
    if np.isfinite(dominant_freq):
        axes[2].axvline(dominant_freq, color="tab:red", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("FFT power")
    axes[2].grid(alpha=0.25)

    annotation = (
        f"period={summary_row['period_sec']:.3g} s, "
        f"frequency={summary_row['dominant_freq_hz']:.3g} Hz\n"
        f"valid={summary_row['valid']}, match={summary_row['mean_match_score']:.3g}, "
        f"reason={summary_row['invalid_reason']}"
    )
    fig.text(0.02, 0.01, annotation, fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _tracker_invalid_reason(
    oscillation,
    n_valid,
    n_frames,
    mean_match_score,
    min_valid_samples,
    min_match_score,
    min_period_sec,
    max_period_sec,
):
    valid_fraction = float(n_valid / n_frames) if n_frames else 0.0
    period_sec = oscillation["period"]
    dominant_freq = oscillation["dominant_freq"]
    amplitude = oscillation["amplitude_peak_to_peak"]
    if n_valid < int(min_valid_samples):
        return "too_few_valid_samples"
    if valid_fraction < 0.8:
        return "low_valid_fraction"
    if np.isfinite(mean_match_score) and mean_match_score < float(min_match_score):
        return "low_match_score"
    if not np.isfinite(dominant_freq):
        return "no_frequency_in_range"
    if period_sec < float(min_period_sec) or period_sec > float(max_period_sec):
        return "no_frequency_in_range"
    if not np.isfinite(amplitude) or amplitude <= 0:
        return "zero_amplitude"
    return "ok"


def add_tracker_contraction_metrics_to_df(
    df,
    frames,
    contraction_rois,
    dt,
    out_dir=None,
    min_period_sec=30.0,
    max_period_sec=600.0,
    min_valid_samples=20,
    search_radius=80,
    min_match_score=0.25,
    template_update_alpha=0.03,
    save_plots=True,
):
    """
    Add Tracker-like contraction metrics based on ROI template displacement.

    This follows the Tracker software idea more closely than fixed-ROI intensity:
    choose a small texture/edge ROI, track its best-matching position over time,
    then estimate frequency from displacement or optical-flow proxy signals.
    """
    rois = parse_contraction_rois(contraction_rois)
    if df is None or len(df) == 0 or not rois:
        return pd.DataFrame()
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("A positive acquisition interval dt is required.")

    n_frames = min(len(df), len(frames))
    time_values = np.arange(n_frames, dtype=float) * float(dt)
    out_dir = Path(out_dir) if out_dir is not None else None
    plot_dir = out_dir / "tracker_contraction_plots" if out_dir is not None else None
    if plot_dir is not None and save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    gray_frames = [_as_tracker_gray(frame) for frame in frames[:n_frames]]
    rows = []
    for roi_def in rois:
        roi_name = roi_def["name"]
        x, y, w, h = _clip_tracker_box(
            roi_def["x"], roi_def["y"], roi_def["w"], roi_def["h"], gray_frames[0].shape
        )
        template = gray_frames[0][y:y + h, x:x + w].copy()
        prev_box = (x, y, w, h)
        prev_crop = None
        xs = []
        ys = []
        boxes = []
        scores = []
        step_px = []
        tracked_mean = []
        flow_mean = []
        flow_p95 = []
        flow_dx = []
        flow_dy = []
        diff_mean = []

        for i, gray in enumerate(gray_frames):
            if i == 0:
                box = prev_box
                score = 1.0
            else:
                box, score = _match_tracker_template(gray, template, prev_box, search_radius)
                if (not np.isfinite(score)) or score < float(min_match_score):
                    box = prev_box
            bx, by, bw, bh = box
            crop = gray[by:by + bh, bx:bx + bw].copy()
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            if xs:
                step_px.append(float(np.hypot(cx - xs[-1], cy - ys[-1])))
            else:
                step_px.append(0.0)
            flow_stats = _tracker_internal_flow(prev_crop, crop)
            xs.append(cx)
            ys.append(cy)
            boxes.append(box)
            scores.append(score)
            tracked_mean.append(float(np.mean(crop)))
            flow_mean.append(flow_stats["flow_mag_mean"])
            flow_p95.append(flow_stats["flow_mag_p95"])
            flow_dx.append(flow_stats["flow_dx_median"])
            flow_dy.append(flow_stats["flow_dy_median"])
            diff_mean.append(flow_stats["diff_mean"])

            if np.isfinite(score) and score >= float(min_match_score):
                alpha = float(np.clip(template_update_alpha, 0.0, 1.0))
                if alpha > 0 and crop.shape == template.shape:
                    template = cv2.addWeighted(template, 1.0 - alpha, crop, alpha, 0)
                prev_box = box
            prev_crop = crop

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        scores = np.asarray(scores, dtype=float)
        step_px = np.asarray(step_px, dtype=float)
        tracked_mean = np.asarray(tracked_mean, dtype=float)
        flow_mean = np.asarray(flow_mean, dtype=float)
        flow_p95 = np.asarray(flow_p95, dtype=float)
        flow_dx = np.asarray(flow_dx, dtype=float)
        flow_dy = np.asarray(flow_dy, dtype=float)
        diff_mean = np.asarray(diff_mean, dtype=float)
        principal = _principal_tracker_displacement(xs, ys)
        displacement = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)

        prefix = f"tracker_{roi_name}"
        df.loc[df.index[:n_frames], f"{prefix}_x"] = xs
        df.loc[df.index[:n_frames], f"{prefix}_y"] = ys
        df.loc[df.index[:n_frames], f"{prefix}_match_score"] = scores
        df.loc[df.index[:n_frames], f"{prefix}_step_px"] = step_px
        df.loc[df.index[:n_frames], f"{prefix}_displacement_px"] = displacement
        df.loc[df.index[:n_frames], f"{prefix}_principal_displacement_px"] = principal
        df.loc[df.index[:n_frames], f"{prefix}_flow_mag_mean"] = flow_mean
        df.loc[df.index[:n_frames], f"{prefix}_flow_mag_p95"] = flow_p95
        df.loc[df.index[:n_frames], f"{prefix}_flow_dx_median"] = flow_dx
        df.loc[df.index[:n_frames], f"{prefix}_flow_dy_median"] = flow_dy
        df.loc[df.index[:n_frames], f"{prefix}_diff_mean"] = diff_mean
        df.loc[df.index[:n_frames], f"{prefix}_tracked_mean"] = tracked_mean
        for field, values in (
            ("box_x", [b[0] for b in boxes]),
            ("box_y", [b[1] for b in boxes]),
            ("box_w", [b[2] for b in boxes]),
            ("box_h", [b[3] for b in boxes]),
        ):
            df.loc[df.index[:n_frames], f"{prefix}_{field}"] = np.asarray(values, dtype=float)

        candidate_signals = {
            "principal_displacement_px": principal,
            "displacement_px": displacement,
            "flow_mag_mean": flow_mean,
            "diff_mean": diff_mean,
        }
        best_name = None
        best_amp = -np.inf
        for signal_name, signal in candidate_signals.items():
            oscillation = analyze_oscillation(
                signal,
                dt,
                min_period_sec=min_period_sec,
                max_period_sec=max_period_sec,
                interpolate_short_gaps=True,
                max_interpolation_gap=3,
            )
            finite = np.isfinite(signal)
            n_valid = int(np.sum(finite))
            valid_fraction = float(n_valid / n_frames) if n_frames else 0.0
            mean_signal = float(np.nanmean(signal)) if np.isfinite(signal).any() else np.nan
            std_signal = float(np.nanstd(signal)) if np.isfinite(signal).any() else np.nan
            mean_match = float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan
            invalid_reason = _tracker_invalid_reason(
                oscillation,
                n_valid,
                n_frames,
                mean_match,
                min_valid_samples,
                min_match_score,
                min_period_sec,
                max_period_sec,
            )
            period_sec = oscillation["period"]
            amplitude = oscillation["amplitude_peak_to_peak"]
            summary_row = {
                "roi_name": roi_name,
                "signal_name": signal_name,
                "n_frames": n_frames,
                "n_valid": n_valid,
                "valid_fraction": valid_fraction,
                "mean_signal": mean_signal,
                "std_signal": std_signal,
                "mean_match_score": mean_match,
                "min_match_score": float(np.nanmin(scores)) if np.isfinite(scores).any() else np.nan,
                "mean_step_px": float(np.nanmean(step_px)) if np.isfinite(step_px).any() else np.nan,
                "amplitude_peak_to_peak": amplitude,
                "amplitude_std": oscillation["amplitude_std"],
                "dominant_freq_hz": oscillation["dominant_freq"],
                "period_sec": period_sec,
                "period_min": float(period_sec / 60.0) if np.isfinite(period_sec) else np.nan,
                "n_peaks": oscillation["n_peaks"],
                "sampling_interval_sec": float(dt),
                "valid": invalid_reason == "ok",
                "invalid_reason": invalid_reason,
                "frequency_quality_flag": oscillation["quality_flag"],
                "signal_interpretation": (
                    "Tracker-like optical motion proxy from template position, "
                    "local optical flow, or frame-difference energy. Not direct force or flow."
                ),
            }
            rows.append(summary_row)
            if np.isfinite(amplitude) and amplitude > best_amp:
                best_amp = float(amplitude)
                best_name = signal_name
            if plot_dir is not None and save_plots:
                _plot_tracker_contraction_signal(
                    roi_name,
                    signal_name,
                    time_values,
                    signal,
                    scores,
                    oscillation,
                    summary_row,
                    plot_dir / f"tracker_{roi_name}_{signal_name}.png",
                )
        if best_name is not None:
            df.loc[df.index[:n_frames], f"{prefix}_best_signal"] = best_name

    tracker_summary = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tracker_summary.to_csv(
            out_dir / "tracker_contraction_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return tracker_summary


def save_tracker_overlay_video(
    frames,
    df,
    contraction_rois,
    out_path,
    fps=30,
    output_scale=0.5,
):
    rois = parse_contraction_rois(contraction_rois)
    if not rois or df is None or len(df) == 0 or not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_scale = float(np.clip(float(output_scale), 0.1, 1.0))
    first = np.asarray(frames[0])
    height, width = first.shape[:2]
    out_size = (max(1, int(round(width * output_scale))), max(1, int(round(height * output_scale))))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        out_size,
        True,
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open tracker video writer for: {out_path}")
    try:
        for i, frame in enumerate(frames[:len(df)]):
            frame_arr = np.asarray(frame)
            if frame_arr.ndim == 2:
                bgr = cv2.cvtColor(_as_gray_uint8(frame_arr), cv2.COLOR_GRAY2BGR)
            else:
                bgr = frame_arr.astype(np.uint8, copy=True)
            for roi_def in rois:
                prefix = f"tracker_{roi_def['name']}"
                required = [f"{prefix}_box_x", f"{prefix}_box_y", f"{prefix}_box_w", f"{prefix}_box_h"]
                if any(col not in df.columns for col in required):
                    continue
                row = df.iloc[i]
                vals = [row[col] for col in required]
                if not all(np.isfinite(v) for v in vals):
                    continue
                x, y, w, h = [int(round(float(v))) for v in vals]
                score = row.get(f"{prefix}_match_score", np.nan)
                cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.circle(bgr, (x + w // 2, y + h // 2), 5, (0, 255, 255), -1)
                label = f"{roi_def['name']} score={score:.2f}" if np.isfinite(score) else roi_def["name"]
                cv2.putText(
                    bgr,
                    label,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if out_size != (width, height):
                bgr = cv2.resize(bgr, out_size, interpolation=cv2.INTER_AREA)
            writer.write(bgr)
    finally:
        writer.release()


def _frame_to_bgr_for_overlay(frame):
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(_as_gray_uint8(arr), cv2.COLOR_GRAY2BGR)
    return arr.astype(np.uint8, copy=True)


def _draw_signal_sparkline(panel, signal, color, label):
    h, w = panel.shape[:2]
    cv2.putText(panel, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if values.size < 2 or not np.any(finite):
        return
    values = values[-min(values.size, max(20, w - 20)):]
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return
    finite_values = values[finite]
    lo = float(np.nanpercentile(finite_values, 5))
    hi = float(np.nanpercentile(finite_values, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite_values))
        hi = float(np.nanmax(finite_values))
    if hi <= lo:
        return
    xs = np.linspace(8, w - 8, values.size)
    ys = h - 10 - (np.clip(values, lo, hi) - lo) / (hi - lo) * (h - 32)
    pts = []
    for x, y, ok in zip(xs, ys, finite):
        if ok and np.isfinite(y):
            pts.append((int(round(x)), int(round(y))))
        else:
            if len(pts) >= 2:
                cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
            pts = []
    if len(pts) >= 2:
        cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)


def save_fixed_roi_intensity_overlay_video(
    frames,
    df,
    contraction_rois,
    out_path,
    contraction_summary=None,
    fps=30,
    output_scale=0.5,
):
    """Save an original-image overlay for fixed ROI intensity contraction."""
    rois = parse_contraction_rois(contraction_rois)
    if not rois or df is None or len(df) == 0 or not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_scale = float(np.clip(float(output_scale), 0.1, 1.0))
    first = np.asarray(frames[0])
    height, width = first.shape[:2]
    out_size = (max(1, int(round(width * output_scale))), max(1, int(round(height * output_scale))))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        out_size,
        True,
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open fixed ROI video writer for: {out_path}")

    summary_map = {}
    if contraction_summary is not None and len(contraction_summary) > 0:
        for _, row in contraction_summary.iterrows():
            summary_map[str(row.get("roi_name"))] = row

    colors = [(0, 255, 0), (0, 210, 255), (255, 180, 0), (255, 0, 255)]
    signals = {}
    for roi_def in rois:
        col = f"contraction_{roi_def['name']}_mean"
        if col in df.columns:
            signals[roi_def["name"]] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        else:
            signals[roi_def["name"]] = np.full(len(df), np.nan, dtype=float)

    try:
        for i, frame in enumerate(frames[:len(df)]):
            bgr = _frame_to_bgr_for_overlay(frame)
            overlay = bgr.copy()
            cv2.rectangle(overlay, (0, 0), (min(width, 760), 92), (0, 0, 0), -1)
            bgr = cv2.addWeighted(overlay, 0.38, bgr, 0.62, 0)
            cv2.putText(
                bgr,
                f"fixed ROI intensity | frame {i + 1}/{min(len(frames), len(df))}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                "green boxes are fixed ROIs; plot shows mean gray intensity proxy",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            panel_h = min(220, max(120, height // 5))
            panel_w = min(width - 24, 560)
            panel_x = 12
            panel_y = max(0, height - panel_h - 12)
            panel = bgr[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w].copy()
            dark = np.zeros_like(panel)
            panel = cv2.addWeighted(panel, 0.25, dark, 0.75, 0)

            for idx, roi_def in enumerate(rois):
                color = colors[idx % len(colors)]
                x, y, w, h = _clip_tracker_box(
                    roi_def["x"], roi_def["y"], roi_def["w"], roi_def["h"], bgr.shape
                )
                cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 3)
                signal = signals[roi_def["name"]]
                value = signal[i] if i < signal.size else np.nan
                summary_row = summary_map.get(str(roi_def["name"]))
                period = np.nan
                valid = ""
                if summary_row is not None:
                    period = summary_row.get("period_sec", np.nan)
                    valid = " valid" if bool(summary_row.get("valid", False)) else " invalid"
                value_text = f"mean={value:.2f}" if np.isfinite(value) else "mean=nan"
                period_text = f"period={period:.1f}s" if np.isfinite(period) else "period=nan"
                label = f"{roi_def['name']} {value_text} {period_text}{valid}"
                cv2.putText(
                    bgr,
                    label,
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                row_h = max(36, panel_h // max(1, len(rois)))
                y0 = min(panel_h - row_h, idx * row_h)
                subpanel = panel[y0:y0 + row_h, 0:panel_w]
                _draw_signal_sparkline(
                    subpanel,
                    signal[: i + 1],
                    color,
                    f"ROI {roi_def['name']} mean gray",
                )

            bgr[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = panel
            cv2.rectangle(bgr, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (220, 220, 220), 1)
            if out_size != (width, height):
                bgr = cv2.resize(bgr, out_size, interpolation=cv2.INTER_AREA)
            writer.write(bgr)
    finally:
        writer.release()


def write_contraction_method_comparison(
    out_dir,
    intensity_summary,
    tracker_summary=None,
    min_tracker_displacement_px=1.0,
    min_tracker_match_score=0.75,
):
    """Write a combined comparison table for fixed ROI and tracker-like signals."""
    out_dir = Path(out_dir)
    rows = []
    if intensity_summary is not None and len(intensity_summary) > 0:
        for _, row in intensity_summary.iterrows():
            rows.append({
                "segment_name": row.get("segment_name", ""),
                "roi_name": str(row.get("roi_name", "")),
                "method": "fixed_roi_intensity",
                "signal_name": "mean_intensity",
                "period_sec": row.get("period_sec", np.nan),
                "period_min": row.get("period_min", np.nan),
                "amplitude": row.get("amplitude_peak_to_peak", np.nan),
                "match_score_mean": np.nan,
                "motion_px": np.nan,
                "valid_flag": bool(row.get("valid", False)),
                "usable_for_frequency": bool(row.get("valid", False)),
                "main_warning": "brightness proxy; sensitive to lighting and fixed ROI placement",
            })
    if tracker_summary is not None and len(tracker_summary) > 0:
        for _, row in tracker_summary.iterrows():
            signal_name = str(row.get("signal_name", ""))
            amplitude = row.get("amplitude_peak_to_peak", np.nan)
            match_score = row.get("mean_match_score", np.nan)
            valid = bool(row.get("valid", False))
            if signal_name in {"principal_displacement_px", "displacement_px"}:
                motion_px = amplitude
                usable = (
                    valid
                    and np.isfinite(amplitude)
                    and float(amplitude) >= float(min_tracker_displacement_px)
                    and np.isfinite(match_score)
                    and float(match_score) >= float(min_tracker_match_score)
                )
                warning = "" if usable else "tracked displacement too small or unstable"
            elif signal_name in {"flow_mag_mean", "diff_mean"}:
                motion_px = np.nan
                usable = (
                    valid
                    and np.isfinite(amplitude)
                    and float(amplitude) > 0
                    and np.isfinite(match_score)
                    and float(match_score) >= float(min_tracker_match_score)
                )
                warning = "optical change proxy; supporting evidence only"
            else:
                motion_px = np.nan
                usable = False
                warning = "unknown tracker signal"
            rows.append({
                "segment_name": row.get("segment_name", ""),
                "roi_name": str(row.get("roi_name", "")),
                "method": "tracker_like",
                "signal_name": signal_name,
                "period_sec": row.get("period_sec", np.nan),
                "period_min": row.get("period_min", np.nan),
                "amplitude": amplitude,
                "match_score_mean": match_score,
                "motion_px": motion_px,
                "valid_flag": valid,
                "usable_for_frequency": bool(usable),
                "main_warning": warning,
            })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "contraction_method_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def plot_contraction_method_summary(comparison, out_path):
    """Plot a compact fixed-ROI vs tracker-like contraction method summary."""
    if comparison is None or len(comparison) == 0:
        return

    df = comparison.copy()
    if "segment_name" not in df.columns or "signal_name" not in df.columns:
        return

    segment_order = [s for s in ["early", "middle", "late"] if s in set(df["segment_name"].astype(str))]
    if not segment_order:
        segment_order = sorted(df["segment_name"].astype(str).unique())

    key_signals = [
        ("fixed_roi_intensity", "mean_intensity"),
        ("tracker_like", "principal_displacement_px"),
        ("tracker_like", "flow_mag_mean"),
        ("tracker_like", "diff_mean"),
    ]
    rows = []
    for method, signal in key_signals:
        sub = df[(df["method"] == method) & (df["signal_name"] == signal)].copy()
        if sub.empty:
            continue
        sub["method_signal"] = f"{method}\n{signal}"
        rows.append(sub)
    if not rows:
        rows = [df.assign(method_signal=df["method"].astype(str) + "\n" + df["signal_name"].astype(str))]
    plot_df = pd.concat(rows, ignore_index=True, sort=False)

    labels = []
    x = []
    period = []
    amplitude = []
    usable = []
    colors = []
    for seg in segment_order:
        seg_df = plot_df[plot_df["segment_name"].astype(str) == seg]
        for method, signal in key_signals:
            sub = seg_df[(seg_df["method"] == method) & (seg_df["signal_name"] == signal)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            labels.append(f"{seg}\n{signal}")
            x.append(len(x))
            period.append(float(row.get("period_sec", np.nan)))
            amplitude.append(float(row.get("amplitude", np.nan)))
            usable_now = bool(row.get("usable_for_frequency", False))
            usable.append(usable_now)
            colors.append("#2ca02c" if usable_now else "#9e9e9e")

    if not x:
        return

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(x) * 0.9), 10), sharex=True)
    axes[0].bar(x, period, color=colors)
    axes[0].set_ylabel("Period (s)")
    axes[0].set_title("Contraction method comparison")
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(x, amplitude, color=colors)
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.25)

    usable_y = [1 if v else 0 for v in usable]
    axes[2].bar(x, usable_y, color=colors)
    axes[2].set_ylabel("Usable")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["no", "yes"])
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")

    fig.text(
        0.01,
        0.01,
        "Green=usable frequency estimate; gray=not recommended. Fixed ROI brightness is optical proxy; tracker flow/diff are supporting proxies.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_one_sampled_segment_worker(payload):
    """Analyze one early/middle/late segment in a separate process."""
    segment_name = payload["segment_name"]
    start = int(payload["start"])
    end = int(payload["end"])
    video_path = Path(payload["video_path"])
    segment_dir = Path(payload["segment_dir"])
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Prevent nested OpenCV/BLAS thread oversubscription inside each process.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    print(f"Analyzing {segment_name}: frames {start} to {end - 1}")
    print(f"{segment_name}: video frames kept in OpenCV BGR format for HSV segmentation")
    frames, names = _read_video_frame_range(video_path, start, end)
    rois = payload.get("rois", []) if payload["compute_contraction"] else []

    if payload["formal_use_temporal_body_recovery"]:
        df, debug_data, _ = analyze_timelapse_formal_temporal(
            frames,
            names=names,
            dt=payload["frame_interval"],
            pixel_size=payload["pixel_size"],
            roi=None,
            analysis_roi=payload["analysis_roi"],
            dish_center=payload["dish_center"],
            dish_radius=payload["dish_radius"],
            hsv_params=payload["hsv_params"],
            vein_params=payload["vein_params"],
            vein_every_n_frames=payload["vein_every_n_frames"],
            compute_body_skeleton=payload["compute_body_skeleton"],
            compute_front_speed=payload["compute_front_speed"],
            compute_contraction=payload["compute_contraction"],
            contraction_rois=rois,
            contraction_intensity_source=payload["contraction_intensity_source"],
            contraction_mask_layer=payload["contraction_mask_layer"],
            contraction_min_body_pixels=payload["contraction_min_body_pixels"],
            draw_contraction_rois=True,
            keep_debug=bool(payload["save_overlay_videos"]),
            formal_frame_scale=payload["formal_frame_scale"],
            n_jobs=payload.get("segment_internal_workers", 1),
            **payload["temporal_kwargs"],
        )
    else:
        # Segment-level multiprocessing is already active, so avoid nested pools.
        df, debug_data, _ = analyze_timelapse(
            frames,
            names=names,
            dt=payload["frame_interval"],
            pixel_size=payload["pixel_size"],
            roi=None,
            analysis_roi=payload["analysis_roi"],
            segmentation_profile="hsv_circle",
            dish_center=payload["dish_center"],
            dish_radius=payload["dish_radius"],
            hsv_params=payload["hsv_params"],
            vein_params=payload["vein_params"],
            vein_every_n_frames=payload["vein_every_n_frames"],
            compute_body_skeleton=payload["compute_body_skeleton"],
            compute_front_speed=payload["compute_front_speed"],
            compute_contraction=payload["compute_contraction"],
            contraction_rois=rois,
            contraction_intensity_source=payload["contraction_intensity_source"],
            contraction_mask_layer=payload["contraction_mask_layer"],
            contraction_min_body_pixels=payload["contraction_min_body_pixels"],
            keep_debug=bool(payload["save_overlay_videos"]),
            n_jobs=payload.get("segment_internal_workers", 1),
        )

    n_rows = len(df)
    original_frames = np.arange(start, start + n_rows, dtype=int)
    df.insert(0, "segment_name", segment_name)
    df.insert(1, "segment_frame", np.arange(n_rows, dtype=int))
    df.insert(2, "original_frame", original_frames)
    df.insert(3, "original_time_sec", original_frames.astype(float) * float(payload["frame_interval"]))
    if "time" in df.columns:
        df.rename(columns={"time": "segment_time_sec"}, inplace=True)
    df["sampling_design"] = "early_middle_late"
    df["frames_per_segment_requested"] = int(payload["frames_per_segment"])

    segment_csv = segment_dir / f"sampled_frames_{segment_name}.csv"

    summary = None
    tracker_summary = None
    if payload["compute_contraction"] and rois:
        summary = analyze_contraction_rois(
            df=df,
            contraction_rois=rois,
            dt=payload["frame_interval"],
            min_period_sec=payload["contraction_min_period_sec"],
            max_period_sec=payload["contraction_max_period_sec"],
            min_valid_samples=payload["contraction_min_valid_samples"],
            interpolate_short_gaps=payload["contraction_interpolate_short_gaps"],
            max_interpolation_gap=payload["contraction_max_interpolation_gap"],
            out_dir=segment_dir,
            save_plots=payload["save_contraction_plots"],
        )
        summary.insert(0, "segment_name", segment_name)
        summary.insert(1, "segment_start_frame", start)
        summary.insert(2, "segment_end_frame_inclusive", int(start + n_rows - 1))
        summary.to_csv(segment_dir / f"contraction_summary_{segment_name}.csv", index=False, encoding="utf-8-sig")
        plot_contraction_summary(summary, segment_dir / f"plasmodium_summary_contraction_{segment_name}.png")

    if payload["save_overlay_videos"] and rois:
        save_fixed_roi_intensity_overlay_video(
            frames,
            df,
            rois,
            segment_dir / f"fixed_roi_intensity_overlay_{segment_name}.mp4",
            contraction_summary=summary,
            fps=payload["overlay_video_fps"],
            output_scale=payload.get("tracker_overlay_output_scale", payload["formal_frame_scale"]),
        )

    if payload.get("compute_tracker_contraction", False) and rois:
        tracker_summary = add_tracker_contraction_metrics_to_df(
            df=df,
            frames=frames,
            contraction_rois=rois,
            dt=payload["frame_interval"],
            out_dir=segment_dir,
            min_period_sec=payload["contraction_min_period_sec"],
            max_period_sec=payload["contraction_max_period_sec"],
            min_valid_samples=payload["contraction_min_valid_samples"],
            search_radius=payload.get("tracker_search_radius", 80),
            min_match_score=payload.get("tracker_min_match_score", 0.25),
            template_update_alpha=payload.get("tracker_template_update_alpha", 0.03),
            save_plots=payload.get("save_tracker_plots", True),
        )
        if tracker_summary is not None and not tracker_summary.empty:
            tracker_summary.insert(0, "segment_name", segment_name)
            tracker_summary.insert(1, "segment_start_frame", start)
            tracker_summary.insert(2, "segment_end_frame_inclusive", int(start + n_rows - 1))
            tracker_summary.to_csv(
                segment_dir / f"tracker_contraction_summary_{segment_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if payload["save_overlay_videos"]:
            save_tracker_overlay_video(
                frames,
                df,
                rois,
                segment_dir / f"tracker_overlay_{segment_name}.mp4",
                fps=payload["overlay_video_fps"],
                output_scale=payload.get("tracker_overlay_output_scale", payload["formal_frame_scale"]),
            )

    df.to_csv(segment_csv, index=False, encoding="utf-8-sig")

    if payload["save_overlay_videos"] and debug_data:
        save_overlay_video(
            debug_data,
            segment_dir / f"overlay_{segment_name}.mp4",
            fps=payload["overlay_video_fps"],
            contraction_rois=rois,
            draw_contraction_rois=True,
        )

    del frames, names, debug_data
    gc.collect()
    return segment_name, df, summary, tracker_summary

def run_sampled_early_middle_late_analysis(
    video_path,
    out_dir,
    frames_per_segment,
    roi_selection_mode,
    reuse_saved_rois,
    frame_interval,
    pixel_size,
    analysis_roi,
    dish_center,
    dish_radius,
    hsv_params,
    vein_params,
    vein_every_n_frames,
    compute_body_skeleton,
    compute_front_speed,
    compute_contraction,
    contraction_intensity_source,
    contraction_mask_layer,
    contraction_min_body_pixels,
    contraction_min_period_sec,
    contraction_max_period_sec,
    contraction_min_valid_samples,
    contraction_interpolate_short_gaps,
    contraction_max_interpolation_gap,
    save_contraction_plots,
    compute_tracker_contraction,
    tracker_search_radius,
    tracker_min_match_score,
    tracker_template_update_alpha,
    save_tracker_plots,
    tracker_overlay_output_scale,
    save_overlay_videos,
    overlay_video_fps,
    formal_use_temporal_body_recovery,
    formal_frame_scale,
    temporal_kwargs,
    enable_cpu_parallel=True,
    cpu_workers=3,
    sample_parallel_strategy="frame",
):
    """
    Analyze early/middle/late video windows independently and export full rows.

    Temporal derivatives and contraction FFT are calculated independently inside
    each segment so the large gaps between segments are never treated as
    continuous acquisition.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        raise ValueError("Could not determine video frame count.")

    ranges = _sampled_segment_ranges(total_frames, frames_per_segment)
    ranges_path = out_dir / "selected_frame_ranges.json"
    import json
    ranges_payload = {
        "video": str(video_path),
        "total_frames": total_frames,
        "frames_per_segment_requested": int(frames_per_segment),
        "frame_interval_sec": float(frame_interval),
        "ranges": {
            k: {"start": int(v[0]), "end_exclusive": int(v[1]), "count": int(v[1] - v[0])}
            for k, v in ranges.items()
        },
    }
    ranges_path.write_text(
        json.dumps(ranges_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    roi_json_path = out_dir / "selected_contraction_rois.json"
    saved_roi_payload = {}
    if reuse_saved_rois and roi_json_path.exists():
        try:
            saved_roi_payload = json.loads(roi_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.warn(f"Could not load saved ROI JSON; UI will be used. {exc}")
            saved_roi_payload = {}

    roi_selection_mode = str(roi_selection_mode).lower().strip()
    if roi_selection_mode not in {"single_reference", "per_segment"}:
        raise ValueError(
            "ROI_SELECTION_MODE must be 'single_reference' or 'per_segment'."
        )

    segment_rois = {}
    if saved_roi_payload:
        raw = saved_roi_payload.get("segment_rois", {})
        try:
            segment_rois = {
                key: parse_contraction_rois(value) for key, value in raw.items()
            }
            print(f"Loaded saved contraction ROIs: {roi_json_path}")
        except Exception as exc:
            warnings.warn(f"Saved ROI definitions are invalid; UI will be used. {exc}")
            segment_rois = {}

    if compute_contraction and not segment_rois:
        if roi_selection_mode == "single_reference":
            mid_start, mid_end = ranges["middle"]
            ref_idx = (mid_start + mid_end - 1) // 2
            ref = _read_video_frame_at(video_path, ref_idx)
            selected = select_contraction_rois_ui(
                ref,
                window_title="Select ROIs - shared by early/middle/late",
                default_prefix="roi",
            )
            if not selected:
                raise ValueError("No contraction ROI was selected.")
            segment_rois = {name: selected for name in ranges}
            _save_sampled_roi_preview(
                ref, selected,
                out_dir / "roi_preview_shared.png",
                title=f"Shared ROI reference frame {ref_idx}",
            )
        else:
            for segment_name, (start, end) in ranges.items():
                ref_idx = (start + end - 1) // 2
                ref = _read_video_frame_at(video_path, ref_idx)
                selected = select_contraction_rois_ui(
                    ref,
                    window_title=f"Select ROIs - {segment_name} frame {ref_idx}",
                    default_prefix="roi",
                )
                if not selected:
                    raise ValueError(
                        f"No contraction ROI was selected for {segment_name}."
                    )
                segment_rois[segment_name] = selected
                _save_sampled_roi_preview(
                    ref, selected,
                    out_dir / f"roi_preview_{segment_name}.png",
                    title=f"{segment_name} reference frame {ref_idx}",
                )

        serializable = {
            name: [
                [r["name"], r["x"], r["y"], r["w"], r["h"]]
                for r in parse_contraction_rois(rois)
            ]
            for name, rois in segment_rois.items()
        }
        roi_json_path.write_text(
            json.dumps(
                {
                    "selection_mode": roi_selection_mode,
                    "coordinate_system": "original_video_pixels",
                    "segment_rois": serializable,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved contraction ROIs: {roi_json_path}")

    sample_parallel_strategy = str(sample_parallel_strategy or "frame").lower().strip()
    if sample_parallel_strategy not in {"frame", "segment", "hybrid"}:
        raise ValueError("sample_parallel_strategy must be 'frame', 'segment', or 'hybrid'.")
    total_worker_budget = max(1, int(cpu_workers or 1)) if enable_cpu_parallel else 1
    if sample_parallel_strategy == "frame":
        segment_worker_count = 1
        segment_internal_workers = total_worker_budget
    elif sample_parallel_strategy == "segment":
        segment_worker_count = min(total_worker_budget, len(ranges))
        segment_internal_workers = 1
    else:
        segment_worker_count = min(total_worker_budget, len(ranges))
        segment_internal_workers = max(1, total_worker_budget // max(1, segment_worker_count))

    print(
        "Sampled contraction parallelism: "
        f"strategy={sample_parallel_strategy}, "
        f"segment_workers={segment_worker_count}, "
        f"workers_per_segment={segment_internal_workers}, "
        f"cpu_budget={total_worker_budget}"
    )

    all_dfs = []
    all_contraction_summaries = []
    all_tracker_summaries = []
    payloads = []
    for segment_name, (seg_start, seg_end) in ranges.items():
        payloads.append({
            "segment_name": segment_name,
            "start": int(seg_start),
            "end": int(seg_end),
            "video_path": str(video_path),
            "segment_dir": str(out_dir / segment_name),
            "frames_per_segment": int(frames_per_segment),
            "rois": segment_rois.get(segment_name, []) if compute_contraction else [],
            "frame_interval": frame_interval,
            "pixel_size": pixel_size,
            "analysis_roi": analysis_roi,
            "dish_center": dish_center,
            "dish_radius": dish_radius,
            "hsv_params": hsv_params,
            "vein_params": vein_params,
            "vein_every_n_frames": vein_every_n_frames,
            "compute_body_skeleton": compute_body_skeleton,
            "compute_front_speed": compute_front_speed,
            "compute_contraction": compute_contraction,
            "contraction_intensity_source": contraction_intensity_source,
            "contraction_mask_layer": contraction_mask_layer,
            "contraction_min_body_pixels": contraction_min_body_pixels,
            "contraction_min_period_sec": contraction_min_period_sec,
            "contraction_max_period_sec": contraction_max_period_sec,
            "contraction_min_valid_samples": contraction_min_valid_samples,
            "contraction_interpolate_short_gaps": contraction_interpolate_short_gaps,
            "contraction_max_interpolation_gap": contraction_max_interpolation_gap,
            "save_contraction_plots": save_contraction_plots,
            "compute_tracker_contraction": compute_tracker_contraction,
            "tracker_search_radius": tracker_search_radius,
            "tracker_min_match_score": tracker_min_match_score,
            "tracker_template_update_alpha": tracker_template_update_alpha,
            "save_tracker_plots": save_tracker_plots,
            "tracker_overlay_output_scale": tracker_overlay_output_scale,
            "save_overlay_videos": save_overlay_videos,
            "overlay_video_fps": overlay_video_fps,
            "formal_use_temporal_body_recovery": formal_use_temporal_body_recovery,
            "formal_frame_scale": formal_frame_scale,
            "temporal_kwargs": temporal_kwargs,
            "segment_internal_workers": int(segment_internal_workers),
        })

    worker_count = int(segment_worker_count)
    if enable_cpu_parallel and worker_count > 1:
        print(f"Using segment-level multiprocessing: {worker_count} workers")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_one_sampled_segment_worker, p) for p in payloads]
            results = [future.result() for future in as_completed(futures)]
        order = {name: i for i, name in enumerate(("early", "middle", "late"))}
        results.sort(key=lambda item: order.get(item[0], 99))
    else:
        print("Using single-process sampled analysis")
        results = [_run_one_sampled_segment_worker(p) for p in payloads]

    for segment_name, df, summary, tracker_summary in results:
        all_dfs.append(df)
        if summary is not None and not summary.empty:
            all_contraction_summaries.append(summary)
        if tracker_summary is not None and not tracker_summary.empty:
            all_tracker_summaries.append(tracker_summary)

    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    combined_path = out_dir / "sampled_frames_all_metrics.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    combined_summary = pd.DataFrame()
    if all_contraction_summaries:
        combined_summary = pd.concat(
            all_contraction_summaries, ignore_index=True, sort=False
        )
        combined_summary.to_csv(
            out_dir / "contraction_summary_all_segments.csv",
            index=False,
            encoding="utf-8-sig",
        )

    tracker_combined_summary = pd.DataFrame()
    if all_tracker_summaries:
        tracker_combined_summary = pd.concat(
            all_tracker_summaries, ignore_index=True, sort=False
        )
        tracker_combined_summary.to_csv(
            out_dir / "tracker_contraction_summary_all_segments.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not combined_summary.empty or not tracker_combined_summary.empty:
        comparison = write_contraction_method_comparison(
            out_dir,
            combined_summary,
            tracker_combined_summary,
        )
        comparison_plot_path = out_dir / "plasmodium_summary_contraction_methods.png"
        plot_contraction_method_summary(comparison, comparison_plot_path)
        print(f"Saved contraction method comparison: {out_dir / 'contraction_method_comparison.csv'}")
        print(f"Saved contraction method summary plot: {comparison_plot_path}")

    print("\nEarly/middle/late sampled analysis completed.")
    print(f"Combined full data: {combined_path}")
    print(f"Frame ranges: {ranges_path}")
    if compute_contraction:
        print(f"ROI definitions: {roi_json_path}")
    return combined


if __name__ == "__main__":
    INPUT_MODE = "video"  # "folder" or "video"
    INPUT_PATH = r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\0526轉.mp4"
    OUTPUT_DIR = r"results_plasmodium"

    # ============================================================
    # Run mode selector
    # ============================================================
    # Recommended modes:
    #
    # "complete"
    #     Run everything: ROI contraction/tracker sampling, full formal analysis,
    #     formal overlay video, and front/middle/rear validation clips.
    #
    # "formal_only"
    #     Run only the full formal analysis, CSVs, summary plots, debug examples,
    #     and formal overlay video. No validation clips, no sampled ROI tracker run.
    #
    # "validation_all"
    #     Export front/middle/rear validation clips and stills only.
    #
    # "validation_front", "validation_middle", "validation_rear"
    #     Export one validation section only. Useful for quick parameter checks.
    #
    # "formal_with_validation"
    #     Run full formal analysis and then front/middle/rear validation clips.
    #
    # "contraction_roi"
    #     Run only early/middle/late ROI contraction + tracker outputs.
    #     This is the fastest mode for redrawing/testing contraction ROIs.
    #
    # "manual_validation_export"
    #     Export front/middle/rear raw images, auto masks, and overlays for
    #     manual ground-truth mask drawing. No formal analysis is run.
    #
    # "manual_validation_annotate"
    #     Open the simple manual mask editor for the exported package.
    #
    # "manual_validation_score"
    #     Calculate IoU/Dice/Precision/Recall/Area error after manual masks exist.
    #
    # Legacy aliases still work:
    #   formal -> formal_only
    #   test_front_middle_rear -> validation_all
    #   test_rear -> validation_rear
    #   formal_plus_front_middle_rear_test -> formal_with_validation
    #   sample_early_middle_late -> contraction_roi
    # Official default: run the complete formal analysis for one full video.
    # Change this value only when you intentionally want validation clips or ROI-only testing.
    RUN_MODE = "contraction_roi"

    MODE_ALIASES = {
        "formal": "formal_only",
        "full": "complete",
        "all": "complete",
        "validation": "validation_all",
        "test_all": "validation_all",
        "test_front": "validation_front",
        "test_middle": "validation_middle",
        "roi_only": "contraction_roi",
        "roi_contraction": "contraction_roi",
        "contraction_only": "contraction_roi",
        "manual_export": "manual_validation_export",
        "manual_annotate": "manual_validation_annotate",
        "manual_score": "manual_validation_score",
        "ground_truth_export": "manual_validation_export",
        "ground_truth_score": "manual_validation_score",
        "test_front_middle_rear": "validation_all",
        "test_rear": "validation_rear",
        "formal_plus_front_middle_rear_test": "formal_with_validation",
        "sample_early_middle_late": "contraction_roi",
    }
    VALID_RUN_MODES = {
        "complete",
        "formal_only",
        "formal_with_validation",
        "validation_all",
        "validation_front",
        "validation_middle",
        "validation_rear",
        "contraction_roi",
        "manual_validation_export",
        "manual_validation_annotate",
        "manual_validation_score",
        *MODE_ALIASES.keys(),
    }
    if RUN_MODE not in VALID_RUN_MODES:
        raise ValueError(
            f"RUN_MODE must be one of {sorted(VALID_RUN_MODES)}, got {RUN_MODE!r}"
        )

    CANONICAL_RUN_MODE = MODE_ALIASES.get(RUN_MODE, RUN_MODE)
    RUN_FORMAL_ANALYSIS = CANONICAL_RUN_MODE in {
        "formal_only",
        "formal_with_validation",
        "complete",
    }
    RUN_TEST_CLIPS = CANONICAL_RUN_MODE in {
        "validation_all",
        "validation_front",
        "validation_middle",
        "validation_rear",
        "formal_with_validation",
        "complete",
    }
    RUN_SAMPLE_CONTRACTION = CANONICAL_RUN_MODE in {"contraction_roi", "complete"}
    RUN_MANUAL_VALIDATION_EXPORT = CANONICAL_RUN_MODE == "manual_validation_export"
    RUN_MANUAL_VALIDATION_ANNOTATE = CANONICAL_RUN_MODE == "manual_validation_annotate"
    RUN_MANUAL_VALIDATION_SCORE = CANONICAL_RUN_MODE == "manual_validation_score"
    TEST_ONLY_MODE = CANONICAL_RUN_MODE in {
        "validation_all",
        "validation_front",
        "validation_middle",
        "validation_rear",
    }
    VALIDATION_SECTION_BY_MODE = {
        "validation_front": ("front",),
        "validation_middle": ("middle",),
        "validation_rear": ("rear",),
    }
    SELECTED_VALIDATION_SECTIONS = VALIDATION_SECTION_BY_MODE.get(
        CANONICAL_RUN_MODE,
        ("front", "middle", "rear"),
    )

    # ============================================================
    # Basic processing settings
    # ============================================================
    CLEAN_OUTPUT_DEBUG = True
    ENABLE_CPU_PARALLEL = True
    CPU_WORKERS = max(8, int(os.cpu_count() or 8))
    LOW_MEMORY_VIDEO_STREAM = True
    PARALLEL_BATCH_FRAMES = 32
    # contraction_roi parallel strategy:
    # "frame" = one segment at a time, use CPU_WORKERS inside each segment; safest high-core mode.
    # "segment" = early/middle/late simultaneously, one worker inside each segment.
    # "hybrid" = early/middle/late simultaneously, split CPU_WORKERS across segments.
    SAMPLE_PARALLEL_STRATEGY = "hybrid"

    # Use the real acquisition interval, not the exported video playback FPS.
    # Example: images captured every 10 s and exported as 30 FPS video still use 10.0 here.
    FRAME_INTERVAL = 10.0
    PIXEL_SIZE = 1.0
    ROI = None  # Example: ROI = (x, y, w, h)

    # Early/middle/late sampled analysis
    SAMPLE_FRAMES_PER_SEGMENT = 300
    USE_ROI_SELECTION_UI = True
    ROI_SELECTION_MODE = "per_segment"  # "per_segment" or "single_reference"
    REUSE_SAVED_CONTRACTION_ROIS = False
    # Intensity-based contraction analysis
    COMPUTE_CONTRACTION = True
    # ROI formats: ("name", x, y, w, h) or (x, y, w, h).
    CONTRACTION_ROIS = [
        # ("center", 1100, 820, 100, 100),
        # ("front", 900, 950, 100, 100),
    ]
    # Gray is recommended for reproducible amplitude and frequency comparison.
    CONTRACTION_INTENSITY_SOURCE = "gray"  # "gray", "enhanced", or "corrected"
    CONTRACTION_MASK_LAYER = "clean"  # "clean", "occupied", "envelope", or "none"
    CONTRACTION_MIN_BODY_PIXELS = 50
    CONTRACTION_MIN_PERIOD_SEC = 30.0
    CONTRACTION_MAX_PERIOD_SEC = 600.0
    CONTRACTION_MIN_VALID_SAMPLES = 20
    CONTRACTION_INTERPOLATE_SHORT_GAPS = True
    CONTRACTION_MAX_INTERPOLATION_GAP = 3
    SAVE_CONTRACTION_PLOTS = True
    DRAW_CONTRACTION_ROIS_ON_OVERLAY = True

    # Tracker-like contraction analysis, closer to Tracker Video Analysis:
    # first-frame ROI becomes a template, then the program tracks x/y displacement.
    COMPUTE_TRACKER_CONTRACTION = True
    TRACKER_SEARCH_RADIUS = 80
    TRACKER_MIN_MATCH_SCORE = 0.25
    TRACKER_TEMPLATE_UPDATE_ALPHA = 0.03
    SAVE_TRACKER_PLOTS = True
    TRACKER_OVERLAY_OUTPUT_SCALE = 0.5

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
    MANUAL_VALIDATION_SAMPLES_PER_SECTION = 10
    MANUAL_VALIDATION_SECTION_FRAMES = 300
    MANUAL_VALIDATION_DISPLAY_SCALE = 0.65
    MANUAL_VALIDATION_BRUSH_RADIUS = 5
    MANUAL_VALIDATION_OVERLAY_ALPHA = 0.23
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
    VALIDATION_CLIP_BATCH_FRAMES = 32
    VALIDATION_CLIP_WRITE_VIDEO = True
    # When validation_all is used, also compute formal-style CSV and summary plots
    # for the same front/middle/rear windows under results_plasmodium/validation_analysis/.
    VALIDATION_ANALYZE_SECTION_DATA = CANONICAL_RUN_MODE == "validation_all"
    VALIDATION_WORKERS_PER_SECTION = CPU_WORKERS if VALIDATION_ANALYZE_SECTION_DATA else 1

    # Temporal recovery can be used in both validation clips and formal metrics.
    # When RUN_FORMAL_ANALYSIS is True and USE_TEMPORAL_BODY_RECOVERY is True,
    # formal CSV, summary plot, debug examples, and overlay use temporal masks.
    USE_TEMPORAL_BODY_RECOVERY = True
    FORMAL_USE_TEMPORAL_BODY_RECOVERY = bool(
        USE_TEMPORAL_BODY_RECOVERY
        and (RUN_FORMAL_ANALYSIS or RUN_SAMPLE_CONTRACTION)
    )
    TEMPORAL_SKIP_EDGE_FRAMES = 10
    TEMPORAL_WINDOW = 5
    # Match the parameter set that produced the successful test_rear result.
    TEMPORAL_MIN_HITS = 4
    TEMPORAL_CHROMA_MIN = 10
    TEMPORAL_MIN_CHANNEL = 35
    TEMPORAL_MAX_VALUE = 240
    TEMPORAL_RIM_EXCLUSION_PX = 30
    TEMPORAL_CONNECT_RADIUS = 8
    TEMPORAL_MIN_OBJ_SIZE = 20
    TEMPORAL_EXTERIOR_ONLY = True
    TEMPORAL_BLOCK_HOLE_MAX_AREA = 0
    TEMPORAL_FINAL_SMOOTH_WINDOW = 1
    TEMPORAL_FINAL_SMOOTH_MIN_HITS = 1

    # Formal-analysis settings. Test-only modes skip the formal analysis, so
    # these values only matter when RUN_FORMAL_ANALYSIS is True.
    MAX_FRAMES = FORMAL_ANALYSIS_MAX_FRAMES
    SAVE_OVERLAY_VIDEO = True
    OVERLAY_VIDEO_FPS = 30
    # Keep formal analysis at original resolution so thin plasmodium tubes and
    # network topology are not lost before HSV/temporal segmentation.
    FORMAL_FRAME_SCALE = 1.0
    # Overlay scale only controls exported video size; it does not affect metrics.
    OVERLAY_VIDEO_OUTPUT_SCALE = 0.5
    VEIN_EVERY_N_FRAMES = 0  # Disable separate vein/ridge analysis for HSV body-mask comparison.
    COMPUTE_BODY_SKELETON = True
    COMPUTE_FRONT_SPEED = True
    RUN_QC_VALIDATION = bool(RUN_FORMAL_ANALYSIS)
    RUN_SEGMENT_TESTS = False

    parsed_contraction_rois = parse_contraction_rois(CONTRACTION_ROIS)
    # In sampled UI mode, ROIs are selected after startup, so an empty static
    # CONTRACTION_ROIS list is expected and should not trigger a false warning.
    roi_will_be_selected_by_ui = bool(
        RUN_SAMPLE_CONTRACTION and USE_ROI_SELECTION_UI
    )
    if (
        COMPUTE_CONTRACTION
        and not parsed_contraction_rois
        and not roi_will_be_selected_by_ui
    ):
        warnings.warn(
            "COMPUTE_CONTRACTION=True but CONTRACTION_ROIS is empty; "
            "contraction analysis will be skipped."
        )

    RUN_STAGES = []
    if RUN_SAMPLE_CONTRACTION:
        RUN_STAGES.append("sampled ROI contraction/tracker")
    if RUN_FORMAL_ANALYSIS:
        RUN_STAGES.append("formal morphology/temporal analysis")
    if RUN_TEST_CLIPS:
        RUN_STAGES.append("validation clips/stills")
    if RUN_MANUAL_VALIDATION_EXPORT:
        RUN_STAGES.append("manual validation package export")
    if RUN_MANUAL_VALIDATION_ANNOTATE:
        RUN_STAGES.append("manual validation annotation")
    if RUN_MANUAL_VALIDATION_SCORE:
        RUN_STAGES.append("manual validation scoring")

    print("Run configuration:")
    print("  MODE = HSV body-mask analysis")
    print(f"  RUN_STAGES = {'; '.join(RUN_STAGES) if RUN_STAGES else 'none'}")
    print(f"  RUN_MODE = {RUN_MODE}")
    print(f"  CANONICAL_RUN_MODE = {CANONICAL_RUN_MODE}")
    print(f"  RUN_FORMAL_ANALYSIS = {RUN_FORMAL_ANALYSIS}")
    print(f"  RUN_TEST_CLIPS = {RUN_TEST_CLIPS}")
    print(f"  RUN_SAMPLE_CONTRACTION = {RUN_SAMPLE_CONTRACTION}")
    print(f"  RUN_MANUAL_VALIDATION_EXPORT = {RUN_MANUAL_VALIDATION_EXPORT}")
    print(f"  RUN_MANUAL_VALIDATION_ANNOTATE = {RUN_MANUAL_VALIDATION_ANNOTATE}")
    print(f"  RUN_MANUAL_VALIDATION_SCORE = {RUN_MANUAL_VALIDATION_SCORE}")
    print(f"  TEST_ONLY_MODE = {TEST_ONLY_MODE}")
    print(f"  FORMAL_ANALYSIS_MAX_TIME_SEC = {FORMAL_ANALYSIS_MAX_TIME_SEC}")
    print(f"  FORMAL_ANALYSIS_MAX_FRAMES = {FORMAL_ANALYSIS_MAX_FRAMES}")
    print(f"  MAX_FRAMES = {MAX_FRAMES}")
    print(f"  FORMAL_FRAME_SCALE = {FORMAL_FRAME_SCALE}")
    print(f"  OVERLAY_VIDEO_OUTPUT_SCALE = {OVERLAY_VIDEO_OUTPUT_SCALE}")
    print(f"  SAVE_OVERLAY_VIDEO = {SAVE_OVERLAY_VIDEO}")
    print(f"  VEIN_EVERY_N_FRAMES = {VEIN_EVERY_N_FRAMES}")
    print(f"  COMPUTE_BODY_SKELETON = {COMPUTE_BODY_SKELETON}")
    print(f"  COMPUTE_FRONT_SPEED = {COMPUTE_FRONT_SPEED}")
    print(f"  RUN_QC_VALIDATION = {RUN_QC_VALIDATION}")
    print(f"  COMPUTE_CONTRACTION = {COMPUTE_CONTRACTION}")
    print(f"  CONTRACTION_ROIS = {CONTRACTION_ROIS}")
    print(f"  USE_ROI_SELECTION_UI = {USE_ROI_SELECTION_UI}")
    print(f"  REUSE_SAVED_CONTRACTION_ROIS = {REUSE_SAVED_CONTRACTION_ROIS}")
    print(f"  CONTRACTION_INTENSITY_SOURCE = {CONTRACTION_INTENSITY_SOURCE}")
    print(f"  CONTRACTION_MASK_LAYER = {CONTRACTION_MASK_LAYER}")
    print(f"  COMPUTE_TRACKER_CONTRACTION = {COMPUTE_TRACKER_CONTRACTION}")
    print(f"  TRACKER_SEARCH_RADIUS = {TRACKER_SEARCH_RADIUS}")
    print(f"  TRACKER_MIN_MATCH_SCORE = {TRACKER_MIN_MATCH_SCORE}")
    print(
        "  CONTRACTION_PERIOD_RANGE_SEC = "
        f"{CONTRACTION_MIN_PERIOD_SEC} to {CONTRACTION_MAX_PERIOD_SEC}"
    )
    print(f"  CLEAN_OUTPUT_DEBUG = {CLEAN_OUTPUT_DEBUG}")
    print(f"  ENABLE_CPU_PARALLEL = {ENABLE_CPU_PARALLEL}")
    print(f"  CPU_AVAILABLE = {os.cpu_count()}")
    print(f"  CPU_WORKERS = {CPU_WORKERS}")
    print(f"  LOW_MEMORY_VIDEO_STREAM = {LOW_MEMORY_VIDEO_STREAM}")
    print(f"  PARALLEL_BATCH_FRAMES = {PARALLEL_BATCH_FRAMES}")
    print(f"  SAMPLE_PARALLEL_STRATEGY = {SAMPLE_PARALLEL_STRATEGY}")
    print(f"  RUN_RELIABILITY_VALIDATION = {RUN_RELIABILITY_VALIDATION}")
    print(f"  RELIABILITY_VALIDATE_VEINS = {RELIABILITY_VALIDATE_VEINS}")
    print(f"  MANUAL_VALIDATION_SAMPLES_PER_SECTION = {MANUAL_VALIDATION_SAMPLES_PER_SECTION}")
    print(f"  MANUAL_VALIDATION_SECTION_FRAMES = {MANUAL_VALIDATION_SECTION_FRAMES}")
    print(f"  RUN_LIGHTWEIGHT_VALIDATION_CLIPS = {RUN_LIGHTWEIGHT_VALIDATION_CLIPS}")
    print(f"  VALIDATION_ONLY_MODE = {VALIDATION_ONLY_MODE}")
    print(f"  VALIDATION_CLIP_SECTIONS = {VALIDATION_CLIP_SECTIONS}")
    print(f"  VALIDATION_CLIP_SECTION_FRAMES = {VALIDATION_CLIP_SECTION_FRAMES}")
    print(f"  VALIDATION_CLIP_CPU_WORKERS = {VALIDATION_CLIP_CPU_WORKERS}")
    print(f"  VALIDATION_CLIP_BATCH_FRAMES = {VALIDATION_CLIP_BATCH_FRAMES}")
    print(f"  VALIDATION_CLIP_WRITE_VIDEO = {VALIDATION_CLIP_WRITE_VIDEO}")
    print(f"  VALIDATION_ANALYZE_SECTION_DATA = {VALIDATION_ANALYZE_SECTION_DATA}")
    print(f"  VALIDATION_WORKERS_PER_SECTION = {VALIDATION_WORKERS_PER_SECTION}")
    print("  VALIDATION_WORKERS_NOTE = capped for Windows/Spyder stability")
    print(f"  USE_TEMPORAL_BODY_RECOVERY = {USE_TEMPORAL_BODY_RECOVERY}")
    print(f"  FORMAL_USE_TEMPORAL_BODY_RECOVERY = {FORMAL_USE_TEMPORAL_BODY_RECOVERY}")
    print(
        "  TEMPORAL_SCOPE = "
        + (
            "formal_metrics_and_validation"
            if FORMAL_USE_TEMPORAL_BODY_RECOVERY
            else "validation_only" if USE_TEMPORAL_BODY_RECOVERY else "off"
        )
    )
    print(f"  TEMPORAL_SKIP_EDGE_FRAMES = {TEMPORAL_SKIP_EDGE_FRAMES}")
    print("  FORMAL_EDGE_EXCLUSION = first/last frames skipped from formal outputs")
    print(f"  TEMPORAL_WINDOW = {TEMPORAL_WINDOW}")
    print(f"  TEMPORAL_MIN_HITS = {TEMPORAL_MIN_HITS}")
    print(f"  TEMPORAL_CHROMA_MIN = {TEMPORAL_CHROMA_MIN}")
    print(f"  TEMPORAL_RIM_EXCLUSION_PX = {TEMPORAL_RIM_EXCLUSION_PX}")
    print(f"  TEMPORAL_CONNECT_RADIUS = {TEMPORAL_CONNECT_RADIUS}")
    print(f"  TEMPORAL_EXTERIOR_ONLY = {TEMPORAL_EXTERIOR_ONLY}")
    print(f"  TEMPORAL_BLOCK_HOLE_MAX_AREA = {TEMPORAL_BLOCK_HOLE_MAX_AREA}")
    print(f"  TEMPORAL_FINAL_SMOOTH_WINDOW = {TEMPORAL_FINAL_SMOOTH_WINDOW}")
    print(f"  TEMPORAL_FINAL_SMOOTH_MIN_HITS = {TEMPORAL_FINAL_SMOOTH_MIN_HITS}")
    print("  TEMPORAL_PROFILE = test_rear_matched")

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
        use_yellow_chroma_extension=True,  # Optimized: enable constrained RGB/yellow-chroma extension.
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
        use_vesselness_recovery=True,
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

    if RUN_SAMPLE_CONTRACTION:
        if INPUT_MODE != "video":
            raise ValueError("contraction_roi/complete modes require INPUT_MODE='video'.")
        if not USE_ROI_SELECTION_UI and not CONTRACTION_ROIS:
            raise ValueError(
                "Set USE_ROI_SELECTION_UI=True or provide CONTRACTION_ROIS."
            )
        if not USE_ROI_SELECTION_UI:
            import json
            fixed = parse_contraction_rois(CONTRACTION_ROIS)
            payload = {
                "selection_mode": "single_reference",
                "coordinate_system": "original_video_pixels",
                "segment_rois": {
                    key: [[r["name"], r["x"], r["y"], r["w"], r["h"]] for r in fixed]
                    for key in ("early", "middle", "late")
                },
            }
            (out_dir / "selected_contraction_rois.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        run_sampled_early_middle_late_analysis(
            video_path=input_path,
            out_dir=out_dir,
            frames_per_segment=SAMPLE_FRAMES_PER_SEGMENT,
            roi_selection_mode=ROI_SELECTION_MODE,
            reuse_saved_rois=(REUSE_SAVED_CONTRACTION_ROIS or not USE_ROI_SELECTION_UI),
            frame_interval=FRAME_INTERVAL,
            pixel_size=PIXEL_SIZE,
            analysis_roi=ANALYSIS_ROI,
            dish_center=DISH_CENTER,
            dish_radius=DISH_RADIUS,
            hsv_params=hsv_params,
            vein_params=vein_params,
            vein_every_n_frames=VEIN_EVERY_N_FRAMES,
            compute_body_skeleton=COMPUTE_BODY_SKELETON,
            compute_front_speed=COMPUTE_FRONT_SPEED,
            compute_contraction=COMPUTE_CONTRACTION,
            contraction_intensity_source=CONTRACTION_INTENSITY_SOURCE,
            contraction_mask_layer=CONTRACTION_MASK_LAYER,
            contraction_min_body_pixels=CONTRACTION_MIN_BODY_PIXELS,
            contraction_min_period_sec=CONTRACTION_MIN_PERIOD_SEC,
            contraction_max_period_sec=CONTRACTION_MAX_PERIOD_SEC,
            contraction_min_valid_samples=CONTRACTION_MIN_VALID_SAMPLES,
            contraction_interpolate_short_gaps=CONTRACTION_INTERPOLATE_SHORT_GAPS,
            contraction_max_interpolation_gap=CONTRACTION_MAX_INTERPOLATION_GAP,
            save_contraction_plots=SAVE_CONTRACTION_PLOTS,
            compute_tracker_contraction=COMPUTE_TRACKER_CONTRACTION,
            tracker_search_radius=TRACKER_SEARCH_RADIUS,
            tracker_min_match_score=TRACKER_MIN_MATCH_SCORE,
            tracker_template_update_alpha=TRACKER_TEMPLATE_UPDATE_ALPHA,
            save_tracker_plots=SAVE_TRACKER_PLOTS,
            tracker_overlay_output_scale=TRACKER_OVERLAY_OUTPUT_SCALE,
            save_overlay_videos=SAVE_OVERLAY_VIDEO,
            overlay_video_fps=OVERLAY_VIDEO_FPS,
            formal_use_temporal_body_recovery=FORMAL_USE_TEMPORAL_BODY_RECOVERY,
            formal_frame_scale=FORMAL_FRAME_SCALE,
            temporal_kwargs=dict(
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
            ),

            enable_cpu_parallel=ENABLE_CPU_PARALLEL,
            cpu_workers=CPU_WORKERS,
            sample_parallel_strategy=SAMPLE_PARALLEL_STRATEGY,
        )
        if CANONICAL_RUN_MODE == "contraction_roi":
            raise SystemExit(0)
        print("Complete mode: sampled contraction/tracker outputs finished; continuing to formal analysis and validation clips.")

    run_mode_indicator = {
        "run_mode": RUN_MODE,
        "canonical_run_mode": CANONICAL_RUN_MODE,
        "run_stages": ";".join(RUN_STAGES),
        "run_formal_analysis": bool(RUN_FORMAL_ANALYSIS),
        "run_test_clips": bool(RUN_TEST_CLIPS),
        "run_sample_contraction": bool(RUN_SAMPLE_CONTRACTION),
        "run_manual_validation_export": bool(RUN_MANUAL_VALIDATION_EXPORT),
        "run_manual_validation_annotate": bool(RUN_MANUAL_VALIDATION_ANNOTATE),
        "run_manual_validation_score": bool(RUN_MANUAL_VALIDATION_SCORE),
        "test_only_mode": bool(TEST_ONLY_MODE),
        "validation_sections": ";".join(VALIDATION_CLIP_SECTIONS),
        "validation_analyze_section_data": bool(VALIDATION_ANALYZE_SECTION_DATA),
        "validation_workers_per_section": int(VALIDATION_WORKERS_PER_SECTION),
        "edge_skip_frames": int(TEMPORAL_SKIP_EDGE_FRAMES),
        "formal_edge_exclusion": "skip_first_and_last_edge_frames",
        "segmentation_profile": SEGMENTATION_PROFILE,
        "hsv_only": not bool(FORMAL_USE_TEMPORAL_BODY_RECOVERY),
        "use_temporal_body_recovery": bool(USE_TEMPORAL_BODY_RECOVERY),
        "formal_use_temporal_body_recovery": bool(FORMAL_USE_TEMPORAL_BODY_RECOVERY),
        "temporal_scope": (
            "formal_metrics_and_validation"
            if FORMAL_USE_TEMPORAL_BODY_RECOVERY
            else "validation_only" if USE_TEMPORAL_BODY_RECOVERY else "off"
        ),
        "formal_metrics_mask": (
            "temporal_recovered_hsv_mask"
            if FORMAL_USE_TEMPORAL_BODY_RECOVERY
            else "strict_hsv_clean_mask"
        ),
        "formal_analysis_max_time_sec": FORMAL_ANALYSIS_MAX_TIME_SEC,
        "formal_analysis_max_frames": FORMAL_ANALYSIS_MAX_FRAMES,
        "formal_frame_scale": FORMAL_FRAME_SCALE,
        "formal_effective_pixel_size": _effective_pixel_size_for_formal_scale(PIXEL_SIZE, FORMAL_FRAME_SCALE),
        "formal_overlay_video_output_scale": OVERLAY_VIDEO_OUTPUT_SCALE,
        "use_yellow_chroma_extension": bool(hsv_params.get("use_yellow_chroma_extension", False)),
        "use_relaxed_body_mask": bool(hsv_params.get("use_relaxed_body_mask", False)),
        "compute_contraction": bool(COMPUTE_CONTRACTION),
        "contraction_roi_count": len(parsed_contraction_rois),
        "contraction_intensity_source": CONTRACTION_INTENSITY_SOURCE,
        "contraction_mask_layer": CONTRACTION_MASK_LAYER,
        "contraction_signal_interpretation": CONTRACTION_PROXY_NOTE,
    }
    run_mode_indicator_csv = out_dir / "run_mode_indicator.csv"
    pd.DataFrame([run_mode_indicator]).to_csv(
        run_mode_indicator_csv, index=False, encoding="utf-8-sig"
    )
    print(f"Saved run mode indicator: {run_mode_indicator_csv}")

    if RUN_MANUAL_VALIDATION_EXPORT:
        if INPUT_MODE != "video" or SEGMENTATION_PROFILE != "hsv_circle":
            raise ValueError(
                "manual_validation_export requires INPUT_MODE='video' and "
                "SEGMENTATION_PROFILE='hsv_circle'."
            )
        export_manual_validation_package(
            input_path,
            out_dir=out_dir,
            dish_center=DISH_CENTER,
            dish_radius=DISH_RADIUS,
            hsv_params=hsv_params,
            samples_per_section=MANUAL_VALIDATION_SAMPLES_PER_SECTION,
            section_frames=MANUAL_VALIDATION_SECTION_FRAMES,
        )
        print("Manual validation package export completed; full analysis was skipped.")
        raise SystemExit(0)

    if RUN_MANUAL_VALIDATION_ANNOTATE:
        annotate_manual_validation_package(
            out_dir / "manual_validation_package",
            display_scale=MANUAL_VALIDATION_DISPLAY_SCALE,
            brush_radius=MANUAL_VALIDATION_BRUSH_RADIUS,
            overlay_alpha=MANUAL_VALIDATION_OVERLAY_ALPHA,
        )
        print("Manual validation annotation step completed; full analysis was skipped.")
        raise SystemExit(0)

    if RUN_MANUAL_VALIDATION_SCORE:
        validate_manual_validation_package(out_dir / "manual_validation_package")
        print("Manual validation scoring completed; full analysis was skipped.")
        raise SystemExit(0)

    if VALIDATION_ONLY_MODE:
        if INPUT_MODE != "video" or SEGMENTATION_PROFILE != "hsv_circle":
            raise ValueError(
                "VALIDATION_ONLY_MODE requires INPUT_MODE='video' and "
                "SEGMENTATION_PROFILE='hsv_circle'."
            )
        if VALIDATION_ANALYZE_SECTION_DATA:
            if not USE_TEMPORAL_BODY_RECOVERY:
                raise ValueError(
                    "VALIDATION_ANALYZE_SECTION_DATA currently requires "
                    "USE_TEMPORAL_BODY_RECOVERY=True so section data matches formal temporal outputs."
                )
            analyze_validation_sections_formal_outputs(
                input_path,
                out_dir=out_dir,
                sections=VALIDATION_CLIP_SECTIONS,
                section_frames=VALIDATION_CLIP_SECTION_FRAMES,
                edge_skip=TEMPORAL_SKIP_EDGE_FRAMES,
                dt=FRAME_INTERVAL,
                pixel_size=PIXEL_SIZE,
                roi=ROI,
                dish_center=DISH_CENTER,
                dish_radius=DISH_RADIUS,
                hsv_params=hsv_params,
                vein_params=vein_params,
                vein_every_n_frames=VEIN_EVERY_N_FRAMES,
                compute_body_skeleton=COMPUTE_BODY_SKELETON,
                compute_front_speed=COMPUTE_FRONT_SPEED,
                compute_contraction=COMPUTE_CONTRACTION,
                contraction_rois=parsed_contraction_rois,
                contraction_intensity_source=CONTRACTION_INTENSITY_SOURCE,
                contraction_mask_layer=CONTRACTION_MASK_LAYER,
                contraction_min_body_pixels=CONTRACTION_MIN_BODY_PIXELS,
                contraction_min_period_sec=CONTRACTION_MIN_PERIOD_SEC,
                contraction_max_period_sec=CONTRACTION_MAX_PERIOD_SEC,
                contraction_min_valid_samples=CONTRACTION_MIN_VALID_SAMPLES,
                contraction_interpolate_short_gaps=CONTRACTION_INTERPOLATE_SHORT_GAPS,
                contraction_max_interpolation_gap=CONTRACTION_MAX_INTERPOLATION_GAP,
                save_contraction_plots=SAVE_CONTRACTION_PLOTS,
                draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
                formal_frame_scale=FORMAL_FRAME_SCALE,
                temporal_kwargs=dict(
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
                ),
                validation_output_scale=VALIDATION_CLIP_OUTPUT_SCALE,
                validation_stills_per_section=VALIDATION_CLIP_STILLS_PER_SECTION,
                validation_write_video=VALIDATION_CLIP_WRITE_VIDEO,
                batch_frames=VALIDATION_CLIP_BATCH_FRAMES,
                n_jobs=VALIDATION_WORKERS_PER_SECTION if ENABLE_CPU_PARALLEL else 1,
            )
        elif USE_TEMPORAL_BODY_RECOVERY:
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
                contraction_rois=parsed_contraction_rois,
                draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
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
                contraction_rois=parsed_contraction_rois,
                draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
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
        contraction_rois=parsed_contraction_rois,
        draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
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
            contraction_rois=parsed_contraction_rois,
            draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
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
        source_total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()
        formal_start_frame = int(TEMPORAL_SKIP_EDGE_FRAMES or 0)
        formal_start_frame = min(formal_start_frame, max(0, source_total_frames - 1))
        usable_formal_frames = max(0, source_total_frames - 2 * formal_start_frame)
        if MAX_FRAMES is not None:
            formal_max_frames = min(int(MAX_FRAMES), usable_formal_frames)
        else:
            formal_max_frames = usable_formal_frames
        total_frames = int(formal_max_frames)
        if formal_max_frames <= 0:
            raise ValueError(
                "No usable formal frames after edge exclusion; reduce TEMPORAL_SKIP_EDGE_FRAMES."
            )
        formal_last_frame = formal_start_frame + formal_max_frames - 1
        print(
            "Formal output frame window: "
            f"{formal_start_frame}-{formal_last_frame} "
            f"({formal_max_frames} frames; skipped {formal_start_frame} at each edge)."
        )
        if formal_max_frames <= 5:
            debug_indices = list(range(formal_start_frame, formal_start_frame + max(formal_max_frames, 0)))
        else:
            debug_indices = sorted(
                set(
                    [
                        formal_start_frame,
                        formal_start_frame + formal_max_frames // 4,
                        formal_start_frame + formal_max_frames // 2,
                        formal_start_frame + (3 * formal_max_frames) // 4,
                        formal_last_frame,
                    ]
                )
            )

        overlay_video = out_dir / "plasmodium_overlay.mp4" if SAVE_OVERLAY_VIDEO else None
        video_analysis_kwargs = dict(
            dt=FRAME_INTERVAL,
            pixel_size=PIXEL_SIZE,
            roi=ROI,
            max_frames=formal_max_frames,
            debug_frame_indices=debug_indices,
            segmentation_profile=SEGMENTATION_PROFILE,
            dish_center=DISH_CENTER,
            dish_radius=DISH_RADIUS,
            hsv_params=hsv_params,
            vein_params=vein_params,
            vein_every_n_frames=VEIN_EVERY_N_FRAMES,
            compute_body_skeleton=COMPUTE_BODY_SKELETON,
            compute_front_speed=COMPUTE_FRONT_SPEED,
            compute_contraction=COMPUTE_CONTRACTION,
            contraction_rois=parsed_contraction_rois,
            contraction_intensity_source=CONTRACTION_INTENSITY_SOURCE,
            contraction_mask_layer=CONTRACTION_MASK_LAYER,
            contraction_min_body_pixels=CONTRACTION_MIN_BODY_PIXELS,
            draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
            overlay_video_path=overlay_video,
            overlay_video_fps=OVERLAY_VIDEO_FPS,
        )
        if FORMAL_USE_TEMPORAL_BODY_RECOVERY:
            if SEGMENTATION_PROFILE != "hsv_circle":
                raise ValueError(
                    "Formal temporal recovery currently requires SEGMENTATION_PROFILE='hsv_circle'."
                )
            df, debug_data, osc = analyze_video_stream_formal_temporal(
                input_path,
                dt=FRAME_INTERVAL,
                pixel_size=PIXEL_SIZE,
                roi=ROI,
                max_frames=formal_max_frames,
                start_frame=formal_start_frame,
                debug_frame_indices=debug_indices,
                dish_center=DISH_CENTER,
                dish_radius=DISH_RADIUS,
                hsv_params=hsv_params,
                vein_params=vein_params,
                vein_every_n_frames=VEIN_EVERY_N_FRAMES,
                compute_body_skeleton=COMPUTE_BODY_SKELETON,
                compute_front_speed=COMPUTE_FRONT_SPEED,
                analysis_roi=ANALYSIS_ROI,
                compute_contraction=COMPUTE_CONTRACTION,
                contraction_rois=parsed_contraction_rois,
                contraction_intensity_source=CONTRACTION_INTENSITY_SOURCE,
                contraction_mask_layer=CONTRACTION_MASK_LAYER,
                contraction_min_body_pixels=CONTRACTION_MIN_BODY_PIXELS,
                draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
                overlay_video_path=overlay_video,
                overlay_video_fps=OVERLAY_VIDEO_FPS,
                overlay_video_output_scale=OVERLAY_VIDEO_OUTPUT_SCALE,
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
                batch_frames=PARALLEL_BATCH_FRAMES,
                formal_frame_scale=FORMAL_FRAME_SCALE,
            )
        elif ENABLE_CPU_PARALLEL and LOW_MEMORY_VIDEO_STREAM:
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
        loaded_edge_skip = int(TEMPORAL_SKIP_EDGE_FRAMES or 0)
        if loaded_edge_skip > 0 and len(frames) > 2 * loaded_edge_skip:
            frames = frames[loaded_edge_skip:-loaded_edge_skip]
            names = names[loaded_edge_skip:-loaded_edge_skip]
            print(
                "Formal loaded-frame edge exclusion: "
                f"skipped first/last {loaded_edge_skip} frames; "
                f"analyzing {len(frames)} frames."
            )
        elif loaded_edge_skip > 0:
            warnings.warn(
                "Not enough loaded frames to apply edge exclusion; analyzing all loaded frames."
            )

        if FORMAL_USE_TEMPORAL_BODY_RECOVERY:
            if SEGMENTATION_PROFILE != "hsv_circle":
                raise ValueError(
                    "Formal temporal recovery currently requires SEGMENTATION_PROFILE='hsv_circle'."
                )
            df, debug_data, osc = analyze_timelapse_formal_temporal(
                frames,
                names=names,
                dt=FRAME_INTERVAL,
                pixel_size=PIXEL_SIZE,
                roi=ROI,
                analysis_roi=ANALYSIS_ROI,
                dish_center=DISH_CENTER,
                dish_radius=DISH_RADIUS,
                hsv_params=hsv_params,
                vein_params=vein_params,
                vein_every_n_frames=VEIN_EVERY_N_FRAMES,
                compute_body_skeleton=COMPUTE_BODY_SKELETON,
                compute_front_speed=COMPUTE_FRONT_SPEED,
                compute_contraction=COMPUTE_CONTRACTION,
                contraction_rois=parsed_contraction_rois,
                contraction_intensity_source=CONTRACTION_INTENSITY_SOURCE,
                contraction_mask_layer=CONTRACTION_MASK_LAYER,
                contraction_min_body_pixels=CONTRACTION_MIN_BODY_PIXELS,
                draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
                keep_debug=True,
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
                formal_frame_scale=FORMAL_FRAME_SCALE,
            )
        else:
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
                compute_contraction=COMPUTE_CONTRACTION,
                contraction_rois=parsed_contraction_rois,
                contraction_intensity_source=CONTRACTION_INTENSITY_SOURCE,
                contraction_mask_layer=CONTRACTION_MASK_LAYER,
                contraction_min_body_pixels=CONTRACTION_MIN_BODY_PIXELS,
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
                output_scale=(1.0 if FORMAL_USE_TEMPORAL_BODY_RECOVERY and FORMAL_FRAME_SCALE < 0.999 else OVERLAY_VIDEO_OUTPUT_SCALE),
                contraction_rois=parsed_contraction_rois,
                draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
            )
            print(f"Saved overlay video: {overlay_video}")

    if "run_mode" not in df.columns:
        df.insert(0, "run_mode", RUN_MODE)
    if "run_formal_analysis" not in df.columns:
        df.insert(1, "run_formal_analysis", bool(RUN_FORMAL_ANALYSIS))
    if "validation_sections" not in df.columns:
        df.insert(2, "validation_sections", ";".join(VALIDATION_CLIP_SECTIONS))
    if "formal_use_temporal_body_recovery" not in df.columns:
        df.insert(3, "formal_use_temporal_body_recovery", bool(FORMAL_USE_TEMPORAL_BODY_RECOVERY))
    if "formal_frame_scale" not in df.columns:
        df.insert(4, "formal_frame_scale", FORMAL_FRAME_SCALE)
    if "formal_effective_pixel_size" not in df.columns:
        df.insert(5, "formal_effective_pixel_size", _effective_pixel_size_for_formal_scale(PIXEL_SIZE, FORMAL_FRAME_SCALE))
    if "formal_metrics_mask" not in df.columns:
        df.insert(
            4,
            "formal_metrics_mask",
            "temporal_recovered_hsv_mask"
            if FORMAL_USE_TEMPORAL_BODY_RECOVERY
            else "strict_hsv_clean_mask",
        )

    contraction_summary = pd.DataFrame()
    if COMPUTE_CONTRACTION and parsed_contraction_rois:
        contraction_summary = analyze_contraction_rois(
            df=df,
            contraction_rois=parsed_contraction_rois,
            dt=FRAME_INTERVAL,
            min_period_sec=CONTRACTION_MIN_PERIOD_SEC,
            max_period_sec=CONTRACTION_MAX_PERIOD_SEC,
            min_valid_samples=CONTRACTION_MIN_VALID_SAMPLES,
            interpolate_short_gaps=CONTRACTION_INTERPOLATE_SHORT_GAPS,
            max_interpolation_gap=CONTRACTION_MAX_INTERPOLATION_GAP,
            out_dir=out_dir,
            save_plots=SAVE_CONTRACTION_PLOTS,
        )
        plot_contraction_summary(
            contraction_summary,
            out_dir / "plasmodium_summary_contraction.png",
        )
        print("Contraction analysis summary:")
        print(
            contraction_summary[
                [
                    "roi_name",
                    "period_sec",
                    "period_min",
                    "amplitude_peak_to_peak",
                    "valid",
                    "invalid_reason",
                ]
            ].rename(
                columns={
                    "roi_name": "ROI",
                    "amplitude_peak_to_peak": "amplitude",
                    "invalid_reason": "reason",
                }
            ).to_string(index=False)
        )
        print(CONTRACTION_PROXY_NOTE)
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
                    contraction_rois=parsed_contraction_rois,
                    draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
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
                    contraction_rois=parsed_contraction_rois,
                    draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
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
        save_debug_examples(
            debug_data,
            debug_out,
            debug_indices_to_save,
            contraction_rois=parsed_contraction_rois,
            draw_contraction_rois=DRAW_CONTRACTION_ROIS_ON_OVERLAY,
        )
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

    summary_df["contraction_roi_count"] = int(len(contraction_summary))
    if not contraction_summary.empty:
        valid_contraction = contraction_summary.loc[
            contraction_summary["valid"].astype(bool)
        ]
    else:
        valid_contraction = contraction_summary
    summary_df["contraction_valid_roi_count"] = int(len(valid_contraction))
    if not valid_contraction.empty:
        valid_periods = pd.to_numeric(
            valid_contraction["period_sec"], errors="coerce"
        ).to_numpy(dtype=float)
        valid_amplitudes = pd.to_numeric(
            valid_contraction["amplitude_peak_to_peak"], errors="coerce"
        ).to_numpy(dtype=float)
        summary_df["contraction_period_mean_sec"] = (
            float(np.nanmean(valid_periods))
            if np.isfinite(valid_periods).any()
            else np.nan
        )
        summary_df["contraction_period_median_sec"] = (
            float(np.nanmedian(valid_periods))
            if np.isfinite(valid_periods).any()
            else np.nan
        )
        summary_df["contraction_period_mean_min"] = (
            float(np.nanmean(valid_periods) / 60.0)
            if np.isfinite(valid_periods).any()
            else np.nan
        )
        summary_df["contraction_amplitude_mean"] = (
            float(np.nanmean(valid_amplitudes))
            if np.isfinite(valid_amplitudes).any()
            else np.nan
        )
    else:
        summary_df["contraction_period_mean_sec"] = np.nan
        summary_df["contraction_period_median_sec"] = np.nan
        summary_df["contraction_period_mean_min"] = np.nan
        summary_df["contraction_amplitude_mean"] = np.nan

    summary_csv = out_dir / "plasmodium_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary CSV: {summary_csv}")
    print(summary_df.to_string(index=False))

    print("\nComplete output checklist:")
    print(f"  Formal metrics CSV: {metrics_csv}")
    print(f"  Formal summary CSV: {summary_csv}")
    print(f"  Formal overlay video: {out_dir / 'plasmodium_overlay.mp4'}")
    print(f"  Debug examples: {debug_out}")
    print(f"  Validation clips/stills: {out_dir / 'validation_clips'}")
    print(f"  Sampled early/middle/late metrics: {out_dir / 'sampled_frames_all_metrics.csv'}")
    print(f"  Fixed ROI contraction summary: {out_dir / 'contraction_summary_all_segments.csv'}")
    print(f"  Tracker contraction summary: {out_dir / 'tracker_contraction_summary_all_segments.csv'}")
    print(f"  Method comparison CSV: {out_dir / 'contraction_method_comparison.csv'}")
    print(f"  Method comparison plot: {out_dir / 'plasmodium_summary_contraction_methods.png'}")
    print("  Per-segment folders: early/, middle/, late/")



