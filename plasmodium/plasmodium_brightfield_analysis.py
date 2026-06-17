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
    area_fraction = vein_area / area
    length_norm = vein_length / np.sqrt(area)
    rel_change = vein_length.pct_change().abs()
    stable_change = rel_change.rolling(
        int(stable_window), min_periods=1
    ).median()

    base_ok = (
        (out["frame"].astype(int) >= int(warmup_frames))
        & (vein_area >= float(min_vein_area_px))
        & (vein_length >= float(min_vein_length_px))
        & (length_norm >= float(min_length_norm))
        & (area_fraction <= float(max_area_fraction))
        & ((stable_change <= float(max_rel_change)) | stable_change.isna())
    )

    run_lengths = []
    run = 0
    for ok in base_ok.fillna(False).to_numpy(dtype=bool):
        run = run + 1 if ok else 0
        run_lengths.append(run)
    trusted = base_ok & (pd.Series(run_lengths, index=out.index) >= int(require_consecutive))

    out["vein_area_fraction"] = area_fraction
    out["vein_length_norm"] = length_norm
    out["vein_rel_change"] = rel_change
    out["vein_stable_rel_change"] = stable_change
    out["vein_trusted"] = trusted.astype(bool)
    out["vein_low_confidence"] = ~out["vein_trusted"]

    for col in ["vein_area_px", "vein_length_px", "vein_length", "vein_endpoints", "vein_branchpoints"]:
        trusted_col = f"{col}_trusted"
        out[trusted_col] = out[col].where(out["vein_trusted"], np.nan)

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


def circular_roi_mask(shape, center, radius):
    """Create a circular ROI mask for dish-inner analysis."""
    if center is None or radius is None:
        return np.ones(shape[:2], dtype=bool)

    cx, cy = [int(v) for v in center]
    radius = int(radius)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


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
):
    """
    Segment yellow plasmodium using one consistent HSV rule inside a dish circle.

    This is intended for color time-lapse videos where plasmodium is yellow
    and the dish/background is mostly gray or transparent.
    """
    if frame.ndim == 2:
        bgr = cv2.cvtColor(_as_gray_uint8(frame), cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mask = (
        (hue >= int(h_min))
        & (hue <= int(h_max))
        & (sat >= int(s_min))
        & (val >= int(v_min))
    )
    mask &= circular_roi_mask(frame.shape[:2], dish_center, dish_radius)

    if morph_open_size and morph_open_size > 0:
        mask = morphology.binary_opening(mask, morphology.disk(int(morph_open_size)))

    if morph_close_size and morph_close_size > 0:
        mask = morphology.binary_closing(mask, morphology.disk(int(morph_close_size)))

    if min_obj_size and min_obj_size > 0:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_obj_size))

    if min_hole_size and min_hole_size > 0:
        mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=int(min_hole_size))

    debug = {
        "gray": _as_gray_uint8(frame),
        "background": np.zeros(frame.shape[:2], dtype=np.uint8),
        "corrected": sat,
        "enhanced": val,
        "analysis_roi": None,
        "dish_center": dish_center,
        "dish_radius": dish_radius,
    }
    return val, mask.astype(bool), debug


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

        reg = region_metrics(mask, pixel_size=pixel_size)
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
            body_mask=mask,
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

            reg = region_metrics(mask, pixel_size=pixel_size)
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
                body_mask=mask,
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

        vein_length_col = "vein_length_trusted" if "vein_length_trusted" in df.columns else "vein_length"
        vein_endpoints_col = (
            "vein_endpoints_trusted" if "vein_endpoints_trusted" in df.columns else "vein_endpoints"
        )
        vein_branchpoints_col = (
            "vein_branchpoints_trusted"
            if "vein_branchpoints_trusted" in df.columns
            else "vein_branchpoints"
        )
        required_vein_cols = {vein_length_col, vein_endpoints_col, vein_branchpoints_col}
        if required_vein_cols.issubset(df.columns):
            vein_fig, vein_axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
            if "vein_length" in df.columns and vein_length_col != "vein_length":
                vein_axes[0].plot(t, df["vein_length"], color="0.75", label="raw visible vein length")
            vein_axes[0].plot(t, df[vein_length_col], color="tab:blue", label="trusted visible vein length")
            vein_axes[0].set_title("Trusted visible vein network")
            vein_axes[0].set_xlabel("Time")
            vein_axes[0].set_ylabel("Length")
            vein_axes[0].legend()
            vein_axes[1].plot(t, df[vein_endpoints_col], color="gold", label="trusted vein endpoints")
            vein_axes[1].plot(t, df[vein_branchpoints_col], color="teal", label="trusted vein branchpoints")
            vein_axes[1].set_xlabel("Time")
            vein_axes[1].set_ylabel("Count")
            vein_axes[1].legend()
            vein_fig.savefig(out_path.with_name(out_path.stem + "_veins.png"), dpi=200)

    return fig


def save_debug_examples(debug_data, out_dir, frame_indices):
    """
    Save gray, prep, mask, and overlay images for selected frames.

    Overlay contains mask, skeleton, endpoints, branchpoints, new region,
    and retracted region.
    """
    if debug_data is None or len(debug_data) == 0:
        warnings.warn("No debug_data available to save.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in frame_indices:
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
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay_bgr)


def save_overlay_video(debug_data, out_path, fps=10, draw_frame_text=True):
    """
    Save an overlay video from debug_data.

    The fps here is only for playback speed of the exported overlay video.
    It does not change biological time calculations, which use FRAME_INTERVAL.
    """
    if debug_data is None or len(debug_data) == 0:
        warnings.warn("No debug_data available to save overlay video.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_gray = _as_gray_uint8(debug_data[0]["gray"])
    height, width = first_gray.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height), True)

    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {out_path}")

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

            if draw_frame_text:
                text = f"frame {int(item['frame'])}"
                cv2.putText(
                    frame_bgr,
                    text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_bgr,
                    text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 0),
                    1,
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

    qc["mask_area_fraction"] = np.nan
    qc["flag_area_fraction"] = False
    qc["flag_touch_roi_border"] = False

    if debug_data:
        for item in debug_data:
            idx = int(item["frame"])
            mask = item["mask"].astype(bool)
            if analysis_roi is None:
                roi_mask = mask
            else:
                ys, xs = _roi_slices(mask.shape, analysis_roi)
                roi_mask = mask[ys, xs]

            roi_area = roi_mask.size
            fraction = float(np.sum(roi_mask) / roi_area) if roi_area > 0 else np.nan
            qc.loc[qc["frame"] == idx, "mask_area_fraction"] = fraction
            lo, hi = mask_area_fraction_limits
            qc.loc[qc["frame"] == idx, "flag_area_fraction"] = bool(
                np.isfinite(fraction) and (fraction < lo or fraction > hi)
            )

            if roi_mask.size > 0 and np.any(roi_mask):
                touches = (
                    np.any(roi_mask[0, :])
                    or np.any(roi_mask[-1, :])
                    or np.any(roi_mask[:, 0])
                    or np.any(roi_mask[:, -1])
                )
                qc.loc[qc["frame"] == idx, "flag_touch_roi_border"] = bool(touches)

    flag_cols = [
        "flag_empty_or_tiny",
        "flag_area_jump",
        "flag_low_focus",
        "flag_area_fraction",
        "flag_touch_roi_border",
    ]
    qc["needs_review"] = qc[flag_cols].any(axis=1)

    qc_csv = out_dir / "plasmodium_qc_flags.csv"
    qc.to_csv(qc_csv, index=False, encoding="utf-8-sig")

    if save_review_images and debug_data:
        review_dir = out_dir / "qc_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        flagged = qc.loc[qc["needs_review"], "frame"].astype(int).tolist()
        sampled = sorted(set(np.linspace(0, len(debug_data) - 1, min(12, len(debug_data)), dtype=int).tolist()))
        review_indices = sorted(set(flagged[:40] + sampled))
        save_debug_examples(debug_data, review_dir, review_indices)

    print(f"Saved QC flags CSV: {qc_csv}")
    print(f"QC frames needing review: {int(qc['needs_review'].sum())} / {len(qc)}")
    return qc


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


def summarize_run(df, label):
    """
    Generate a one-row summary table for one experimental run.
    """
    if df is None or df.empty:
        raise ValueError("df is empty; cannot summarize run.")

    def mean_col(col):
        return float(np.nanmean(df[col])) if col in df.columns else np.nan

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
                "circularity_mean": mean_col("circularity"),
                "solidity_mean": mean_col("solidity"),
                "branchpoints_mean": mean_col("n_branchpoints"),
                "endpoints_mean": mean_col("n_endpoints"),
                "skeleton_length_mean": mean_col("skeleton_length"),
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
    OVERLAY_VALIDATION_MODE = True

    # Use the real acquisition interval, not the exported video playback FPS.
    # Example: images captured every 10 s and exported as 60 FPS video still use 10.0 here.
    FRAME_INTERVAL = 10.0
    PIXEL_SIZE = 1.0
    ROI = None  # Example: ROI = (x, y, w, h)
    SEGMENTATION_PROFILE = "hsv_circle"  # "hsv_circle", "classic", or "grabcut_blob"
    ANALYSIS_ROI = (700, 520, 850, 700)  # Used only by "classic" / "grabcut_blob".
    DISH_CENTER = (1280, 960)  # x, y. Adjust once per camera setup if needed.
    DISH_RADIUS = 760
    MANUAL_MASK_DIR = None  # Optional folder with hand-labeled masks for IoU/Dice validation.
    SEGMENT_TEST_FRAMES_PER_PART = 3

    if OVERLAY_VALIDATION_MODE:
        MAX_FRAMES = 900
        SAVE_OVERLAY_VIDEO = True
        OVERLAY_VIDEO_FPS = 30
        VEIN_EVERY_N_FRAMES = 10
        COMPUTE_BODY_SKELETON = True
        COMPUTE_FRONT_SPEED = False
        RUN_QC_VALIDATION = False
        RUN_SEGMENT_TESTS = False
    else:
        MAX_FRAMES = None
        SAVE_OVERLAY_VIDEO = False
        OVERLAY_VIDEO_FPS = 30
        VEIN_EVERY_N_FRAMES = 5
        COMPUTE_BODY_SKELETON = True
        COMPUTE_FRONT_SPEED = True
        RUN_QC_VALIDATION = True
        RUN_SEGMENT_TESTS = False

    print("Run configuration:")
    print(f"  OVERLAY_VALIDATION_MODE = {OVERLAY_VALIDATION_MODE}")
    print(f"  MAX_FRAMES = {MAX_FRAMES}")
    print(f"  SAVE_OVERLAY_VIDEO = {SAVE_OVERLAY_VIDEO}")
    print(f"  VEIN_EVERY_N_FRAMES = {VEIN_EVERY_N_FRAMES}")
    print(f"  COMPUTE_BODY_SKELETON = {COMPUTE_BODY_SKELETON}")
    print(f"  COMPUTE_FRONT_SPEED = {COMPUTE_FRONT_SPEED}")
    print(f"  RUN_QC_VALIDATION = {RUN_QC_VALIDATION}")

    hsv_params = dict(
        h_min=16,
        h_max=48,
        s_min=60,
        v_min=45,
        min_obj_size=150,
        min_hole_size=0,
        morph_open_size=1,
        morph_close_size=0,
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

    if INPUT_MODE == "video" and SEGMENTATION_PROFILE == "hsv_circle":
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
        df, debug_data, osc = analyze_video_stream(
            input_path,
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
        print(f"Analyzed {len(df)} frames.")
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
        )

    metrics_csv = out_dir / "plasmodium_metrics.csv"
    df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"Saved metrics CSV: {metrics_csv}")

    summary_plot = out_dir / "plasmodium_summary.png"
    plot_summary(df, out_prefix=summary_plot)
    plt.show()
    print(f"Saved summary plot: {summary_plot}")

    debug_out = out_dir / "debug_examples"
    if len(debug_data) > 0:
        save_debug_examples(debug_data, debug_out, list(range(len(debug_data))))
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
    summary_df["osc_dominant_freq"] = osc.get("dominant_freq", np.nan)
    summary_df["osc_period"] = osc.get("period", np.nan)
    summary_df["osc_n_peaks"] = osc.get("n_peaks", 0)

    summary_csv = out_dir / "plasmodium_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary CSV: {summary_csv}")
    print(summary_df.to_string(index=False))
