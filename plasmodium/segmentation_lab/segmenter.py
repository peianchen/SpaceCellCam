"""Review-first colour/structure segmentation; no trained model or temporal fill.

All masks stay in input coordinates. Local background normalization and
hysteresis retain thin connected evidence without a largest-component rule.
Food exclusion is a conservative image heuristic, exported for human review.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .contracts import RoiSpec, SegmentationResult


DEFAULTS = {
    "background_scale": 4,
    "background_radius_px": 48,
    "low_evidence": 0.045,
    "high_evidence": 0.105,
    "noise_multiplier": 3.5,
    "min_component_px": 12,
    "min_seed_px": 3,
    "food_exclusion": True,
    "food_seed_rg": 0.11,
    "food_grow_rg": 0.095,
    "food_max_center_fraction": 0.45,
    "food_refine": True,
    "refine_roi": False,
    "saturation_clip": 250,
}


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)


def _background(image: np.ndarray, radius: float, scale: int = 4) -> np.ndarray:
    """Approximate a smooth lower envelope at coarse scale, not a fake flat image."""
    small = cv2.resize(image, (max(8, image.shape[1] // scale), max(8, image.shape[0] // scale)), interpolation=cv2.INTER_AREA)
    opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, _disk(max(2, round(radius / scale))))
    opened = cv2.GaussianBlur(opened, (0, 0), max(1., radius / scale / 3))
    return cv2.resize(opened, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def _components(candidate: np.ndarray, seed: np.ndarray, min_area: int, min_seed: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    seed_counts = np.bincount(labels[seed].ravel(), minlength=count)
    keep = (stats[:, cv2.CC_STAT_AREA] >= min_area) & (seed_counts >= min_seed)
    keep[0] = False
    return keep[labels]


def _food_mask(image: np.ndarray, roi: RoiSpec, inside: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Detect only a compact, centrally situated brown core, never its filled halo."""
    empty = np.zeros(inside.shape, bool)
    if not cfg["food_exclusion"]:
        return empty, ["food_exclusion_disabled"]
    blue, green, red = cv2.split(image.astype(np.float32) / 255.)
    rg = (red - green) / (red + green + .02)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    central = RoiSpec(roi.center, roi.radius * float(cfg["food_max_center_fraction"]), 0).mask(inside.shape)
    pool = (rg > cfg["food_grow_rg"]) & (hsv[..., 0] < 25) & (hsv[..., 1] > 55) & (green < .75) & central & inside
    # Dark orange tissue and brown food can overlap in HSV. Start from the
    # redder core of this frame, then let the constrained colour model grow it.
    seed_rg = max(float(cfg["food_seed_rg"]), float(np.percentile(rg[pool], 75))) if pool.any() else float(cfg["food_seed_rg"])
    grow_rg = max(float(cfg["food_grow_rg"]), seed_rg-.07)
    brown = pool & (rg > grow_rg)
    strong = brown & (rg >= seed_rg) & (hsv[..., 0] < 22)
    # Remove tiny colour speckle, then grow only within brown evidence.
    strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_OPEN, _disk(2)).astype(bool)
    food = _components(brown, strong, max(30, int(roi.radius ** 2 * .0005)), 20)
    labels, count = ndi.label(food)
    selected = empty.copy()
    max_area = np.pi * roi.radius ** 2 * .12
    for index, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        part = labels[sl] == index
        area = int(part.sum())
        if area > max_area:
            continue
        yy, xx = np.nonzero(part)
        center = (xx.mean() + sl[1].start, yy.mean() + sl[0].start)
        if np.hypot(center[0] - roi.center[0], center[1] - roi.center[1]) < roi.radius * .32:
            selected[sl] |= part
    if selected.any():
        refinement_failed = False
        # Fill only holes in the detected compact food object. Never fill tissue holes.
        selected = ndi.binary_fill_holes(selected)
        selected = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_CLOSE, _disk(2)).astype(bool)
        if cfg["food_refine"]:
            # Learn this frame's brown core vs its immediate surround. The graph
            # cut is constrained to the neighbourhood of the chromatic food seed.
            fy, fx = np.nonzero(selected)
            pad = max(12, int(roi.radius * .08))
            sl = np.s_[max(0, fy.min()-pad):min(image.shape[0], fy.max()+pad+1),
                       max(0, fx.min()-pad):min(image.shape[1], fx.max()+pad+1)]
            piece = np.ascontiguousarray(image[sl])
            original = selected[sl]
            support = cv2.dilate(original.astype(np.uint8), _disk(max(6, int(roi.radius*.035)))).astype(bool)
            cut = np.full(original.shape, cv2.GC_PR_BGD, np.uint8)
            cut[original] = cv2.GC_PR_FGD
            cut[~support] = cv2.GC_BGD
            sure = cv2.erode(original.astype(np.uint8), _disk(3)).astype(bool) & strong[sl]
            cut[sure] = cv2.GC_FGD
            if sure.sum() >= 10 and (~support).sum() >= 10:
                try:
                    cv2.setRNGSeed(0)
                    cv2.grabCut(piece, cut, None, np.zeros((1,65),np.float64), np.zeros((1,65),np.float64), 3, cv2.GC_INIT_WITH_MASK)
                    refined = ((cut == cv2.GC_FGD) | (cut == cv2.GC_PR_FGD)) & support
                    selected[sl] = ndi.binary_fill_holes(refined)
                except cv2.error:
                    refinement_failed = True
        # Visible yellow tissue on top of food remains eligible tissue, including
        # enclosed patches. Geometry-based food hole filling must not erase it.
        yellow_surface = (rg < min(.09, grow_rg)) & (hsv[..., 0] >= 22) & (hsv[..., 1] > 55)
        selected &= ~yellow_surface
        return selected & inside, ["food_boundary_requires_review"] + (["food_refinement_failed"] if refinement_failed else [])
    return empty, ["food_not_detected"]


def _refine_roi(image: np.ndarray, roi: RoiSpec) -> tuple[RoiSpec, dict]:
    """Optional constrained rim estimate. Leave unchanged if insufficient evidence."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    theta = np.linspace(0, 2*np.pi, 180, endpoint=False)
    radii = np.linspace(roi.radius*.70, roi.radius*1.12, 240)
    mx = (roi.center[0] + np.cos(theta[:, None])*radii).astype(np.float32)
    my = (roi.center[1] + np.sin(theta[:, None])*radii).astype(np.float32)
    values = cv2.remap(gray, mx, my, cv2.INTER_LINEAR)
    sat = cv2.remap(hue[..., 1], mx, my, cv2.INTER_LINEAR)
    gradient = np.gradient(cv2.GaussianBlur(values, (5, 1), 0), axis=1)
    gradient[sat > 85] = 0
    # Several concentric plastic rims exist. Pick the first clear neutral bright
    # edge after the inner chamber, not the strongest (often an outer ring).
    threshold = np.maximum(2., gradient.max(axis=1)*.32)
    candidates = (gradient > threshold[:, None]) & (values > np.median(values[:, :25], axis=1)[:, None]+20)
    peaks = candidates.argmax(axis=1)
    amplitude = gradient[np.arange(len(theta)), peaks]
    valid = candidates.any(axis=1) & (amplitude > 2.0)
    if valid.sum() < 90:
        return roi, {"roi_refined": False, "rim_angles": int(valid.sum())}
    x, y = mx[np.arange(len(theta)), peaks][valid], my[np.arange(len(theta)), peaks][valid]
    for _ in range(4):
        (cx, cy), (width, height), angle = cv2.fitEllipse(np.column_stack((x, y)).astype(np.float32).reshape(-1, 1, 2))
        radians = np.deg2rad(angle)
        dx, dy = x-cx, y-cy
        u, v = np.cos(radians)*dx+np.sin(radians)*dy, -np.sin(radians)*dx+np.cos(radians)*dy
        residual = np.abs(np.sqrt((u/(width/2))**2+(v/(height/2))**2)-1)*(width+height)/4
        keep = residual < max(5, min(18, float(np.median(residual))*2.5))
        if keep.sum() < 60:
            return roi, {"roi_refined": False, "rim_angles": int(keep.sum())}
        x, y = x[keep], y[keep]
    radius = float(np.sqrt(width*height)/2)
    accepted = .65*roi.radius < min(width,height)/2 and max(width,height)/2 < 1.16*roi.radius and np.hypot(cx-roi.center[0], cy-roi.center[1]) < .2*roi.radius
    if not accepted:
        return roi, {"roi_refined": False, "rim_angles": len(x)}
    return RoiSpec((float(cx), float(cy)), radius, roi.margin_px), {"roi_refined": True, "roi_shape": "ellipse", "roi_ellipse_axes": [float(width/2),float(height/2)], "roi_ellipse_angle": float(angle), "rim_angles": len(x), "rim_fit_error_px": float(np.median(residual))}


def segment(image_bgr: np.ndarray, roi: RoiSpec, method: str = "adaptive_chroma", config: dict | None = None) -> SegmentationResult:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    if not np.isfinite([*roi.center, roi.radius, roi.margin_px]).all() or roi.radius <= roi.margin_px or roi.margin_px < 0:
        raise ValueError("Invalid circular ROI")
    if method not in {"adaptive_chroma", "local_lab"}:
        raise ValueError(f"Unknown new method: {method}")
    cfg = dict(DEFAULTS)
    if config:
        unknown = set(config)-set(cfg)
        if unknown:
            raise ValueError(f"Unknown segmentation parameters: {sorted(unknown)}")
        cfg.update(config)
    for key in ("food_exclusion", "food_refine", "refine_roi"):
        if not isinstance(cfg[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean")
    for key, value in cfg.items():
        if key in ("food_exclusion", "food_refine", "refine_roi"):
            continue
        if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    for key in ("background_scale", "min_component_px", "min_seed_px", "saturation_clip"):
        if int(cfg[key]) != cfg[key]:
            raise ValueError(f"{key} must be an integer")
    if not 1 <= cfg["saturation_clip"] <= 255 or cfg["noise_multiplier"] <= 0 or not 0 < cfg["food_max_center_fraction"] <= 1:
        raise ValueError("Invalid clipping, noise or food search settings")
    if not 0 <= cfg["food_grow_rg"] <= cfg["food_seed_rg"] < 1:
        raise ValueError("Food chroma thresholds must satisfy 0 <= grow <= seed < 1")
    if cfg["low_evidence"] <= 0 or cfg["high_evidence"] <= cfg["low_evidence"]:
        raise ValueError("Evidence thresholds must satisfy 0 < low < high")
    if cfg["background_scale"] < 1 or cfg["background_radius_px"] <= 0 or cfg["min_component_px"] < 1 or cfg["min_seed_px"] < 1:
        raise ValueError("Invalid spatial scale/component settings")
    inside = roi.mask(image_bgr.shape[:2])
    if inside.sum() < 64:
        raise ValueError("Analysis ROI is outside image or too small")
    roi_details = {"roi_refined": False}
    if cfg["refine_roi"]:
        roi, roi_details = _refine_roi(image_bgr, roi)
        inside = roi.mask(image_bgr.shape[:2])
        if roi_details.get("roi_shape") == "ellipse":
            if min(roi_details["roi_ellipse_axes"]) <= roi.margin_px:
                raise ValueError("Refined ROI is smaller than its exclusion margin")
            inside = np.zeros(image_bgr.shape[:2], np.uint8)
            axes = tuple(max(1, int(round(x-roi.margin_px))) for x in roi_details["roi_ellipse_axes"])
            cv2.ellipse(inside, tuple(int(round(x)) for x in roi.center), axes, roi_details["roi_ellipse_angle"], 0, 360, 1, -1)
            inside = inside.astype(bool)
    if inside.sum() < 64:
        raise ValueError("Refined analysis ROI is too small")
    # Crop work arrays to ROI + a context halo; all exported arrays are restored.
    yy, xx = np.nonzero(inside)
    pad = int(cfg["background_radius_px"] * 2)
    y0, y1 = max(0, yy.min()-pad), min(image_bgr.shape[0], yy.max()+pad+1)
    x0, x1 = max(0, xx.min()-pad), min(image_bgr.shape[1], xx.max()+pad+1)
    crop = image_bgr[y0:y1, x0:x1]
    region = inside[y0:y1, x0:x1]
    local_roi = RoiSpec((roi.center[0]-x0, roi.center[1]-y0), roi.radius, roi.margin_px)
    food, flags = _food_mask(crop, local_roi, region, cfg)
    if cfg["refine_roi"] and not roi_details["roi_refined"]:
        flags.append("roi_refinement_failed_review_geometry")
    if inside[0].any() or inside[-1].any() or inside[:, 0].any() or inside[:, -1].any():
        flags.append("roi_clipped_by_image")
    b, g, r = cv2.split(crop.astype(np.float32)/255.)
    yellow = (np.minimum(r, g)-b)/(np.minimum(r, g)+b+.08)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = (lab[..., 2]-128.)/128.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Achromatic glare must not become a thin white 'vein'.
    usable = region & ~food & (np.maximum.reduce([b, g, r]) > .045)
    chromatic = (hsv[..., 0] >= 10) & (hsv[..., 0] <= 58) & (hsv[..., 1] >= 12)
    clipping = np.min(crop, axis=2) >= int(cfg["saturation_clip"])
    usable &= ~clipping
    bg_pool = yellow[usable]
    if bg_pool.size < 32:
        bg_pool = yellow[region]
    cutoff = float(np.percentile(bg_pool, 60))
    low_background = bg_pool[bg_pool <= cutoff]
    bg_level = float(np.median(low_background))
    bg_sigma = max(.003, float(np.median(np.abs(low_background-bg_level))*1.4826))
    # Avoid rim and food halos in the background estimate by replacing exclusions
    # with the observed neutral background, not constant black.
    y_work = yellow.copy()
    y_work[~region | food] = bg_level
    y_bg = _background(y_work, float(cfg["background_radius_px"]), int(cfg["background_scale"]))
    lab_work = lab_b.copy()
    lab_work[~region | food] = float(np.median(lab_b[region & ~food])) if np.any(region & ~food) else 0.
    lab_bg = _background(lab_work, float(cfg["background_radius_px"]), int(cfg["background_scale"]))
    local_yellow = yellow-y_bg
    local_lab = lab_b-lab_bg
    # Broad bodies need an absolute/background-relative path as local flattening
    # alone would erase their uniform interiors. Fine branches use local contrast.
    broad = np.maximum(yellow-bg_level, 0.)
    smooth = cv2.GaussianBlur(y_work, (0, 0), .65)
    ridge = np.maximum.reduce([smooth-cv2.GaussianBlur(y_work, (0, 0), s) for s in (2., 4., 8.)])
    evidence = np.maximum(.7*broad, np.maximum(local_yellow, 1.1*local_lab))
    evidence = np.maximum(evidence, ridge*1.6)
    low = max(float(cfg["low_evidence"]), min(.09, bg_sigma*float(cfg["noise_multiplier"])))
    high = max(float(cfg["high_evidence"]), low*1.8)
    if method == "local_lab":
        # Intentionally simpler comparator: local Lab contrast with no hysteresis.
        evidence = local_lab
        candidate = (evidence > .045) & usable & chromatic
        seed = candidate
        high, low = .045, .045
    else:
        # Strict seed + permissive evidence only; never dilate pixels into gaps.
        candidate = (evidence > low) & usable & chromatic
        seed = (evidence > high) & usable & chromatic
    mask = _components(candidate, seed, int(cfg["min_component_px"]), int(cfg["min_seed_px"]))
    score = np.clip(evidence/max(high, .001), 0, 1).astype(np.float32)
    score[~usable] = 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    light_bg = cv2.GaussianBlur(gray, (0, 0), max(4., roi.radius*.04))
    corrected = np.clip(crop.astype(np.float32)*(110./np.maximum(light_bg, 15.))[..., None], 0, 255).astype(np.uint8)
    fraction = float(mask.sum()/max(1, region.sum()))
    evidence_fraction = float(seed.sum()/max(1, region.sum()))
    border = region & ~cv2.erode(region.astype(np.uint8), _disk(5)).astype(bool)
    rim_fraction = float((mask & border).sum()/max(1, mask.sum()))
    uncertain = candidate & ~seed
    if not mask.any():
        flags.append("empty_no_supported_tissue")
    if fraction > .65:
        flags.append("large_coverage_check_background")
    if (mask & border).any():
        flags.append("tissue_near_roi_boundary")
    if clipping[region].mean() > .02:
        flags.append("glare_in_roi")
    if bg_sigma > .035:
        flags.append("heterogeneous_background")
    if mask.any() and (uncertain & mask).sum()/mask.sum() > .45:
        flags.append("low_contrast_branches_review")
    # Quality score is an explicit heuristic, never named confidence/accuracy.
    quality = float(np.clip(1. - .2*("heterogeneous_background" in flags)-.2*("glare_in_roi" in flags)-min(.3, rim_fraction*2), 0, 1))
    full_mask = np.zeros(inside.shape, bool)
    full_candidate = np.zeros_like(full_mask)
    full_food = np.zeros_like(full_mask)
    full_score = np.zeros(inside.shape, np.float32)
    preprocessed = image_bgr.copy()
    sl = np.s_[y0:y1, x0:x1]
    full_mask[sl], full_candidate[sl], full_food[sl], full_score[sl] = mask, candidate, food, score
    preprocessed[sl] = corrected
    result = SegmentationResult(method, full_mask, preprocessed, full_score, full_candidate, inside, full_food, flags, {
        "parameters": cfg, "score_interpretation": "feature_evidence_not_probability",
        "quality_score": quality, "quality_interpretation": "heuristic_not_accuracy",
        "occupied_area_px": int(mask.sum()), "clean_area_px": int(mask.sum()),
        "envelope_area_px": int(ndi.binary_fill_holes(mask).sum()),
        "food_excluded_px": int(food.sum()), "area_fraction": fraction,
        "seed_fraction": evidence_fraction, "background_yellow_level": bg_level,
        "background_yellow_mad_sigma": bg_sigma, "low_threshold": low, "high_threshold": high,
        "rim_fraction": rim_fraction, "temporal_mask_editing": False,
        "roi_center": list(roi.center), "roi_radius": roi.radius,
        "roi_margin_px": roi.margin_px, **roi_details,
    })
    result.validate(image_bgr.shape[:2])
    return result
