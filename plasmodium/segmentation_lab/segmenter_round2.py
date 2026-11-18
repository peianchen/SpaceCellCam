"""Round 2 additive branch path: local noise and directional ridge evidence.

Frozen round-1 broad-body/food/ROI decisions remain explicit dependencies.
Scores are feature evidence, not calibrated confidence. No temporal smoothing.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np
from scipy import ndimage as ndi

from .contracts import RoiSpec, SegmentationResult
from .segmenter import segment as segment_round1, _components

DEFAULTS = {
    "noise_floor": .003, "noise_tile_px": 64,
    "contrast_floor": .018, "contrast_noise_multiplier": 2.5,
    "ridge_floor": .008, "ridge_noise_multiplier": 1.8,
    "min_component_px": 24, "min_seed_px": 5, "min_branch_span_px": 20,
}


def _source_hash(image):
    return hashlib.sha256(f"BGR|{image.shape}|{image.dtype}".encode()+np.ascontiguousarray(image).tobytes()).hexdigest()


def prepare_baseline(image_bgr, roi, base_config=None):
    """Compute the frozen baseline and bind reuse to source pixels and geometry."""
    result = segment_round1(image_bgr,roi,config=base_config)
    result.diagnostics = {**result.diagnostics, "round2_cache_binding": {
        "source_sha256":_source_hash(image_bgr),"roi":[*roi.center,roi.radius,roi.margin_px],
        "base_config":dict(base_config or {})}}
    return result


def _noise_map(yellow: np.ndarray, usable: np.ndarray, tile: int, floor: float):
    # High-frequency residual estimates sensor/compression variation locally;
    # broad shading and between-region color differences are not counted as noise.
    residual = yellow - cv2.GaussianBlur(yellow, (0, 0), .8)
    h, w = yellow.shape
    grid = np.full(((h+tile-1)//tile, (w+tile-1)//tile), floor, np.float32)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            sl = np.s_[row*tile:min(h,(row+1)*tile), col*tile:min(w,(col+1)*tile)]
            values = residual[sl][usable[sl]]
            if len(values) >= 32:
                med = np.median(values)
                grid[row,col] = max(floor, float(np.median(np.abs(values-med))*1.4826))
    noise = cv2.resize(grid, (w,h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(noise, (0,0), tile/4)


def _ridge(yellow: np.ndarray):
    """Scale-normalized Gaussian Hessian; suppress dots and flat areas."""
    strength = np.zeros_like(yellow)
    for sigma in (1., 2., 4.):
        xx = ndi.gaussian_filter(yellow, sigma, order=(0,2))*sigma**2
        yy = ndi.gaussian_filter(yellow, sigma, order=(2,0))*sigma**2
        xy = ndi.gaussian_filter(yellow, sigma, order=(1,1))*sigma**2
        discriminant = np.sqrt((xx-yy)**2+4*xy**2)
        lo, hi = (xx+yy-discriminant)/2, (xx+yy+discriminant)/2
        # Bright ridge: strong negative curvature across it and little curvature
        # along it. Positive hi at branches is allowed; blobs are discouraged.
        anisotropy = np.clip(1-np.abs(hi)/(np.abs(lo)+1e-6), 0, 1)
        strength = np.maximum(strength, np.maximum(-lo,0)*anisotropy)
    return strength


def segment(image_bgr: np.ndarray, roi: RoiSpec, config: dict | None = None,
            base_config: dict | None = None,
            baseline: SegmentationResult | None = None) -> SegmentationResult:
    cfg = dict(DEFAULTS)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    if not np.isfinite([*roi.center,roi.radius,roi.margin_px]).all() or roi.radius <= roi.margin_px or roi.margin_px < 0:
        raise ValueError("Invalid circular ROI")
    if config:
        if set(config)-set(cfg):
            raise ValueError(f"Unknown round2 parameters: {sorted(set(config)-set(cfg))}")
        cfg.update(config)
    for key,value in cfg.items():
        if isinstance(value,(bool,str)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    for key in ("noise_tile_px","min_component_px","min_seed_px"):
        if int(cfg[key]) != cfg[key]:
            raise ValueError(f"{key} must be an integer")
    if cfg["noise_tile_px"] < 8:
        raise ValueError("noise_tile_px must be at least 8")
    if baseline is not None:
        binding=baseline.diagnostics.get("round2_cache_binding",{})
        expected={"source_sha256":_source_hash(image_bgr),"roi":[*roi.center,roi.radius,roi.margin_px],"base_config":base_config or {}}
        if binding != expected:
            raise ValueError("Cached baseline does not match source image, ROI or base_config; use prepare_baseline")
    base = baseline if baseline is not None else prepare_baseline(image_bgr,roi,base_config)
    base.validate(image_bgr.shape[:2])
    yy,xx = np.nonzero(base.roi_mask)
    y0,y1 = max(0,yy.min()-48),min(image_bgr.shape[0],yy.max()+49)
    x0,x1 = max(0,xx.min()-48),min(image_bgr.shape[1],xx.max()+49)
    sl = np.s_[y0:y1,x0:x1]
    crop = image_bgr[sl]
    b,g,r = cv2.split(crop.astype(np.float32)/255.)
    yellow = (np.minimum(r,g)-b)/(np.minimum(r,g)+b+.08)
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    usable = base.roi_mask[sl] & ~base.excluded_food[sl] & (crop.max(axis=2)>12) & (crop.min(axis=2)<250)
    chromatic = (hsv[...,0]>=10)&(hsv[...,0]<=58)&(hsv[...,1]>=12)
    noise = _noise_map(yellow,usable,int(cfg["noise_tile_px"]),float(cfg["noise_floor"]))
    smooth = cv2.GaussianBlur(yellow,(0,0),.8)
    local_background = cv2.GaussianBlur(yellow,(0,0),12.)
    contrast = smooth-local_background
    ridge = _ridge(smooth)
    threshold = np.maximum(cfg["contrast_floor"], noise*cfg["contrast_noise_multiplier"])
    seed_threshold = np.maximum(cfg["ridge_floor"],noise*cfg["ridge_noise_multiplier"])
    candidate = (contrast>threshold)&usable&chromatic
    seed = candidate & (ridge>seed_threshold)
    branches = _components(candidate,seed,int(cfg["min_component_px"]),int(cfg["min_seed_px"]))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(branches.astype(np.uint8),8)
    keep = np.hypot(stats[:,cv2.CC_STAT_WIDTH],stats[:,cv2.CC_STAT_HEIGHT]) >= cfg["min_branch_span_px"]
    keep[0] = False
    branches = keep[labels]
    mask = base.mask.copy()
    mask[sl] |= branches
    added = mask & ~base.mask
    all_candidate = base.candidate.copy()
    all_candidate[sl] |= candidate
    score = base.score.copy()
    score[sl] = np.maximum(score[sl],np.where(usable,np.clip(ridge/(2*seed_threshold),0,1),0))
    flags = list(base.qc_flags)
    if added.any():
        flags.append("local_ridge_additions_require_review")
    if mask.any() and "empty_no_supported_tissue" in flags:
        flags.remove("empty_no_supported_tissue")
    border = base.roi_mask & ~cv2.erode(base.roi_mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))).astype(bool)
    rim_fraction = float((mask&border).sum()/max(1,mask.sum()))
    for flag in ("tissue_near_roi_boundary","large_coverage_check_background"):
        if flag in flags:flags.remove(flag)
    if (mask&border).any():flags.append("tissue_near_roi_boundary")
    if mask.sum()/base.roi_mask.sum()>.65:flags.append("large_coverage_check_background")
    quality = float(np.clip(1-.2*("heterogeneous_background" in flags)-.2*("glare_in_roi" in flags)-min(.3,rim_fraction*2),0,1))
    diagnostics = dict(base.diagnostics)
    diagnostics.update({"round2_parameters":cfg,"base_method":base.method,
        "base_area_px":int(base.mask.sum()),"added_branch_area_px":int(added.sum()),
        "occupied_area_px":int(mask.sum()),"clean_area_px":int(mask.sum()),
        "envelope_area_px":int(ndi.binary_fill_holes(mask).sum()),
        "area_fraction":float(mask.sum()/base.roi_mask.sum()),
        "local_noise_median":float(np.median(noise[usable])) if usable.any() else None,
        "local_noise_p90":float(np.percentile(noise[usable],90)) if usable.any() else None,
        "line_seed_area_px":int(seed.sum()),"temporal_mask_editing":False,
        "quality_interpretation":"heuristic_not_accuracy", "quality_score":quality,
        "rim_fraction":rim_fraction,
        "branch_method":"local_noise_normalized_multiscale_hessian_hysteresis"})
    result = SegmentationResult("local_ridge_round2",mask,base.preprocessed.copy(),score,
         all_candidate,base.roi_mask.copy(),base.excluded_food.copy(),flags,diagnostics)
    result.validate(image_bgr.shape[:2])
    return result
