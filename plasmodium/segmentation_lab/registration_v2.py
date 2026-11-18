"""Rigid dish-motion estimate from outer static features, with held-out checks.

No organism mask is accepted or optimized. No image is modified. Matrices map
moving image coordinates into reference image coordinates. Held-out features
are excluded from fitting but used for acceptance, not a biological ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contracts import RoiSpec


DEFAULTS = {
    "max_dimension": 1280, "inner_radius_ratio": 1.10, "outer_radius_ratio": 1.70,
    "max_corners": 1000, "quality_level": .008, "min_distance_px": 14.,
    "max_forward_backward_px": 1.5, "min_patch_ncc": .70,
    "ransac_threshold_px": 2., "min_matches": 24, "min_holdout": 8,
    "min_inlier_fraction": .65, "min_sectors": 3,
    "max_center_shift_px": 35., "max_rotation_deg": 2.,
    "max_holdout_median_px": .9, "max_holdout_p90_px": 2.5,
    "max_split_disagreement_px": 1.2, "min_motion_px": .35,
    "min_residual_gain_px": .08, "max_residual_ratio": .90,
}


def _config(config):
    cfg = dict(DEFAULTS)
    if config:
        unknown = set(config)-set(cfg)
        if unknown:
            raise ValueError(f"Unknown registration_v2 keys: {sorted(unknown)}")
        cfg.update(config)
    if any(not np.isfinite(float(v)) for v in cfg.values()):
        raise ValueError("Registration values must be finite")
    if not (32 <= cfg["max_dimension"] and 1 <= cfg["inner_radius_ratio"] < cfg["outer_radius_ratio"]
            and cfg["min_matches"] >= cfg["min_holdout"]+8 and cfg["min_holdout"] >= 4
            and 0 < cfg["quality_level"] < 1 and cfg["max_corners"] >= cfg["min_matches"]):
        raise ValueError("Invalid registration geometry or feature counts")
    for name in ("min_inlier_fraction", "max_residual_ratio"):
        if not 0 < cfg[name] <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    if not -1 <= cfg["min_patch_ncc"] <= 1 or not 1 <= cfg["min_sectors"] <= 8:
        raise ValueError("Invalid patch correlation or sector count")
    for name in ("min_distance_px", "max_forward_backward_px", "ransac_threshold_px",
                 "max_center_shift_px", "max_rotation_deg", "max_holdout_median_px",
                 "max_holdout_p90_px", "max_split_disagreement_px", "min_motion_px"):
        if cfg[name] <= 0:
            raise ValueError(f"{name} must be positive")
    return cfg


def _gray(image, max_dimension):
    scale = min(1., float(max_dimension)/max(image.shape[:2]))
    size = (int(round(image.shape[1]*scale)), int(round(image.shape[0]*scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray), scale


def _apply(points, matrix):
    return np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]


def _rigid(source, target):
    """Least-squares Euclidean fit (no scale/shear), source -> target."""
    a, b = np.asarray(source, np.float64), np.asarray(target, np.float64)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return np.column_stack((rotation, cb-rotation@ca)).astype(np.float32)


def _ncc(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    a -= a.mean(); b -= b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den > 1e-8 else -1.


def _patch_ncc(reference, moving, reference_points, moving_points):
    return np.array([_ncc(cv2.getRectSubPix(reference, (11, 11), tuple(map(float,p))),
                         cv2.getRectSubPix(moving, (11, 11), tuple(map(float,q))))
                     for p, q in zip(reference_points, moving_points)])


def evaluate_motion(matrix, reference_points, moving_points):
    """Residuals of independently tracked points; points are never refitted here."""
    before = np.linalg.norm(np.asarray(reference_points)-np.asarray(moving_points), axis=1)
    after = np.linalg.norm(np.asarray(reference_points)-_apply(moving_points,matrix), axis=1)
    return {"background_residual_before_median_px": float(np.median(before)),
            "background_residual_after_median_px": float(np.median(after)),
            "background_residual_before_p90_px": float(np.percentile(before,90)),
            "background_residual_after_p90_px": float(np.percentile(after,90))}


def estimate_motion(reference_bgr, moving_bgr, roi: RoiSpec, config=None):
    """Return (2x3 moving-to-reference matrix, JSON diagnostics).

    Rejection returns identity and valid=False. A statistically unresolved small
    displacement may return a checked identity with valid=True, applied=False.
    Consumers must retain valid/reason, not treat every identity as successful.
    """
    cfg = _config(config)
    if (reference_bgr.shape != moving_bgr.shape or reference_bgr.ndim != 3
            or reference_bgr.shape[2] != 3 or reference_bgr.dtype != np.uint8
            or moving_bgr.dtype != np.uint8):
        raise ValueError("Images must be matching H x W x 3 uint8 BGR arrays")
    if not np.isfinite([*roi.center,roi.radius]).all() or roi.radius <= 0:
        raise ValueError("A finite positive dish ROI is required")
    identity = np.float32([[1,0,0],[0,1,0]])
    info = dict(valid=False, applied=False, reason="not_estimated", method="outer_feature_lk_ransac_rigid",
                input_mutated=False, config=cfg, matches=0, train_matches=0, holdout_matches=0,
                inlier_fraction=None, inlier_sectors=0, split_disagreement_px=None,
                center_shift_px=None, rotation_deg=None, rim_displacement_px=None,
                background_residual_before_median_px=None, background_residual_after_median_px=None,
                background_residual_before_p90_px=None, background_residual_after_p90_px=None,
                heldout_background_ncc_before=None, heldout_background_ncc_after=None,
                heldout_reference_points=[], heldout_moving_points=[],
                evidence_scope="outer_dish_features_holdout_excluded_from_fit_used_for_acceptance")

    def reject(reason):
        info["reason"] = reason
        # A rejected candidate is not the returned identity. Keep its residual
        # explicitly separate; do not publish it as a successful aligned result.
        for key in ("background_residual_after_median_px", "background_residual_after_p90_px",
                    "heldout_background_ncc_after"):
            if info[key] is not None:
                info["candidate_"+key] = info[key]
                info[key] = None
        return identity.copy(), info

    ref, scale = _gray(reference_bgr, cfg["max_dimension"])
    cur, _ = _gray(moving_bgr, cfg["max_dimension"])
    yy, xx = np.ogrid[:ref.shape[0],:ref.shape[1]]
    rr = np.hypot(xx-roi.center[0]*scale, yy-roi.center[1]*scale)
    domain = ((rr >= roi.radius*scale*cfg["inner_radius_ratio"])
              & (rr <= roi.radius*scale*cfg["outer_radius_ratio"]))
    domain[:8]=False; domain[-8:]=False; domain[:,:8]=False; domain[:,-8:]=False
    info.update(feature_region_area_px=int(domain.sum()), working_scale=float(scale))
    p0 = cv2.goodFeaturesToTrack(ref, int(cfg["max_corners"]), float(cfg["quality_level"]),
                                max(3.,cfg["min_distance_px"]*scale), mask=domain.astype(np.uint8)*255,
                                blockSize=5)
    if p0 is None or len(p0)<cfg["min_matches"]:
        return reject("insufficient_outer_static_features")
    info["detected_features"] = int(len(p0))
    lk = dict(winSize=(21,21),maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
    p1, ok1, _ = cv2.calcOpticalFlowPyrLK(ref,cur,p0,None,**lk)
    if p1 is None:
        return reject("forward_tracking_failed")
    pback, okback, _ = cv2.calcOpticalFlowPyrLK(cur,ref,p1,None,**lk)
    if pback is None:
        return reject("backward_tracking_failed")
    a,b = p0.reshape(-1,2),p1.reshape(-1,2)
    fb = np.linalg.norm(a-pback.reshape(-1,2),axis=1)/scale
    good = (ok1.ravel()>0)&(okback.ravel()>0)&np.isfinite(b).all(axis=1)&(fb<=cfg["max_forward_backward_px"])
    # Require a track to remain outside the biological interior in both frames.
    rr_cur = np.linalg.norm(b-np.array(roi.center)*scale,axis=1)
    good &= (rr_cur>=roi.radius*scale*cfg["inner_radius_ratio"])&(rr_cur<=roi.radius*scale*cfg["outer_radius_ratio"])
    good &= (b[:,0]>=8)&(b[:,0]<cur.shape[1]-8)&(b[:,1]>=8)&(b[:,1]<cur.shape[0]-8)
    a,b = a[good],b[good]
    if len(a):
        photometric = _patch_ncc(ref,cur,a,b)>=cfg["min_patch_ncc"]
        a,b = a[photometric]/scale,b[photometric]/scale
    info["matches"] = int(len(a))
    if len(a)<cfg["min_matches"]:
        return reject("insufficient_consistent_tracks")
    # Fixed split, independent of fit residual and motion direction. Held-out
    # tracks never participate in RANSAC or Euclidean least-squares fitting.
    order = np.random.default_rng(4629).permutation(len(a))
    ntest=max(int(cfg["min_holdout"]),len(a)//3)
    test,train=order[:ntest],order[ntest:]
    info.update(train_matches=int(len(train)),holdout_matches=int(len(test)),
                heldout_reference_points=a[test].astype(float).tolist(),
                heldout_moving_points=b[test].astype(float).tolist())
    proposal,inliers=cv2.estimateAffinePartial2D(b[train],a[train],method=cv2.RANSAC,
         ransacReprojThreshold=float(cfg["ransac_threshold_px"]),maxIters=2000,confidence=.999,refineIters=10)
    if proposal is None or inliers is None:
        return reject("ransac_failed")
    info["ransac_similarity_scale"] = float(np.linalg.norm(proposal[:,0]))
    inliers=inliers.ravel().astype(bool)
    info["inlier_fraction"]=float(inliers.mean())
    if inliers.sum()<8 or inliers.mean()<cfg["min_inlier_fraction"]:
        return reject("inconsistent_outer_motion")
    ta,tb=a[train][inliers],b[train][inliers]
    candidate=_rigid(tb,ta)
    # Similarity RANSAC proposes inliers only. The actual returned model is
    # refitted as Euclidean, so scale, shear and nonrigid warps are impossible.
    residual=np.linalg.norm(ta-_apply(tb,candidate),axis=1)
    accepted=residual<=cfg["ransac_threshold_px"]
    if accepted.sum()<8:
        return reject("rigid_model_inconsistent")
    ta,tb=ta[accepted],tb[accepted]
    candidate=_rigid(tb,ta)
    info["rigid_inlier_count"]=int(len(ta))
    info["train_rigid_residual_median_px"]=float(np.median(np.linalg.norm(ta-_apply(tb,candidate),axis=1)))
    angles=np.mod(np.arctan2(ta[:,1]-roi.center[1],ta[:,0]-roi.center[0]),2*np.pi)
    sectors=np.unique((angles*8/(2*np.pi)).astype(int))
    info["inlier_sectors"]=int(len(sectors))
    if len(sectors)<cfg["min_sectors"]:
        return reject("features_concentrated_in_too_few_regions")
    controls=np.array([roi.center,(roi.center[0]+roi.radius,roi.center[1]),
                       (roi.center[0]-roi.radius,roi.center[1]),
                       (roi.center[0],roi.center[1]+roi.radius),
                       (roi.center[0],roi.center[1]-roi.radius)],np.float32)
    first=_rigid(tb[::2],ta[::2]);second=_rigid(tb[1::2],ta[1::2])
    disagreement=float(np.max(np.linalg.norm(_apply(controls,first)-_apply(controls,second),axis=1)))
    info["split_disagreement_px"]=disagreement
    info["candidate_matrix"]=candidate.astype(float).tolist()
    motion=np.linalg.norm(_apply(controls,candidate)-controls,axis=1)
    rotation=float(np.degrees(np.arctan2(candidate[1,0],candidate[0,0])))
    info.update(center_shift_px=float(motion[0]),rotation_deg=rotation,rim_displacement_px=float(motion.max()))
    info.update(evaluate_motion(candidate,a[test],b[test]))
    holdout_angles=np.mod(np.arctan2(a[test,1]-roi.center[1],a[test,0]-roi.center[0]),2*np.pi)
    holdout_sectors=(holdout_angles*8/(2*np.pi)).astype(int)
    holdout_residual=np.linalg.norm(a[test]-_apply(b[test],candidate),axis=1)
    info["heldout_sector_residuals_px"]={str(s):float(np.median(holdout_residual[holdout_sectors==s]))
        for s in np.unique(holdout_sectors) if np.count_nonzero(holdout_sectors==s)>=5}
    if motion[0]>cfg["max_center_shift_px"] or abs(rotation)>cfg["max_rotation_deg"]:
        return reject("motion_outside_allowed_rigid_range")
    if disagreement>cfg["max_split_disagreement_px"]:
        return reject("independent_fit_splits_disagree")
    if (info["background_residual_after_median_px"]>cfg["max_holdout_median_px"]
            or info["background_residual_after_p90_px"]>cfg["max_holdout_p90_px"]):
        return reject("heldout_static_features_not_aligned")
    if any(value>cfg["max_holdout_p90_px"] for value in info["heldout_sector_residuals_px"].values()):
        return reject("local_static_regions_disagree")
    small=candidate.copy();small[:,2]*=scale
    aligned=cv2.warpAffine(cur,small,(ref.shape[1],ref.shape[0]),flags=cv2.INTER_LINEAR)
    zeros=a[test]*scale
    info["heldout_background_ncc_before"]=float(np.median(_patch_ncc(ref,cur,zeros,zeros)))
    info["heldout_background_ncc_after"]=float(np.median(_patch_ncc(ref,aligned,zeros,zeros)))
    if motion.max()<cfg["min_motion_px"]:
        info.update(valid=True,reason="motion_below_resolution_identity",applied=False)
        info["candidate_background_residual_after_median_px"]=info["background_residual_after_median_px"]
        info.update(evaluate_motion(identity,a[test],b[test]))
        info["heldout_background_ncc_after"]=info["heldout_background_ncc_before"]
        return identity.copy(),info
    before=info["background_residual_before_median_px"]
    after=info["background_residual_after_median_px"]
    if before-after<cfg["min_residual_gain_px"] or after>before*cfg["max_residual_ratio"]:
        return reject("no_independent_residual_improvement")
    info.update(valid=True,applied=True,reason="accepted_heldout_static_features")
    return candidate,info
