"""Round-3 reliability gate for rigid registration of candidate static features.

R2 remains the estimator. R3 asks whether its estimated motion is resolved above
the remaining spatial disagreement and whether any sufficiently supported region
is made worse. These are development-derived acceptance heuristics, not calibrated
uncertainty or proof of biological-plane alignment. No organism masks are used.
"""
from __future__ import annotations

import numpy as np

from . import registration_v2 as r2
from .contracts import RoiSpec


GATE_DEFAULTS = {
    "min_motion_to_residual": 2.0,
    "residual_floor_px": .05,
    "max_sector_regression_px": .20,
    "min_sector_points": 8,
    "min_evaluated_sectors": 3,
}
DEFAULTS = {**r2.DEFAULTS, **GATE_DEFAULTS}


def _config(config):
    cfg = dict(DEFAULTS)
    if config:
        if set(config)-set(cfg):
            raise ValueError(f"Unknown registration_v3 keys: {sorted(set(config)-set(cfg))}")
        cfg.update(config)
    for key, value in cfg.items():
        if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite scalar")
    gate = {key: cfg.pop(key) for key in GATE_DEFAULTS}
    if any(v <= 0 for v in gate.values()):
        raise ValueError("Gate values must be positive")
    for key in ("min_sector_points", "min_evaluated_sectors"):
        if int(gate[key]) != gate[key]:
            raise ValueError(f"{key} must be an integer")
    if gate["min_sector_points"] < 3 or not 1 <= gate["min_evaluated_sectors"] <= 8:
        raise ValueError("Invalid regional support counts")
    return r2._config(cfg), gate


def regional_evidence(matrix, reference_points, moving_points, roi: RoiSpec, config=None):
    """Evaluate already tracked points without fitting or selecting a matrix.

    Points must be from R2's fit-excluded acceptance split. They are not an external
    test set: both R2 and R3 use them to accept/reject the candidate transform.
    """
    _, cfg = _config(config)
    a = np.asarray(reference_points, np.float64).reshape(-1, 2)
    b = np.asarray(moving_points, np.float64).reshape(-1, 2)
    matrix = np.asarray(matrix, np.float64)
    if (a.shape != b.shape or not len(a) or matrix.shape != (2, 3)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(matrix).all()):
        raise ValueError("Finite matching nonempty point arrays and a 2x3 matrix required")
    transformed = r2._apply(b, matrix)
    before = np.linalg.norm(a-b, axis=1)
    after = np.linalg.norm(a-transformed, axis=1)
    correction = np.linalg.norm(transformed-b, axis=1)
    sectors = (np.mod(np.arctan2(a[:, 1]-roi.center[1], a[:, 0]-roi.center[0]),
                      2*np.pi)*8/(2*np.pi)).astype(int)
    records = []
    for sector in range(8):
        inside = sectors == sector
        count = int(inside.sum())
        if count < cfg["min_sector_points"]:
            continue
        pre, post = float(np.median(before[inside])), float(np.median(after[inside]))
        records.append({"sector": sector, "points": count,
                        "before_median_px": pre, "after_median_px": post,
                        "regression_px": post-pre,
                        "worsened": bool(post-pre > cfg["max_sector_regression_px"])})
    median_correction = float(np.median(correction))
    median_residual = float(np.median(after))
    return {"correction_median_px": median_correction,
            "residual_median_px": median_residual,
            "motion_to_residual": median_correction/max(cfg["residual_floor_px"], median_residual),
            "regions": records, "evaluated_regions": len(records),
            "worsened_regions": [r["sector"] for r in records if r["worsened"]],
            "support_requirement_met": len(records) >= cfg["min_evaluated_sectors"],
            "interpretation": "heuristic_signal_vs_residual_not_calibrated_uncertainty"}


def estimate_motion(reference_bgr, moving_bgr, roi: RoiSpec, config=None):
    """R2-compatible (moving-to-reference rigid matrix, JSON diagnostics).

    Invalid estimates return identity with valid=False and aligned metrics null.
    Checked sub-resolution identity remains valid=True/applied=False. R3 never
    increases the set of applied transforms relative to the same R2 parameters.
    """
    base_cfg, gate_cfg = _config(config)
    matrix, original = r2.estimate_motion(reference_bgr, moving_bgr, roi, base_cfg)
    info = dict(original)
    info.update(method="outer_feature_rigid_with_regional_resolution_gate",
                estimator_method=original["method"], gate_config=gate_cfg,
                upstream_valid=bool(original["valid"]), upstream_reason=original["reason"],
                biological_plane_correction_verified=False,
                gate_evidence=None,
                evidence_scope="R2_fit_excluded_tracks_reused_for_acceptance_not_external_test")
    if not info["valid"]:
        info["gate_status"] = "upstream_rejected"
        return matrix.copy(), info
    if not info["applied"]:
        info["gate_status"] = "identity_no_correction_to_validate"
        return matrix.copy(), info
    evidence = regional_evidence(matrix, original["heldout_reference_points"],
                                 original["heldout_moving_points"], roi, gate_cfg)
    info["gate_evidence"] = evidence
    failures = []
    if not evidence["support_requirement_met"]:
        failures.append("insufficient_regional_validation_support")
    if evidence["motion_to_residual"] < gate_cfg["min_motion_to_residual"]:
        failures.append("motion_not_resolved_above_spatial_residual")
    if evidence["worsened_regions"]:
        failures.append("correction_worsens_supported_static_region")
    info["gate_failures"] = failures
    if not failures:
        info.update(reason="accepted_resolved_regionally_nonworsening_motion", gate_status="accepted")
        return matrix.copy(), info
    info.update(valid=False, applied=False, reason=failures[0], gate_status="rejected",
                rejected_candidate_matrix=np.asarray(matrix, float).tolist())
    for key in ("background_residual_after_median_px", "background_residual_after_p90_px",
                "heldout_background_ncc_after"):
        info["candidate_"+key] = info[key]
        info[key] = None
    return np.float32([[1, 0, 0], [0, 1, 0]]), info
