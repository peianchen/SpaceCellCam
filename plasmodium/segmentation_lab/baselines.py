"""Reproduce existing spatial masks without executing an application entry point.

These adapters deliberately preserve baseline weaknesses, including food pixels
and the original integer circular ROI.  They compare single original-resolution
images; they are not replays of the old half-resolution/temporal pipeline.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

from .contracts import RoiSpec, SegmentationResult


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_PATH = _PROJECT_ROOT / "plasmodium_brightfield_analysis.py"
_V2_ROOT = _PROJECT_ROOT / "plasmodium_analysis_v2" / "plasmodium_v2"


def _read_source(path: Path) -> tuple[str, str]:
    content = path.read_bytes()
    return content.decode("utf-8-sig"), hashlib.sha256(content).hexdigest()


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


@lru_cache(maxsize=8)
def _extract_main_hsv_params(source: str, entry: str) -> tuple[dict, int]:
    """Read only the literal HSV dictionary in the selected entry-point scope."""
    tree = ast.parse(source)
    scopes = [
        node for node in tree.body
        if (_is_main_guard(node) if entry == "__main__" else (
            isinstance(node, ast.FunctionDef) and node.name == entry
        ))
    ]
    if len(scopes) != 1:
        raise ValueError(f"Expected exactly one {entry!r} configuration scope")
    assignments = [
        node for node in scopes[0].body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "hsv_params"
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Expected exactly one entry-point hsv_params assignment")
    assignment = assignments[0]
    value = assignment.value
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "dict" and not value.args
            and all(keyword.arg is not None for keyword in value.keywords)):
        raise ValueError("hsv_params must be dict(...) with literal named values")
    params = {keyword.arg: ast.literal_eval(keyword.value) for keyword in value.keywords}
    return params, assignment.lineno


def _import_snapshot(name: str, path: Path, source: str, package: str = "") -> ModuleType:
    """Import a recorded source snapshot under a non-main name, without pyc writes."""
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = package
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _source_record(path: Path, digest: str, role: str, line: int | None = None) -> dict:
    record = {"path": str(path), "sha256": digest, "role": role}
    if line is not None:
        record["parameter_assignment_line"] = line
    return record


def _baseline_implementation(method: str):
    if method == "legacy_original":
        source, digest = _read_source(_LEGACY_PATH)
        params, line = _extract_main_hsv_params(source, "__main__")
        module = _import_snapshot(f"_segmentation_lab_legacy_{digest[:16]}", _LEGACY_PATH, source)
        records = [_source_record(_LEGACY_PATH, digest, "implementation_and_parameters", line)]
        return module.segment_hsv_circle, dict(params), records
    if method == "v2_spatial":
        engine_path = _V2_ROOT / "engine.py"
        engine_source, engine_digest = _read_source(engine_path)
        params, line = _extract_main_hsv_params(engine_source, "run")
        segmentation_path = _V2_ROOT / "segmentation.py"
        helpers_path = _V2_ROOT / "mask_utils.py"
        segmentation_source, segmentation_digest = _read_source(segmentation_path)
        helpers_source, helpers_digest = _read_source(helpers_path)
        package_name = f"_segmentation_lab_v2_{segmentation_digest[:12]}_{helpers_digest[:12]}"
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = [str(_V2_ROOT)]
            package.__package__ = package_name
            sys.modules[package_name] = package
        _import_snapshot(f"{package_name}.mask_utils", helpers_path, helpers_source, package_name)
        module = _import_snapshot(
            f"{package_name}.segmentation", segmentation_path, segmentation_source, package_name
        )
        records = [
            _source_record(engine_path, engine_digest, "parameters_only", line),
            _source_record(segmentation_path, segmentation_digest, "implementation"),
            _source_record(helpers_path, helpers_digest, "mask_helpers"),
        ]
        return module.segment_hsv_circle, dict(params), records
    raise ValueError(f"Unknown baseline method {method!r}; use legacy_original or v2_spatial")


def run_baseline(image_bgr: np.ndarray, roi: RoiSpec, method: str) -> SegmentationResult:
    """Return the existing occupied mask at original image resolution.

    Input must be uint8 BGR, as decoded from the video.  No normalization, food
    exclusion, ROI margin, resizing, or temporal post-processing is added.
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or min(image_bgr.shape[:2]) == 0):
        raise ValueError("Baseline input must be a nonempty H x W x 3 uint8 BGR image")
    geometry = np.asarray([*roi.center, roi.radius, roi.margin_px], dtype=float)
    if geometry.size != 4 or not np.isfinite(geometry).all() or int(roi.radius) < 1:
        raise ValueError("Baseline ROI must have finite geometry and a radius of at least one pixel")
    center = tuple(int(value) for value in roi.center)
    radius = int(roi.radius)
    segment, params, sources = _baseline_implementation(method)
    _, returned_clean, debug = segment(
        image_bgr, dish_center=center, dish_radius=radius, **params
    )
    occupied = np.asarray(debug["occupied_mask"], dtype=bool).copy()
    clean = np.asarray(debug.get("clean_mask", returned_clean), dtype=bool)
    envelope = np.asarray(debug["envelope_mask"], dtype=bool)
    yy, xx = np.ogrid[:image_bgr.shape[0], :image_bgr.shape[1]]
    roi_mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    candidate = np.zeros_like(occupied)
    for key in ("seed_mask", "yellow_chroma_candidate_mask", "relaxed_candidate_mask",
                "vesselness_candidate_mask"):
        if key in debug:
            candidate |= np.asarray(debug[key], dtype=bool)
    flags = ["baseline_food_not_excluded", "baseline_single_frame_no_motion_compensation"]
    if params.get("keep_largest"):
        flags.append("baseline_keeps_largest_component")
    if not occupied.any():
        flags.append("baseline_empty_mask")
    clean_diff = int(np.count_nonzero(clean ^ occupied))
    if clean_diff:
        flags.append("baseline_clean_differs_from_reported_occupied")
    diagnostics = {
        "source_files": sources,
        "hsv_params": params,
        "analysis_scale": 1.0,
        "comparison_scope": "single_frame_spatial_at_common_original_resolution",
        "complete_original_run_replayed": False,
        "temporal_recovery_applied": False,
        "food_exclusion_applied": False,
        "score_semantics": "binary_occupied_mask_not_probability",
        "preprocessed_semantics": "original_bgr_without_external_preprocessing",
        "candidate_semantics": "union_of_existing_debug_candidates_after_legacy_component_filtering",
        "reported_mask_layer": "occupied_mask",
        "occupied_area_px": int(occupied.sum()),
        "clean_area_px": int(clean.sum()),
        "envelope_area_px": int(envelope.sum()),
        "clean_differs_from_occupied": bool(clean_diff),
        "clean_occupied_difference_px": clean_diff,
        "clean_added_pixels_px": int(np.count_nonzero(clean & ~occupied)),
        "clean_removed_pixels_px": int(np.count_nonzero(occupied & ~clean)),
        "requested_roi": {"center": [float(value) for value in roi.center],
                          "radius": float(roi.radius), "margin_px": float(roi.margin_px)},
        "effective_roi": {"center": list(center), "radius": radius, "margin_px": 0},
    }
    result = SegmentationResult(
        method=method, mask=occupied, preprocessed=image_bgr.copy(),
        score=occupied.astype(np.float32), candidate=candidate, roi_mask=roi_mask,
        excluded_food=np.zeros_like(occupied), qc_flags=flags, diagnostics=diagnostics,
    )
    result.validate(image_bgr.shape[:2])
    return result
