"""Read-only video sampling and temporal diagnostics; never edit tissue masks.

Run: python -m segmentation_lab.temporal_qc VIDEO --start 100 --count 12
     --output NEW_DIRECTORY --config CONFIG.json
Registration changes diagnostic coordinates only. Acquisition time is unknown
unless an acquisition interval is supplied; playback FPS is never substituted.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import RoiSpec
from .segmenter import DEFAULTS as SEGMENT_DEFAULTS, segment
from .visualize import mask_overlay, write_image


REGISTRATION_DEFAULTS = {
    "max_dimension": 720,
    "max_center_shift_px": 35.0,
    "max_rotation_deg": 2.0,
    "min_ecc_score": 0.50,
    "min_phase_score": 0.35,
    "min_rim_feature_std": 1.0,
    "min_registered_rim_correlation": 0.40,
    "min_overlap_fraction": 0.90,
}
_ROOT = Path(__file__).resolve().parents[1]
_REGISTRATION_SOURCE = _ROOT / "ai_segmentation_pilot" / "plasmodium_ai" / "registration.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _registration_module():
    """Load only the existing registration file, without its dataset package."""
    content = _REGISTRATION_SOURCE.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    name = "_segmentation_lab_rim_registration_" + digest[:16]
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    module.__file__ = str(_REGISTRATION_SOURCE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(content, str(_REGISTRATION_SOURCE), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.source_sha256 = digest
    return module


def _registration_config(config: dict | None) -> dict:
    cfg = dict(REGISTRATION_DEFAULTS)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown registration parameters: {sorted(unknown)}")
        cfg.update(config)
    if any(not np.isfinite(float(value)) for value in cfg.values()):
        raise ValueError("Registration parameters must be finite")
    if (int(cfg["max_dimension"]) < 32 or cfg["max_center_shift_px"] <= 0
            or not 0 < cfg["max_rotation_deg"] <= 10 or cfg["min_rim_feature_std"] <= 0):
        raise ValueError("Invalid registration scale or motion limits")
    if any(not 0 <= cfg[key] <= 1 for key in (
            "min_ecc_score", "min_phase_score", "min_registered_rim_correlation", "min_overlap_fraction")):
        raise ValueError("Registration quality thresholds must be between zero and one")
    return cfg


def _warp(mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(bool)


def _estimate_rim_registration(previous: np.ndarray, current: np.ndarray,
                               roi: RoiSpec, cfg: dict) -> tuple[np.ndarray, dict]:
    module = _registration_module()
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    info = {"registration_valid": False, "registration_method": "not_estimated",
            "registration_score": None, "registration_center_shift_px": None,
            "registration_rotation_deg": None, "registration_rim_correlation": None,
            "registration_overlap_fraction": None, "registration_fallback_reason": "",
            "registration_ref_rim_std": None, "registration_current_rim_std": None}
    ref_feature, rim, scale = module._rim_feature(previous, roi.center, roi.radius, int(cfg["max_dimension"]))
    cur_feature, _, current_scale = module._rim_feature(current, roi.center, roi.radius, int(cfg["max_dimension"]))
    rim_inside = rim > 0
    if rim_inside.sum() < 64:
        info["registration_fallback_reason"] = "insufficient_visible_rim"
        return identity, info
    ref_std = float(np.std(ref_feature[rim_inside]))
    cur_std = float(np.std(cur_feature[rim_inside]))
    info.update(registration_ref_rim_std=ref_std, registration_current_rim_std=cur_std)
    if min(ref_std, cur_std) < cfg["min_rim_feature_std"]:
        info["registration_fallback_reason"] = "insufficient_rim_texture"
        return identity, info
    # The old estimator limits matrix translation at image origin. For rotation
    # about the dish this can be large even with little actual dish displacement.
    # Allow that parameterization, then apply a physical center-displacement gate.
    candidate = module.estimate_registration(
        previous, current, roi.center, roi.radius, max_dimension=int(cfg["max_dimension"]),
        max_translation_px=float("inf"), max_rotation_deg=float(cfg["max_rotation_deg"]),
        min_ecc_score=float(cfg["min_ecc_score"]),
    )
    matrix = np.asarray(candidate.matrix, dtype=np.float32)
    info.update(registration_method=str(candidate.method), registration_score=_finite(candidate.score))
    if not candidate.applied:
        info["registration_fallback_reason"] = str(candidate.method)
        return identity, info
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        info["registration_fallback_reason"] = "nonfinite_or_invalid_matrix"
        return identity, info
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    center = np.array([*roi.center, 1.0], dtype=np.float64)
    center_shift = float(np.linalg.norm(matrix @ center - center[:2]))
    info.update(registration_center_shift_px=center_shift, registration_rotation_deg=rotation)
    if (abs(np.linalg.det(matrix[:, :2]) - 1) > .02
            or abs(rotation) > cfg["max_rotation_deg"]
            or center_shift > cfg["max_center_shift_px"]):
        info["registration_fallback_reason"] = "motion_outside_allowed_rigid_range"
        return matrix, info
    if candidate.method == "phase_translation" and (
            not np.isfinite(candidate.score) or candidate.score < cfg["min_phase_score"]):
        info["registration_fallback_reason"] = "phase_score_below_threshold"
        return matrix, info
    small_matrix = matrix.copy()
    small_matrix[:, 2] *= scale
    aligned = cv2.warpAffine(cur_feature, small_matrix, (ref_feature.shape[1], ref_feature.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    support = _warp(np.ones(previous.shape[:2], bool), matrix)
    overlap = float(support.mean())
    small_support = cv2.resize(support.astype(np.uint8), (ref_feature.shape[1], ref_feature.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    compare = rim_inside & small_support
    correlation = None
    if compare.sum() >= 64 and np.std(aligned[compare]) > 1e-6 and np.std(ref_feature[compare]) > 1e-6:
        correlation = _finite(np.corrcoef(ref_feature[compare], aligned[compare])[0, 1])
    info.update(registration_rim_correlation=correlation, registration_overlap_fraction=overlap)
    if overlap < cfg["min_overlap_fraction"]:
        info["registration_fallback_reason"] = "insufficient_common_field_of_view"
        return matrix, info
    if correlation is None or correlation < cfg["min_registered_rim_correlation"]:
        info["registration_fallback_reason"] = "registered_rim_correlation_below_threshold"
        return matrix, info
    info["registration_valid"] = True
    info["registration_fallback_reason"] = (
        "ecc_not_accepted_translation_only" if candidate.method == "phase_translation" else ""
    )
    return matrix, info


def _overlap_metrics(previous: np.ndarray, current: np.ndarray, prefix: str) -> dict:
    intersection = int(np.count_nonzero(previous & current))
    union = int(np.count_nonzero(previous | current))
    new = int(np.count_nonzero(current & ~previous))
    retracted = int(np.count_nonzero(previous & ~current))
    return {prefix + "iou": intersection / union if union else None,
            prefix + "new_area_px": new, prefix + "retracted_area_px": retracted,
            prefix + "changed_fraction": (new + retracted) / union if union else None,
            prefix + "union_area_px": union}


def _first_pair_record() -> dict:
    """Keep first-frame CSV columns explicit without inventing a comparison."""
    row = {name: None for name in (
        "previous_area_px", "current_area_px", "area_delta_px", "area_jump_fraction",
        "area_jump_flag", "foreground_appeared_after_empty", "registration_score",
        "registration_center_shift_px", "registration_rotation_deg", "registration_rim_correlation",
        "registration_overlap_fraction", "registration_ref_rim_std", "registration_current_rim_std",
        "registered_previous_area_in_common_fov_px", "registered_current_area_in_common_fov_px",
        "registered_common_fov_area_px", "registered_common_fov_fraction_of_previous_roi",
        "same_domain_raw_iou", "same_domain_registered_iou", "same_domain_area_px",
    )}
    row.update({prefix+key: None for prefix in ("raw_", "registered_")
                for key in ("iou", "new_area_px", "retracted_area_px", "changed_fraction", "union_area_px")})
    row.update({f"registration_m{y}{x}": None for y in range(2) for x in range(3)})
    row.update(registration_valid=False, registration_method="no_previous_sample",
               registration_fallback_reason="no_previous_sample", temporal_mask_editing=False,
               registered_comparison_valid=False, registered_comparison_reason="no_previous_sample")
    return row


def analyze_pair(previous_image: np.ndarray, current_image: np.ndarray,
                 previous_mask: np.ndarray, current_mask: np.ndarray, roi: RoiSpec,
                 *, previous_roi_mask: np.ndarray | None = None,
                 current_roi_mask: np.ndarray | None = None,
                 registration_config: dict | None = None, area_jump_fraction: float = .20) -> dict:
    """Compare immutable masks inside the intersection of registered valid ROIs.

    Raw metrics retain the original masks, including any ROI clipping. Registered
    metrics exclude locations not observed inside BOTH effective segmentation
    ROIs; crossing a crop boundary must not be called growth or retraction.
    Do not subtract global raw_iou from common-ROI registered_iou to assess
    alignment. The same_domain pair additionally intersects the unwarped current
    ROI and evaluates BOTH masks on exactly the same reference pixel domain.
    """
    if (previous_image.shape != current_image.shape or previous_image.ndim != 3
            or previous_image.shape[2] != 3 or previous_image.dtype != np.uint8
            or current_image.dtype != np.uint8):
        raise ValueError("Pair images must have matching H x W x 3 uint8 BGR shapes")
    shape = previous_image.shape[:2]
    if (previous_mask.shape != shape or current_mask.shape != shape
            or previous_mask.dtype != np.bool_ or current_mask.dtype != np.bool_):
        raise ValueError("Pair masks must be bool arrays matching the image shape")
    if not np.isfinite(area_jump_fraction) or area_jump_fraction <= 0:
        raise ValueError("area_jump_fraction must be positive and finite")
    if (not np.isfinite([*roi.center, roi.radius, roi.margin_px]).all()
            or roi.radius <= 0 or len(roi.center) != 2):
        raise ValueError("Registration requires a finite positive circular ROI")
    previous_roi_mask = roi.mask(shape) if previous_roi_mask is None else previous_roi_mask
    current_roi_mask = roi.mask(shape) if current_roi_mask is None else current_roi_mask
    for valid_roi in (previous_roi_mask, current_roi_mask):
        if valid_roi.shape != shape or valid_roi.dtype != np.bool_:
            raise ValueError("Pair ROI masks must be bool arrays matching the image shape")
    cfg = _registration_config(registration_config)
    previous_area, current_area = int(previous_mask.sum()), int(current_mask.sum())
    delta = current_area - previous_area
    relative_jump = abs(delta) / previous_area if previous_area else None
    row = {"previous_area_px": previous_area, "current_area_px": current_area,
           "area_delta_px": delta, "area_jump_fraction": relative_jump,
           "area_jump_flag": bool(relative_jump > area_jump_fraction) if relative_jump is not None else bool(current_area),
           "foreground_appeared_after_empty": bool(current_area and not previous_area),
           "temporal_mask_editing": False, **_overlap_metrics(previous_mask, current_mask, "raw_")}
    matrix, registration = _estimate_rim_registration(previous_image, current_image, roi, cfg)
    row.update(registration)
    row.update({f"registration_m{y}{x}": _finite(matrix[y, x]) for y in range(2) for x in range(3)})
    row.update({"registered_" + key: None for key in
                ("iou", "new_area_px", "retracted_area_px", "changed_fraction", "union_area_px")})
    row["registered_previous_area_in_common_fov_px"] = None
    row["registered_current_area_in_common_fov_px"] = None
    row["registered_common_fov_area_px"] = None
    row["registered_common_fov_fraction_of_previous_roi"] = None
    row["registered_comparison_valid"] = False
    row["registered_comparison_reason"] = "registration_untrusted"
    row.update(same_domain_raw_iou=None, same_domain_registered_iou=None,
               same_domain_area_px=None)
    if registration["registration_valid"]:
        support = _warp(np.ones(shape, bool), matrix)
        common = previous_roi_mask & _warp(current_roi_mask, matrix) & support
        row["registered_common_fov_area_px"] = int(common.sum())
        previous_roi_area = int(previous_roi_mask.sum())
        row["registered_common_fov_fraction_of_previous_roi"] = (
            float(common.sum() / previous_roi_area) if previous_roi_area else None)
        # The global raw and registered common-ROI metrics have different
        # support. Hold support fixed for a fair unaligned/aligned comparison.
        same_domain = common & current_roi_mask
        row["same_domain_area_px"] = int(same_domain.sum())
        if same_domain.any():
            same_previous = previous_mask & same_domain
            row["same_domain_raw_iou"] = _overlap_metrics(
                same_previous, current_mask & same_domain, "")["iou"]
            row["same_domain_registered_iou"] = _overlap_metrics(
                same_previous, _warp(current_mask, matrix) & same_domain, "")["iou"]
        if common.any():
            aligned = _warp(current_mask, matrix) & common
            previous_supported = previous_mask & common
            row.update(_overlap_metrics(previous_supported, aligned, "registered_"))
            row["registered_previous_area_in_common_fov_px"] = int(previous_supported.sum())
            row["registered_current_area_in_common_fov_px"] = int(aligned.sum())
            row["registered_comparison_valid"] = True
            row["registered_comparison_reason"] = ""
        else:
            row["registered_comparison_reason"] = "no_common_valid_analysis_roi"
    return row


def _font(size: int):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _panel(image: np.ndarray, roi: RoiSpec, width=420) -> Image.Image:
    cx, cy = roi.center
    x0, y0 = max(0, int(cx-roi.radius)), max(0, int(cy-roi.radius))
    x1, y1 = min(image.shape[1], int(cx+roi.radius+1)), min(image.shape[0], int(cy+roi.radius+1))
    crop = image[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else image
    picture = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    picture.thumbnail((width, width), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, width), "#111b25")
    panel.paste(picture, ((width-picture.width)//2, (width-picture.height)//2))
    return panel


def _save_frame_contact(path: Path, image: np.ndarray, mask: np.ndarray, roi: RoiSpec,
                        frame_index: int, pair: dict | None, previous_image=None,
                        previous_mask=None, previous_roi_mask=None, current_roi_mask=None) -> None:
    panels = [("Current source", _panel(image, roi)),
              ("Current mask (unchanged)", _panel(mask_overlay(image, mask), roi))]
    if pair is not None:
        panels.insert(0, ("Previous mask (unchanged)", _panel(mask_overlay(previous_image, previous_mask), roi)))
        if pair["registered_comparison_valid"]:
            matrix = np.array([[pair[f"registration_m{y}{x}"] for x in range(3)] for y in range(2)], np.float32)
            before_roi = roi.mask(mask.shape) if previous_roi_mask is None else previous_roi_mask
            now_roi = roi.mask(mask.shape) if current_roi_mask is None else current_roi_mask
            common = before_roi & _warp(now_roi, matrix) & _warp(np.ones(mask.shape, bool), matrix)
            aligned = _warp(mask, matrix) & common
            panels.append(("Aligned in common ROI: diagnostic", _panel(mask_overlay(previous_image, aligned), roi)))
        else:
            missing = Image.new("RGB", (420, 420), "#f4e9e6")
            ImageDraw.Draw(missing).text((20, 160), "REGISTRATION UNAVAILABLE", fill="#922b21", font=_font(20))
            panels.append(("No registered comparison", missing))
    canvas = Image.new("RGB", (420*len(panels), 510), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    message = "First sampled frame" if pair is None else (
        f"Same-domain IoU raw={pair['same_domain_raw_iou']} | aligned={pair['same_domain_registered_iou']} | "
        f"valid={pair['registration_valid']} | {pair['registration_fallback_reason']}"
    )
    draw.text((12, 7), f"Frame {frame_index} | original occupied mask; no temporal edits", font=_font(18), fill="#173149")
    draw.text((12, 32), message[:200], font=_font(14), fill="#34495e")
    for index, (label, panel) in enumerate(panels):
        draw.text((index*420+10, 62), label, font=_font(17), fill="#173149")
        canvas.paste(panel, (index*420, 88))
    canvas.save(path, quality=93)


def _load_config(path: Path | None) -> tuple[dict, dict, dict]:
    raw = json.loads(path.read_text(encoding="utf-8-sig")) if path else {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object")
    if "methods" in raw:
        segmentation = dict(raw["methods"].get("adaptive_chroma", {}))
    else:
        segmentation = {key: value for key, value in raw.items() if key not in {"roi", "temporal_qc"}}
    qc = dict(raw.get("temporal_qc", {}))
    allowed = {"registration", "area_jump_fraction", "acquisition_interval_sec"}
    if set(qc) - allowed:
        raise ValueError(f"Unknown temporal_qc keys: {sorted(set(qc)-allowed)}")
    return raw, segmentation, qc


def run_window(video: Path, output: Path, *, start=0, count=12, step=1,
               roi: RoiSpec = RoiSpec((1280, 920), 680), segmentation_config: dict | None = None,
               registration_config: dict | None = None, acquisition_interval_sec: float | None = None,
               area_jump_fraction=.20, config_path: Path | None = None) -> dict:
    """Export an independent bounded diagnostic window into a new directory."""
    video, output = Path(video).resolve(), Path(output).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if start < 0 or count < 1 or step < 1:
        raise ValueError("start >= 0, count >= 1 and step >= 1 are required")
    if acquisition_interval_sec is not None and (not np.isfinite(acquisition_interval_sec) or acquisition_interval_sec <= 0):
        raise ValueError("acquisition_interval_sec must be positive, or omitted when unknown")
    if not np.isfinite(area_jump_fraction) or area_jump_fraction <= 0:
        raise ValueError("area_jump_fraction must be positive and finite")
    registration_config = _registration_config(registration_config)
    segmentation_config = dict(segmentation_config or {})
    unknown = set(segmentation_config) - set(SEGMENT_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown segmentation parameters: {sorted(unknown)}")
    # Fail before creating output if the decoder cannot open the named source.
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video}")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except BaseException:
        cap.release()
        raise
    sources = [Path(__file__).resolve(), Path(__file__).with_name("segmenter.py"),
               Path(__file__).with_name("contracts.py"), Path(__file__).with_name("visualize.py"),
               _REGISTRATION_SOURCE]
    stat = video.stat()
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "source_video": str(video), "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns, "start_frame": int(start),
                "requested_count": int(count), "step": int(step), "method": "adaptive_chroma",
                "roi": asdict(roi), "segmentation_parameters": {**SEGMENT_DEFAULTS, **segmentation_config},
                "registration_parameters": registration_config, "area_jump_threshold": float(area_jump_fraction),
                "acquisition_interval_sec": acquisition_interval_sec,
                "biological_time_known": acquisition_interval_sec is not None,
                "playback_fps_metadata_only": _finite(cap.get(cv2.CAP_PROP_FPS)),
                "source_reported_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "temporal_mask_editing": False, "comparison_semantics": "diagnostic_rigid_alignment_only_not_biological_truth",
                "registered_comparison_support": "previous_effective_roi_intersect_warped_current_effective_roi_and_frame_support",
                "same_domain_comparison_support": "registered_common_fov_intersect_unwarped_current_effective_roi",
                "iou_comparison_note": "Compare same_domain_raw_iou with same_domain_registered_iou; global raw_iou and registered_iou have different support",
                "code_files": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in sources],
                "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__}}
    if config_path:
        manifest.update(config_path=str(Path(config_path).resolve()), config_sha256=_sha256(Path(config_path)))
    rows, thumbnails = [], []
    previous_image = previous_result = None
    error = None
    try:
        _json_write(output / "run_manifest.json", manifest)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(start)):
            raise ValueError(f"Could not seek required source frame {start}")
        for sample in range(int(count)):
            index = int(start + sample*step)
            ok, image = cap.read()
            if not ok or image is None:
                raise ValueError(f"Could not decode required source frame {index}")
            decoded_index = float(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1.0
            if not np.isfinite(decoded_index) or abs(decoded_index-index) > .5:
                raise ValueError(f"Decoder returned frame {decoded_index} instead of required source frame {index}")
            result = segment(image, roi, method="adaptive_chroma", config=segmentation_config)
            pair = None if previous_image is None else analyze_pair(
                previous_image, image, previous_result.mask, result.mask, roi,
                previous_roi_mask=previous_result.roi_mask, current_roi_mask=result.roi_mask,
                registration_config=registration_config, area_jump_fraction=area_jump_fraction,
            )
            folder = output / f"frame_{index:06d}"
            folder.mkdir()
            overlay = mask_overlay(image, result.mask)
            write_image(folder / "mask.png", result.mask.astype(np.uint8)*255)
            write_image(folder / "overlay.jpg", overlay)
            _save_frame_contact(folder / "joint_contact.jpg", image, result.mask, roi, index,
                                pair, previous_image, None if previous_result is None else previous_result.mask,
                                None if previous_result is None else previous_result.roi_mask, result.roi_mask)
            flags = list(result.qc_flags)
            if acquisition_interval_sec is None:
                flags.append("acquisition_interval_unknown")
            if pair is not None and not pair["registration_valid"]:
                flags.append("registration_untrusted_registered_metrics_unavailable")
            elif pair is not None and not pair["registered_comparison_valid"]:
                flags.append("no_common_valid_analysis_roi_registered_metrics_unavailable")
            if pair is not None and pair["area_jump_flag"]:
                flags.append("area_jump_review_segmentation_or_real_change")
            row = {"frame_index": index, "decoded_frame_index": decoded_index,
                   "previous_frame_index": index-step if sample else None,
                   "acquisition_interval_sec": acquisition_interval_sec,
                   "time_from_video_frame0_sec": index*acquisition_interval_sec if acquisition_interval_sec is not None else None,
                   "time_from_window_start_sec": sample*step*acquisition_interval_sec if acquisition_interval_sec is not None else None,
                   "sample_interval_sec": step*acquisition_interval_sec if acquisition_interval_sec is not None else None,
                   "occupied_area_px": int(result.mask.sum()), "area_fraction": result.diagnostics["area_fraction"],
                   "food_excluded_px": int(result.excluded_food.sum()),
                   "raw_frame_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                   "mask_sha256": hashlib.sha256(result.mask.tobytes()).hexdigest(),
                   "mask_path": f"{folder.name}/mask.png", "overlay_path": f"{folder.name}/overlay.jpg",
                   "contact_path": f"{folder.name}/joint_contact.jpg", "qc_path": f"{folder.name}/qc.json",
                   "qc_flags": json.dumps(flags, ensure_ascii=False),
                   **(pair if pair is not None else _first_pair_record())}
            _json_write(folder / "qc.json", {"frame": row, "segmentation": result.diagnostics,
                                              "qc_flags": flags, "pair": pair, "temporal_mask_editing": False})
            rows.append(row)
            thumb = _panel(overlay, roi, 360)
            labeled = Image.new("RGB", (360, 420), "#f4f7fb")
            labeled.paste(thumb, (0, 60))
            draw = ImageDraw.Draw(labeled)
            draw.text((8, 5), f"frame {index} | area {int(result.mask.sum())} px", font=_font(16), fill="#173149")
            raw = "n/a" if pair is None or pair["same_domain_raw_iou"] is None else f"{pair['same_domain_raw_iou']:.3f}"
            reg = "n/a" if pair is None or pair["same_domain_registered_iou"] is None else f"{pair['same_domain_registered_iou']:.3f}"
            draw.text((8, 30), f"Same-domain IoU raw {raw} / aligned {reg}", font=_font(14), fill="#173149")
            thumbnails.append(labeled)
            previous_image, previous_result = image, result
            if sample+1 < count:
                for skipped in range(step-1):
                    if not cap.grab():
                        raise ValueError(f"Could not advance through source frame {index+skipped+1}")
    except BaseException as exc:
        error = exc
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        cap.release()
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        if rows:
            with (output / "diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            columns = min(3, len(thumbnails))
            sheet = Image.new("RGB", (360*columns, 420*((len(thumbnails)+columns-1)//columns)), "white")
            for number, thumb in enumerate(thumbnails):
                sheet.paste(thumb, ((number % columns)*360, (number//columns)*420))
            sheet.save(output / "joint_contact.jpg", quality=93)
        changed = [record["path"] for record in manifest["code_files"]
                   if _sha256(Path(record["path"])) != record["sha256"]]
        manifest.update(completed_count=len(rows), finished_utc=datetime.now(timezone.utc).isoformat(),
                        code_files_changed_during_run=changed)
        if error is None:
            manifest["status"] = "complete" if not changed else "complete_source_changed_review_required"
        _json_write(output / "run_manifest.json", manifest)
    if error is not None:
        raise error
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--video", dest="video_option", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--center", type=float, nargs=2)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--acquisition-interval-sec", type=float, default=None)
    parser.add_argument("--area-jump-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    if (args.video is None) == (args.video_option is None):
        parser.error("Supply one source video, either positional or --video")
    raw, segmentation, qc = _load_config(args.config)
    geometry = raw.get("roi", {})
    roi = RoiSpec(tuple(args.center or geometry.get("center", (1280, 920))),
                  args.radius if args.radius is not None else float(geometry.get("radius", 680)),
                  args.margin if args.margin is not None else float(geometry.get("margin_px", 8)))
    manifest = run_window(
        args.video or args.video_option, args.output, start=args.start, count=args.count, step=args.step,
        roi=roi, segmentation_config=segmentation, registration_config=qc.get("registration"),
        acquisition_interval_sec=args.acquisition_interval_sec if args.acquisition_interval_sec is not None else qc.get("acquisition_interval_sec"),
        area_jump_fraction=args.area_jump_fraction if args.area_jump_fraction is not None else qc.get("area_jump_fraction", .20),
        config_path=args.config,
    )
    print(json.dumps({"status": manifest["status"], "completed_count": manifest["completed_count"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
