"""Full-video temporal-recovery candidate analysis with strict-mask controls.

This driver keeps the established analysis implementation in
plasmodium_brightfield_analysis.py unchanged. It adds symmetric temporal
context to bounded parallel batches, preserves strict HSV measurements, and
writes a separate temporal-candidate result set for formal validation.

The default status is deliberately not publication-ready. Promote the temporal
mask only after the independent frozen test workflow under
publication_validation_system passes its pre-specified criteria.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path
import gc
import hashlib
import json
import platform

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
from scipy import ndimage as ndi
from tqdm import tqdm

import plasmodium_brightfield_analysis as base


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

INPUT_PATH = Path(r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\output2.mp4")
OUTPUT_DIR = Path("results_plasmodium_temporal_formal_candidate")

FRAME_INTERVAL_S = 10.0
PIXEL_SIZE = 1.0
DISH_CENTER = (1280, 960)
DISH_RADIUS = 750
ROI = None

MAX_FRAMES = None
CPU_WORKERS = 8
CORE_BATCH_FRAMES = 16
OVERLAY_VIDEO_FPS = 60
SAVE_OVERLAY_VIDEO = True
SHOW_FIRST_FRAME_PREVIEW = True
CLEAN_DEBUG_OUTPUTS = True

COMPUTE_TEMPORAL_SKELETON = True
STRICT_SKELETON_EVERY_N_FRAMES = 5
VEIN_EVERY_N_FRAMES = 5
COMPUTE_FRONT_SPEED = True

FORMAL_STATUS = "candidate_pending_independent_validation"
PRIMARY_MASK = "occupied_temporal_candidate"

HSV_PARAMS = dict(
    h_min=15,
    h_max=49,
    s_min=54,
    v_min=30,
    min_obj_size=150,
    min_hole_size=0,
    morph_open_size=1,
    morph_close_size=0,
    keep_largest=True,
    fill_small_body_holes=False,
    use_yellow_chroma_extension=True,
    yellow_chroma_min=23,
    yellow_chroma_max_value=225,
    yellow_chroma_min_channel=55,
    yellow_chroma_rim_exclusion_px=20,
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

VEIN_PARAMS = dict(
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

TEMPORAL_PARAMS = dict(
    window=5,
    min_hits=3,
    chroma_min=9,
    min_channel=40,
    max_value=240,
    rim_exclusion_px=30,
    connect_radius=10,
    min_obj_size=15,
    exterior_only=False,
    block_hole_max_area=1500,
    final_smooth_window=5,
    final_smooth_min_hits=3,
)


REGION_KEYS = (
    "area_px",
    "area",
    "perimeter_px",
    "perimeter",
    "circularity",
    "solidity",
    "bbox_area_px",
    "eccentricity",
)
VEIN_KEYS = (
    "vein_area_px",
    "vein_length_px",
    "vein_length",
    "vein_endpoints",
    "vein_branchpoints",
)
DYNAMIC_KEYS = (
    "new_area_px",
    "retracted_area_px",
    "explore_ratio",
    "retreat_ratio",
    "net_growth_px",
    "front_speed_mean",
    "front_speed_p95",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _prefix(values: dict, prefix: str) -> dict:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _overlap_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = first.astype(bool)
    second = second.astype(bool)
    intersection = int(np.sum(first & second))
    union = int(np.sum(first | second))
    total = int(first.sum() + second.sum())
    return {
        "strict_temporal_iou": float(intersection / union) if union else 1.0,
        "strict_temporal_dice": float(2 * intersection / total) if total else 1.0,
    }


def _adjacent_iou(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return np.nan
    intersection = int(np.sum(previous.astype(bool) & current.astype(bool)))
    union = int(np.sum(previous.astype(bool) | current.astype(bool)))
    return float(intersection / union) if union else 1.0


def _numeric_vein(vein: dict | None) -> dict[str, float]:
    if vein is None:
        return {key: np.nan for key in VEIN_KEYS}
    return {key: vein.get(key, np.nan) for key in VEIN_KEYS}


def _analyze_mask_pair_worker(args):
    """Compute strict and temporal static metrics for one prepared core frame."""
    (
        frame_idx,
        frame,
        strict_mask,
        temporal_mask,
        candidate_mask,
        temporal_valid,
        pixel_size,
        dish_center,
        dish_radius,
        temporal_rim_width,
        compute_temporal_skeleton,
        strict_skeleton_every_n,
        vein_every_n,
        vein_params,
        roi,
    ) = args
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    strict_mask = strict_mask.astype(bool)
    temporal_mask = temporal_mask.astype(bool)
    candidate_mask = candidate_mask.astype(bool)
    temporal_added = temporal_mask & (~strict_mask)
    temporal_removed = strict_mask & (~temporal_mask)
    gray = base._as_gray_uint8(frame)

    strict_reg = base.region_metrics(strict_mask, pixel_size=pixel_size)
    temporal_reg = base.region_metrics(temporal_mask, pixel_size=pixel_size)
    strict_qc = base.mask_qc_metrics(strict_mask, analysis_roi=None)
    temporal_qc = base.mask_qc_metrics(temporal_mask, analysis_roi=None)

    strict_envelope = ndi.binary_fill_holes(strict_mask).astype(bool)
    temporal_envelope = ndi.binary_fill_holes(temporal_mask).astype(bool)
    strict_body = base.body_mask_layer_metrics(
        strict_mask,
        strict_mask,
        strict_envelope,
        pixel_size=pixel_size,
    )
    temporal_body = base.body_mask_layer_metrics(
        temporal_mask,
        temporal_mask,
        temporal_envelope,
        pixel_size=pixel_size,
    )
    strict_body["analysis_mask_type"] = "strict_occupied"
    temporal_body["analysis_mask_type"] = PRIMARY_MASK

    temporal_skeleton = (
        base.skeleton_metrics(temporal_mask, pixel_size=pixel_size)
        if compute_temporal_skeleton
        else base.empty_skeleton_metrics_like(temporal_mask)
    )
    strict_skeleton_sampled = (
        strict_skeleton_every_n > 0
        and frame_idx % strict_skeleton_every_n == 0
    )
    strict_skeleton = (
        base.skeleton_metrics(strict_mask, pixel_size=pixel_size)
        if strict_skeleton_sampled
        else base.empty_skeleton_metrics_like(strict_mask)
    )

    compute_vein = vein_every_n > 0 and frame_idx % vein_every_n == 0
    temporal_vein = None
    strict_vein = None
    if compute_vein:
        temporal_vein = base.vein_network_metrics(
            frame,
            dish_center=dish_center,
            dish_radius=dish_radius,
            body_mask=temporal_mask,
            pixel_size=pixel_size,
            **vein_params,
        )
        strict_vein = base.vein_network_metrics(
            frame,
            dish_center=dish_center,
            dish_radius=dish_radius,
            body_mask=strict_mask,
            pixel_size=pixel_size,
            **vein_params,
        )

    outer_dish = base.circular_roi_mask(strict_mask.shape, dish_center, dish_radius)
    inner_radius = max(1, int(dish_radius) - int(temporal_rim_width))
    inner_dish = base.circular_roi_mask(strict_mask.shape, dish_center, inner_radius)
    rim_band = outer_dish & (~inner_dish)
    rim_added_area = int(np.sum(temporal_added & rim_band))

    strict_area_px = int(strict_reg["area_px"])
    temporal_area_px = int(temporal_reg["area_px"])
    added_area = int(temporal_added.sum())
    removed_area = int(temporal_removed.sum())

    row = {
        "frame": int(frame_idx),
        "name": f"frame_{int(frame_idx):06d}",
        "time": float(frame_idx * FRAME_INTERVAL_S),
        "focus_score": base.focus_score_laplacian(gray),
        "temporal_valid": bool(temporal_valid),
        "temporal_candidate_area_px": int(candidate_mask.sum()),
        "temporal_added_area_px": added_area,
        "temporal_removed_area_px": removed_area,
        "temporal_added_fraction": float(added_area / max(temporal_area_px, 1)),
        "temporal_removed_fraction": float(removed_area / max(strict_area_px, 1)),
        "temporal_rim_added_area_px": rim_added_area,
        "temporal_rim_added_fraction": float(rim_added_area / max(added_area, 1)),
        "strict_skeleton_sampled": bool(strict_skeleton_sampled),
        **temporal_reg,
        **temporal_qc,
        **temporal_body,
        **_prefix(strict_reg, "strict_"),
        **_prefix(strict_qc, "strict_"),
        **_prefix(strict_body, "strict_"),
        **_overlap_metrics(strict_mask, temporal_mask),
        "n_endpoints": temporal_skeleton["n_endpoints"],
        "n_branchpoints": temporal_skeleton["n_branchpoints"],
        "skeleton_length_px": temporal_skeleton["skeleton_length_px"],
        "skeleton_length": temporal_skeleton["skeleton_length"],
        "strict_n_endpoints": strict_skeleton["n_endpoints"],
        "strict_n_branchpoints": strict_skeleton["n_branchpoints"],
        "strict_skeleton_length_px": strict_skeleton["skeleton_length_px"],
        "strict_skeleton_length": strict_skeleton["skeleton_length"],
        **_numeric_vein(temporal_vein),
        **_prefix(_numeric_vein(strict_vein), "strict_"),
    }
    if roi is not None:
        row["roi_mean_intensity"] = base.roi_mean_intensity(gray, roi)

    debug_item = {
        "frame": int(frame_idx),
        "name": f"frame_{int(frame_idx):06d}",
        "gray": gray,
        "enhanced": gray,
        "mask": temporal_mask,
        "occupied_mask": temporal_mask,
        "strict_mask": strict_mask,
        "temporal_mask": temporal_mask,
        "temporal_candidate_mask": candidate_mask,
        "temporal_added_mask": temporal_added,
        "temporal_removed_mask": temporal_removed,
        "envelope_mask": temporal_envelope,
        "skeleton": temporal_skeleton["skeleton"],
        "endpoint_mask": temporal_skeleton["endpoint_mask"],
        "branch_mask": temporal_skeleton["branch_mask"],
        "vein_mask": temporal_vein["vein_mask"] if temporal_vein is not None else None,
        "vein_skeleton": (
            temporal_vein["vein_skeleton"] if temporal_vein is not None else None
        ),
    }
    return int(frame_idx), row, debug_item, strict_mask, temporal_mask


def _normalized_temporal_params(params: dict) -> tuple[dict, int]:
    normalized = dict(params)
    normalized["window"] = max(3, int(normalized["window"]))
    if normalized["window"] % 2 == 0:
        normalized["window"] += 1
    normalized["min_hits"] = max(
        1, min(int(normalized["min_hits"]), normalized["window"])
    )
    normalized["final_smooth_window"] = max(
        1, int(normalized["final_smooth_window"])
    )
    if normalized["final_smooth_window"] > 1 and normalized["final_smooth_window"] % 2 == 0:
        normalized["final_smooth_window"] += 1
    normalized["final_smooth_min_hits"] = max(
        1,
        min(
            int(normalized["final_smooth_min_hits"]),
            normalized["final_smooth_window"],
        ),
    )
    halo = normalized["window"] // 2 + normalized["final_smooth_window"] // 2
    return normalized, halo


def _read_frame_range(
    cap: cv2.VideoCapture,
    first_frame: int,
    last_frame: int,
) -> list[tuple[int, np.ndarray]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame))
    frames = []
    for frame_idx in range(first_frame, last_frame + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_idx}")
        frames.append((frame_idx, frame))
    return frames


def _dynamic_values(
    previous_mask: np.ndarray | None,
    current_mask: np.ndarray,
    previous_area: float,
    current_area: float,
    dt: float,
    pixel_size: float,
    compute_front_speed: bool,
) -> tuple[dict, float]:
    dynamics = base.expansion_retraction_metrics(
        previous_mask,
        current_mask,
        dt=dt,
        pixel_size=pixel_size,
        compute_front_speed=compute_front_speed,
    )
    if previous_mask is None or not np.isfinite(previous_area) or dt <= 0:
        dA_dt = np.nan
    else:
        dA_dt = float((current_area - previous_area) / dt)
    return dynamics, dA_dt


def _invalidate_temporal_dynamics(row: dict) -> None:
    row["dA_dt"] = np.nan
    for key in DYNAMIC_KEYS:
        row[key] = np.nan


def analyze_video_temporal_candidate(
    video_path: Path,
    output_overlay_path: Path | None,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Run bounded full-resolution temporal analysis with strict controls."""
    temporal_params, temporal_halo = _normalized_temporal_params(TEMPORAL_PARAMS)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_total <= 0:
        cap.release()
        raise ValueError(f"Video frame count is unavailable: {video_path}")
    total_frames = (
        min(source_total, int(MAX_FRAMES)) if MAX_FRAMES is not None else source_total
    )
    if total_frames <= 2 * temporal_halo:
        cap.release()
        raise ValueError(
            f"Need more than {2 * temporal_halo} frames for symmetric temporal context"
        )

    debug_indices = sorted(
        {
            temporal_halo,
            total_frames // 4,
            total_frames // 2,
            3 * total_frames // 4,
            total_frames - temporal_halo - 1,
        }
    )
    debug_index_set = set(debug_indices)

    writer = None
    rows: list[dict] = []
    debug_data: list[dict] = []
    previous_strict = None
    previous_temporal = None
    previous_strict_area = np.nan
    previous_temporal_area = np.nan
    previous_temporal_valid = False
    core_start = 0
    peak_prepared_frames = 0

    try:
        executor_context = (
            ProcessPoolExecutor(max_workers=CPU_WORKERS)
            if CPU_WORKERS > 1
            else nullcontext(None)
        )
        with executor_context as executor:
            with tqdm(
                total=total_frames,
                desc=f"Temporal formal candidate with {CPU_WORKERS} CPU workers",
            ) as progress:
                while core_start < total_frames:
                    core_end = min(
                        total_frames - 1,
                        core_start + int(CORE_BATCH_FRAMES) - 1,
                    )
                    read_start = max(0, core_start - temporal_halo)
                    read_end = min(total_frames - 1, core_end + temporal_halo)
                    source_frames = _read_frame_range(cap, read_start, read_end)
                    peak_prepared_frames = max(peak_prepared_frames, len(source_frames))

                    prepare_tasks = [
                        (
                            frame_idx,
                            frame,
                            DISH_CENTER,
                            DISH_RADIUS,
                            HSV_PARAMS,
                            temporal_params["chroma_min"],
                            temporal_params["min_channel"],
                            temporal_params["max_value"],
                            temporal_params["rim_exclusion_px"],
                            temporal_params["min_obj_size"],
                        )
                        for frame_idx, frame in source_frames
                    ]
                    if executor is None:
                        prepared = [
                            base._prepare_temporal_validation_frame_for_parallel(task)
                            for task in prepare_tasks
                        ]
                    else:
                        prepared = list(
                            executor.map(
                                base._prepare_temporal_validation_frame_for_parallel,
                                prepare_tasks,
                            )
                        )
                    prepared.sort(key=lambda item: int(item["frame_idx"]))
                    recovered = base.apply_temporal_body_recovery_to_sequence(
                        prepared,
                        window=temporal_params["window"],
                        min_hits=temporal_params["min_hits"],
                        connect_radius=temporal_params["connect_radius"],
                        exterior_only=temporal_params["exterior_only"],
                        block_hole_max_area=temporal_params["block_hole_max_area"],
                        final_smooth_window=temporal_params["final_smooth_window"],
                        final_smooth_min_hits=temporal_params[
                            "final_smooth_min_hits"
                        ],
                    )
                    recovered_by_frame = {
                        int(item["frame_idx"]): item for item in recovered
                    }

                    metric_tasks = []
                    for frame_idx in range(core_start, core_end + 1):
                        item = recovered_by_frame[frame_idx]
                        strict_mask = item["strict_mask"].astype(bool)
                        temporal_valid = (
                            temporal_halo <= frame_idx < total_frames - temporal_halo
                        )
                        temporal_mask = item["mask"].astype(bool)
                        if not temporal_valid:
                            temporal_mask = strict_mask.copy()
                        metric_tasks.append(
                            (
                                frame_idx,
                                item["frame"],
                                strict_mask,
                                temporal_mask,
                                item["candidate"],
                                temporal_valid,
                                PIXEL_SIZE,
                                DISH_CENTER,
                                DISH_RADIUS,
                                temporal_params["rim_exclusion_px"],
                                COMPUTE_TEMPORAL_SKELETON,
                                STRICT_SKELETON_EVERY_N_FRAMES,
                                VEIN_EVERY_N_FRAMES,
                                VEIN_PARAMS,
                                ROI,
                            )
                        )

                    if executor is None:
                        frame_results = [
                            _analyze_mask_pair_worker(task) for task in metric_tasks
                        ]
                    else:
                        frame_results = list(
                            executor.map(_analyze_mask_pair_worker, metric_tasks)
                        )
                    frame_results.sort(key=lambda result: result[0])

                    for frame_idx, row, debug_item, strict_mask, temporal_mask in frame_results:
                        strict_dynamics, strict_dA_dt = _dynamic_values(
                            previous_strict,
                            strict_mask,
                            previous_strict_area,
                            float(row["strict_area"]),
                            FRAME_INTERVAL_S,
                            PIXEL_SIZE,
                            COMPUTE_FRONT_SPEED,
                        )
                        temporal_dynamics, temporal_dA_dt = _dynamic_values(
                            previous_temporal,
                            temporal_mask,
                            previous_temporal_area,
                            float(row["area"]),
                            FRAME_INTERVAL_S,
                            PIXEL_SIZE,
                            COMPUTE_FRONT_SPEED,
                        )
                        row["strict_dA_dt"] = strict_dA_dt
                        row.update(
                            {
                                f"strict_{key}": strict_dynamics[key]
                                for key in DYNAMIC_KEYS
                            }
                        )
                        row["dA_dt"] = temporal_dA_dt
                        row.update(
                            {key: temporal_dynamics[key] for key in DYNAMIC_KEYS}
                        )
                        row["strict_adjacent_iou"] = _adjacent_iou(
                            previous_strict, strict_mask
                        )
                        row["temporal_adjacent_iou"] = _adjacent_iou(
                            previous_temporal, temporal_mask
                        )
                        row["temporal_dynamic_valid"] = bool(
                            row["temporal_valid"] and previous_temporal_valid
                        )
                        if not row["temporal_dynamic_valid"]:
                            _invalidate_temporal_dynamics(row)
                            temporal_dynamics["new_region"] = np.zeros_like(
                                temporal_mask, dtype=bool
                            )
                            temporal_dynamics["retracted_region"] = np.zeros_like(
                                temporal_mask, dtype=bool
                            )

                        debug_item["new_region"] = temporal_dynamics["new_region"]
                        debug_item["retracted_region"] = temporal_dynamics[
                            "retracted_region"
                        ]

                        if output_overlay_path is not None:
                            overlay_rgb = base.overlay_mask_skeleton(
                                debug_item["gray"],
                                mask=temporal_mask,
                                skeleton=debug_item["skeleton"],
                                endpoint_mask=debug_item["endpoint_mask"],
                                branch_mask=debug_item["branch_mask"],
                                new_region=debug_item["new_region"],
                                retracted_region=debug_item["retracted_region"],
                                vein_mask=debug_item.get("vein_mask"),
                                vein_skeleton=debug_item.get("vein_skeleton"),
                            )
                            overlay_bgr = cv2.cvtColor(
                                overlay_rgb, cv2.COLOR_RGB2BGR
                            )
                            cv2.putText(
                                overlay_bgr,
                                f"temporal candidate | frame {frame_idx}/{total_frames - 1}",
                                (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            if not row["temporal_valid"]:
                                cv2.putText(
                                    overlay_bgr,
                                    "edge context unavailable: strict mask shown",
                                    (30, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_AA,
                                )
                            if writer is None:
                                output_overlay_path.parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                height, width = overlay_bgr.shape[:2]
                                writer = cv2.VideoWriter(
                                    str(output_overlay_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    float(OVERLAY_VIDEO_FPS),
                                    (width, height),
                                    True,
                                )
                                if not writer.isOpened():
                                    raise ValueError(
                                        f"Could not open overlay writer: {output_overlay_path}"
                                    )
                            writer.write(overlay_bgr)

                        if frame_idx in debug_index_set:
                            debug_data.append(debug_item)
                        rows.append(row)
                        previous_strict = strict_mask
                        previous_temporal = temporal_mask
                        previous_strict_area = float(row["strict_area"])
                        previous_temporal_area = float(row["area"])
                        previous_temporal_valid = bool(row["temporal_valid"])

                    processed = core_end - core_start + 1
                    progress.update(processed)
                    core_start = core_end + 1
                    del (
                        source_frames,
                        prepare_tasks,
                        prepared,
                        recovered,
                        recovered_by_frame,
                        metric_tasks,
                        frame_results,
                    )
                    gc.collect()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not rows:
        raise ValueError("No video frames were analyzed")

    df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    df = base.apply_vein_confidence(df)
    df = base.apply_body_mask_confidence(df)
    df = base.add_body_area_derivatives(df, FRAME_INTERVAL_S)
    if ROI is not None and "roi_mean_intensity" in df.columns:
        oscillation = base.analyze_oscillation(
            df["roi_mean_intensity"].to_numpy(), FRAME_INTERVAL_S
        )
    else:
        oscillation = {
            "dominant_freq": np.nan,
            "period": np.nan,
            "n_peaks": 0,
            "detrended_signal": np.array([]),
            "fft_freq": np.array([]),
            "fft_power": np.array([]),
        }
    run_info = {
        "source_total_frames": source_total,
        "processed_frames": len(df),
        "temporal_halo_frames": temporal_halo,
        "peak_prepared_frames": peak_prepared_frames,
        "debug_frame_indices": debug_indices,
        "normalized_temporal_params": temporal_params,
    }
    return df, debug_data, {"oscillation": oscillation, "run_info": run_info}


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _stat(series: pd.Series, kind: str) -> float:
    values = _finite(series)
    if values.size == 0:
        return np.nan
    if kind == "mean":
        return float(np.mean(values))
    if kind == "median":
        return float(np.median(values))
    if kind == "p05":
        return float(np.percentile(values, 5))
    if kind == "p95":
        return float(np.percentile(values, 95))
    if kind == "min":
        return float(np.min(values))
    raise ValueError(f"Unknown statistic: {kind}")


def comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    n_frames = max(int(work["frame"].max()) + 1, 1)
    fraction = work["frame"].astype(float) / max(n_frames - 1, 1)
    work["section"] = np.select(
        [fraction < 1 / 3, fraction < 2 / 3],
        ["front", "middle"],
        default="rear",
    )
    valid = work[work["temporal_valid"].astype(bool)].copy()
    sections = [("overall", valid)] + [
        (section, valid[valid["section"] == section])
        for section in ("front", "middle", "rear")
    ]
    records = []
    for section, subset in sections:
        if subset.empty:
            continue
        area_difference = (
            subset["area_px"].astype(float) - subset["strict_area_px"].astype(float)
        ) / subset["strict_area_px"].replace(0, np.nan).astype(float)
        sampled_skeleton = subset[subset["strict_skeleton_sampled"].astype(bool)]
        if sampled_skeleton.empty:
            skeleton_difference_mean = np.nan
        else:
            skeleton_difference = (
                sampled_skeleton["skeleton_length_px"].astype(float)
                - sampled_skeleton["strict_skeleton_length_px"].astype(float)
            ) / sampled_skeleton["strict_skeleton_length_px"].replace(0, np.nan).astype(float)
            skeleton_difference_mean = _stat(skeleton_difference, "mean")
        records.append(
            {
                "section": section,
                "n_valid_frames": int(len(subset)),
                "strict_area_px_mean": _stat(subset["strict_area_px"], "mean"),
                "temporal_area_px_mean": _stat(subset["area_px"], "mean"),
                "temporal_area_difference_fraction_mean": _stat(
                    area_difference, "mean"
                ),
                "temporal_area_difference_fraction_p95": _stat(
                    area_difference.abs(), "p95"
                ),
                "temporal_added_fraction_mean": _stat(
                    subset["temporal_added_fraction"], "mean"
                ),
                "temporal_added_fraction_p95": _stat(
                    subset["temporal_added_fraction"], "p95"
                ),
                "temporal_removed_fraction_mean": _stat(
                    subset["temporal_removed_fraction"], "mean"
                ),
                "temporal_removed_fraction_p95": _stat(
                    subset["temporal_removed_fraction"], "p95"
                ),
                "temporal_rim_added_fraction_p95": _stat(
                    subset["temporal_rim_added_fraction"], "p95"
                ),
                "strict_temporal_dice_mean": _stat(
                    subset["strict_temporal_dice"], "mean"
                ),
                "strict_temporal_dice_min": _stat(
                    subset["strict_temporal_dice"], "min"
                ),
                "temporal_adjacent_iou_mean": _stat(
                    subset["temporal_adjacent_iou"], "mean"
                ),
                "temporal_adjacent_iou_p05": _stat(
                    subset["temporal_adjacent_iou"], "p05"
                ),
                "strict_adjacent_iou_mean": _stat(
                    subset["strict_adjacent_iou"], "mean"
                ),
                "skeleton_length_difference_fraction_mean_sampled": (
                    skeleton_difference_mean
                ),
            }
        )
    return pd.DataFrame(records)


def technical_readiness(summary: pd.DataFrame, df: pd.DataFrame) -> dict:
    overall = summary.loc[summary["section"] == "overall"].iloc[0]
    valid_fraction = float(np.mean(df["temporal_valid"].astype(bool)))
    checks = {
        "temporal_valid_fraction_ge_0_99": valid_fraction >= 0.99,
        "strict_temporal_dice_mean_ge_0_90": (
            float(overall["strict_temporal_dice_mean"]) >= 0.90
        ),
        "temporal_adjacent_iou_p05_ge_0_95": (
            float(overall["temporal_adjacent_iou_p05"]) >= 0.95
        ),
        "temporal_added_fraction_p95_le_0_10": (
            float(overall["temporal_added_fraction_p95"]) <= 0.10
        ),
        "temporal_removed_fraction_p95_le_0_10": (
            float(overall["temporal_removed_fraction_p95"]) <= 0.10
        ),
        "temporal_rim_added_fraction_p95_le_0_001": (
            float(overall["temporal_rim_added_fraction_p95"]) <= 0.001
        ),
    }
    return {
        "formal_status": FORMAL_STATUS,
        "primary_mask": PRIMARY_MASK,
        "technical_trial_checks_passed": bool(all(checks.values())),
        "technical_trial_checks": checks,
        "publication_ready": False,
        "publication_blocker": (
            "Independent frozen test-set validation has not yet established "
            "accuracy for this temporal primary mask."
        ),
        "important": (
            "Technical stability checks do not replace Dice, precision, recall, "
            "boundary, topology, and human-agreement validation."
        ),
    }


def save_comparison_plot(df: pd.DataFrame, path: Path) -> None:
    t = df["time"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(t, df["strict_area"], label="strict", color="0.5")
    axes[0].plot(t, df["area"], label="temporal", color="tab:green")
    axes[0].set_title("Area: strict vs temporal")
    axes[0].legend()

    axes[1].plot(t, df["strict_dA_dt"], label="strict", color="0.5")
    axes[1].plot(t, df["dA_dt"], label="temporal", color="tab:blue")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("dA/dt")
    axes[1].legend()

    axes[2].plot(t, df["strict_explore_ratio"], label="strict", color="0.5")
    axes[2].plot(t, df["explore_ratio"], label="temporal", color="tab:purple")
    axes[2].set_title("Explore ratio")
    axes[2].legend()

    sampled = df[df["strict_skeleton_sampled"].astype(bool)]
    axes[3].plot(
        sampled["time"],
        sampled["strict_skeleton_length"],
        marker="o",
        markersize=2,
        label="strict sampled",
        color="0.5",
    )
    axes[3].plot(t, df["skeleton_length"], label="temporal", color="crimson")
    axes[3].set_title("Skeleton length")
    axes[3].legend()

    axes[4].plot(t, df["temporal_added_fraction"], label="added", color="teal")
    axes[4].plot(t, df["temporal_removed_fraction"], label="removed", color="red")
    axes[4].set_title("Temporal change relative to strict")
    axes[4].legend()

    axes[5].plot(t, df["strict_temporal_dice"], label="strict-temporal Dice")
    axes[5].plot(t, df["temporal_adjacent_iou"], label="temporal adjacent IoU")
    axes[5].set_ylim(0, 1.02)
    axes[5].set_title("Agreement and temporal stability")
    axes[5].legend()
    for axis in axes:
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_temporal_debug(debug_data: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in debug_data:
        stem = f"frame_{int(item['frame']):06d}"
        gray = base._as_gray_uint8(item["gray"])
        strict_mask = item["strict_mask"].astype(bool)
        temporal_mask = item["temporal_mask"].astype(bool)
        temporal_only = temporal_mask & (~strict_mask)
        strict_only = strict_mask & (~temporal_mask)
        common = strict_mask & temporal_mask

        comparison = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

        def blend(mask, color, alpha):
            if np.any(mask):
                comparison[mask] = (
                    (1.0 - alpha) * comparison[mask]
                    + alpha * np.asarray(color, dtype=np.float32)
                )

        blend(common, (0, 200, 0), 0.55)
        blend(temporal_only, (255, 255, 0), 0.75)
        blend(strict_only, (0, 0, 255), 0.75)
        comparison = np.clip(comparison, 0, 255).astype(np.uint8)
        cv2.putText(
            comparison,
            "green=common cyan=temporal-only red=strict-only",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(out_dir / f"{stem}_gray.png"), gray)
        cv2.imwrite(
            str(out_dir / f"{stem}_strict_mask.png"),
            strict_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(out_dir / f"{stem}_temporal_mask.png"),
            temporal_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(out_dir / f"{stem}_temporal_candidate_mask.png"),
            item["temporal_candidate_mask"].astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(out_dir / f"{stem}_temporal_added_mask.png"),
            item["temporal_added_mask"].astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(out_dir / f"{stem}_temporal_removed_mask.png"),
            item["temporal_removed_mask"].astype(np.uint8) * 255,
        )
        cv2.imwrite(str(out_dir / f"{stem}_strict_temporal_compare.png"), comparison)


def configuration_snapshot(run_info: dict) -> dict:
    source_path = Path(base.__file__).resolve()
    this_path = Path(__file__).resolve()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_status": FORMAL_STATUS,
        "primary_mask": PRIMARY_MASK,
        "input_path": str(INPUT_PATH.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "frame_interval_s": FRAME_INTERVAL_S,
        "pixel_size": PIXEL_SIZE,
        "dish_center": DISH_CENTER,
        "dish_radius": DISH_RADIUS,
        "max_frames": MAX_FRAMES,
        "cpu_workers": CPU_WORKERS,
        "core_batch_frames": CORE_BATCH_FRAMES,
        "overlay_video_fps": OVERLAY_VIDEO_FPS,
        "vein_every_n_frames": VEIN_EVERY_N_FRAMES,
        "strict_skeleton_every_n_frames": STRICT_SKELETON_EVERY_N_FRAMES,
        "hsv_params": HSV_PARAMS,
        "temporal_params": TEMPORAL_PARAMS,
        "vein_params": VEIN_PARAMS,
        "base_source_file": str(source_path),
        "base_source_sha256": _sha256(source_path),
        "driver_file": str(this_path),
        "driver_sha256": _sha256(this_path),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
        },
        "run_info": run_info,
        "warning": (
            "This output is a temporal formal candidate, not publication-ready "
            "evidence until the independent frozen test workflow passes."
        ),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"INPUT_PATH does not exist: {INPUT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CLEAN_DEBUG_OUTPUTS:
        base.clean_debug_outputs(OUTPUT_DIR, clean_segment_tests=False)

    print("Temporal formal candidate configuration:")
    print(f"  INPUT_PATH = {INPUT_PATH}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"  MAX_FRAMES = {MAX_FRAMES}")
    print(f"  CPU_WORKERS = {CPU_WORKERS}")
    print(f"  CORE_BATCH_FRAMES = {CORE_BATCH_FRAMES}")
    print(f"  SAVE_OVERLAY_VIDEO = {SAVE_OVERLAY_VIDEO}")
    print(f"  OVERLAY_VIDEO_FPS = {OVERLAY_VIDEO_FPS}")
    print(f"  PRIMARY_MASK = {PRIMARY_MASK}")
    print(f"  FORMAL_STATUS = {FORMAL_STATUS}")
    print("  Strict HSV metrics will be retained as control columns.")

    if SHOW_FIRST_FRAME_PREVIEW:
        first_frame, total_hint = base._load_first_frame_for_preview(
            INPUT_PATH, "video", as_gray=False
        )
        print(f"Video contains {total_hint} frames. Showing strict first-frame preview.")
        base.preview_first_frame(
            first_frame,
            segmentation_profile="hsv_circle",
            analysis_roi=None,
            dish_center=DISH_CENTER,
            dish_radius=DISH_RADIUS,
            hsv_params=HSV_PARAMS,
            preprocess_params={},
            segment_params={},
        )

    overlay_path = (
        OUTPUT_DIR / "plasmodium_overlay_temporal_candidate.mp4"
        if SAVE_OVERLAY_VIDEO
        else None
    )
    df, debug_data, details = analyze_video_temporal_candidate(
        INPUT_PATH,
        overlay_path,
    )
    print(f"Analyzed {len(df)} frames.")
    if overlay_path is not None:
        print(f"Saved temporal overlay video: {overlay_path}")

    metrics_path = OUTPUT_DIR / "plasmodium_metrics_temporal_candidate.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved dual strict/temporal metrics: {metrics_path}")

    comparison = comparison_summary(df)
    comparison_path = OUTPUT_DIR / "strict_temporal_comparison_summary.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"Saved strict/temporal comparison: {comparison_path}")

    readiness = technical_readiness(comparison, df)
    readiness_path = OUTPUT_DIR / "temporal_technical_readiness.json"
    _write_json(readiness_path, readiness)
    print(f"Saved technical readiness checks: {readiness_path}")

    comparison_plot = OUTPUT_DIR / "strict_temporal_comparison.png"
    save_comparison_plot(df, comparison_plot)
    print(f"Saved strict/temporal plot: {comparison_plot}")

    temporal_summary_plot = OUTPUT_DIR / "plasmodium_summary_temporal_candidate.png"
    base.plot_summary(df, out_prefix=temporal_summary_plot)
    plt.close("all")
    print(f"Saved temporal summary plot: {temporal_summary_plot}")

    debug_dir = OUTPUT_DIR / "debug_examples"
    save_temporal_debug(debug_data, debug_dir)
    print(f"Saved temporal debug masks: {debug_dir}")

    base.validate_analysis_qc(
        df,
        debug_data,
        out_dir=OUTPUT_DIR,
        analysis_roi=None,
        area_jump_threshold=0.35,
        save_review_images=False,
    )

    summary_df = base.summarize_run(
        df,
        label=f"{INPUT_PATH.stem}_temporal_candidate",
    )
    oscillation = details["oscillation"]
    summary_df["osc_dominant_freq"] = oscillation.get("dominant_freq", np.nan)
    summary_df["osc_period"] = oscillation.get("period", np.nan)
    summary_df["osc_n_peaks"] = oscillation.get("n_peaks", 0)
    summary_df["formal_status"] = FORMAL_STATUS
    summary_df["technical_trial_checks_passed"] = readiness[
        "technical_trial_checks_passed"
    ]
    summary_path = OUTPUT_DIR / "plasmodium_summary_table_temporal_candidate.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved temporal summary table: {summary_path}")

    config_path = OUTPUT_DIR / "temporal_run_configuration.json"
    _write_json(config_path, configuration_snapshot(details["run_info"]))
    print(f"Saved reproducibility configuration: {config_path}")
    print(summary_df.to_string(index=False))
    print(
        "Temporal candidate run completed. Publication status remains blocked "
        "until independent frozen test validation passes."
    )


if __name__ == "__main__":
    freeze_support()
    main()
