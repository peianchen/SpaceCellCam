"""Development-set check for strict and temporal plasmodium masks.

This script uses the existing 30 manually reviewed masks to tune and regress
the temporal candidate. These masks were used during development, so the
results must not be reported as independent publication test performance.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import freeze_support
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

import plasmodium_brightfield_analysis as base
import plasmodium_brightfield_analysis_temporal_formal as candidate


VIDEO_PATH = candidate.INPUT_PATH
PACKAGE_DIR = Path("results_plasmodium/manual_validation_package")
MANIFEST_PATH = PACKAGE_DIR / "manual_validation_manifest.csv"
MANUAL_MASK_DIR = PACKAGE_DIR / "manual_masks"
OUTPUT_DIR = Path("results_plasmodium_temporal_manual_development_check")

CPU_WORKERS = 8
SAVE_REVIEW_IMAGES = True

# Development gates for deciding whether a parameter set is worth a full run.
# These do not replace the frozen publication-validation acceptance criteria.
MIN_SECTION_DICE_MEAN = 0.90
MIN_OVERALL_PRECISION_MEAN = 0.90
MIN_OVERALL_RECALL_MEAN = 0.90
MAX_OVERALL_ABS_AREA_ERROR_MEAN = 0.10
MAX_OVERALL_DICE_DROP_VS_STRICT = 0.01


def _binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask > 0


def _segmentation_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    predicted = predicted.astype(bool)
    reference = reference.astype(bool)
    tp = int(np.sum(predicted & reference))
    fp = int(np.sum(predicted & (~reference)))
    fn = int(np.sum((~predicted) & reference))
    predicted_area = int(predicted.sum())
    reference_area = int(reference.sum())
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    dice_denominator = 2 * tp + fp + fn
    dice = 2 * tp / dice_denominator if dice_denominator else 1.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    signed_area_error = (predicted_area - reference_area) / max(reference_area, 1)
    return {
        "predicted_area_px": predicted_area,
        "reference_area_px": reference_area,
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "signed_area_error": float(signed_area_error),
        "absolute_area_error": float(abs(signed_area_error)),
        "false_positive_area_px": fp,
        "false_negative_area_px": fn,
    }


def _read_frames(
    cap: cv2.VideoCapture,
    first_frame: int,
    last_frame: int,
) -> list[tuple[int, np.ndarray]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame))
    frames = []
    for frame_idx in range(first_frame, last_frame + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_idx} from {VIDEO_PATH}")
        frames.append((frame_idx, frame))
    return frames


def _prepare_tasks(frames: list[tuple[int, np.ndarray]]) -> list[tuple]:
    params = candidate.TEMPORAL_PARAMS
    return [
        (
            frame_idx,
            frame,
            candidate.DISH_CENTER,
            candidate.DISH_RADIUS,
            candidate.HSV_PARAMS,
            params["chroma_min"],
            params["min_channel"],
            params["max_value"],
            params["rim_exclusion_px"],
            params["min_obj_size"],
        )
        for frame_idx, frame in frames
    ]


def _predict_target(
    cap: cv2.VideoCapture,
    executor: ProcessPoolExecutor | None,
    frame_idx: int,
    source_total: int,
    temporal_params: dict,
    temporal_halo: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    temporal_valid = temporal_halo <= frame_idx < source_total - temporal_halo
    if temporal_valid:
        frames = _read_frames(
            cap,
            frame_idx - temporal_halo,
            frame_idx + temporal_halo,
        )
    else:
        frames = _read_frames(cap, frame_idx, frame_idx)

    tasks = _prepare_tasks(frames)
    if executor is None:
        prepared = [
            base._prepare_temporal_validation_frame_for_parallel(task)
            for task in tasks
        ]
    else:
        prepared = list(
            executor.map(base._prepare_temporal_validation_frame_for_parallel, tasks)
        )
    prepared.sort(key=lambda item: int(item["frame_idx"]))

    if temporal_valid:
        recovered = base.apply_temporal_body_recovery_to_sequence(
            prepared,
            window=temporal_params["window"],
            min_hits=temporal_params["min_hits"],
            connect_radius=temporal_params["connect_radius"],
            exterior_only=temporal_params["exterior_only"],
            block_hole_max_area=temporal_params["block_hole_max_area"],
            final_smooth_window=temporal_params["final_smooth_window"],
            final_smooth_min_hits=temporal_params["final_smooth_min_hits"],
        )
        item = next(x for x in recovered if int(x["frame_idx"]) == frame_idx)
        strict_mask = item["strict_mask"].astype(bool)
        temporal_mask = item["mask"].astype(bool)
        temporal_candidate = item["candidate"].astype(bool)
        frame = item["frame"]
    else:
        item = prepared[0]
        strict_mask = item["mask"].astype(bool)
        temporal_mask = strict_mask.copy()
        temporal_candidate = item["candidate"].astype(bool)
        frame = item["frame"]
    return frame, strict_mask, temporal_mask, temporal_candidate, temporal_valid


def _outline(mask: np.ndarray) -> np.ndarray:
    return mask & (~ndi.binary_erosion(mask, iterations=2))


def _error_overlay(
    frame_bgr: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    title: str,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    tp = predicted & reference
    fp = predicted & (~reference)
    fn = (~predicted) & reference
    tint = np.zeros_like(canvas)
    tint[tp] = (40, 180, 40)
    tint[fp] = (30, 30, 255)
    tint[fn] = (255, 160, 0)
    occupied = tp | fp | fn
    canvas[occupied] = cv2.addWeighted(canvas, 0.45, tint, 0.55, 0)[occupied]
    canvas[_outline(reference)] = (255, 255, 255)
    cv2.putText(
        canvas,
        title,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green=TP  red=FP  blue=FN  white=manual boundary",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _save_review(
    frame_idx: int,
    frame_bgr: np.ndarray,
    manual_mask: np.ndarray,
    strict_mask: np.ndarray,
    temporal_mask: np.ndarray,
    strict_metrics: dict,
    temporal_metrics: dict,
) -> None:
    review_dir = OUTPUT_DIR / "review_images"
    review_dir.mkdir(parents=True, exist_ok=True)
    strict = _error_overlay(
        frame_bgr,
        strict_mask,
        manual_mask,
        f"strict  Dice={strict_metrics['dice']:.3f}  P={strict_metrics['precision']:.3f}  R={strict_metrics['recall']:.3f}",
    )
    temporal = _error_overlay(
        frame_bgr,
        temporal_mask,
        manual_mask,
        f"temporal  Dice={temporal_metrics['dice']:.3f}  P={temporal_metrics['precision']:.3f}  R={temporal_metrics['recall']:.3f}",
    )
    comparison = np.concatenate([strict, temporal], axis=1)
    cv2.imwrite(
        str(review_dir / f"frame_{frame_idx:06d}_strict_vs_temporal.jpg"),
        comparison,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    valid_temporal = metrics[metrics["temporal_valid"].astype(bool)]
    for section in ("front", "middle", "rear", "overall"):
        subset = valid_temporal if section == "overall" else valid_temporal[
            valid_temporal["section"] == section
        ]
        if subset.empty:
            continue
        record = {"section": section, "n_temporal_valid": int(len(subset))}
        for method in ("strict", "temporal"):
            for metric in (
                "iou",
                "dice",
                "precision",
                "recall",
                "absolute_area_error",
            ):
                values = subset[f"{method}_{metric}"].to_numpy(dtype=float)
                record[f"{method}_{metric}_mean"] = float(np.mean(values))
                record[f"{method}_{metric}_min"] = float(np.min(values))
            record[f"{method}_dice_p05"] = float(
                np.percentile(subset[f"{method}_dice"].to_numpy(dtype=float), 5)
            )
        record["temporal_minus_strict_dice_mean"] = (
            record["temporal_dice_mean"] - record["strict_dice_mean"]
        )
        record["temporal_minus_strict_recall_mean"] = (
            record["temporal_recall_mean"] - record["strict_recall_mean"]
        )
        records.append(record)
    return pd.DataFrame(records)


def _development_decision(summary: pd.DataFrame) -> dict:
    overall = summary.loc[summary["section"] == "overall"].iloc[0]
    section_dice = summary.loc[
        summary["section"].isin(["front", "middle", "rear"]),
        "temporal_dice_mean",
    ]
    checks = {
        "all_section_temporal_dice_mean_ge_threshold": bool(
            (section_dice >= MIN_SECTION_DICE_MEAN).all()
        ),
        "overall_temporal_precision_mean_ge_threshold": bool(
            overall["temporal_precision_mean"] >= MIN_OVERALL_PRECISION_MEAN
        ),
        "overall_temporal_recall_mean_ge_threshold": bool(
            overall["temporal_recall_mean"] >= MIN_OVERALL_RECALL_MEAN
        ),
        "overall_temporal_abs_area_error_mean_le_threshold": bool(
            overall["temporal_absolute_area_error_mean"]
            <= MAX_OVERALL_ABS_AREA_ERROR_MEAN
        ),
        "overall_temporal_dice_drop_vs_strict_le_threshold": bool(
            overall["temporal_minus_strict_dice_mean"]
            >= -MAX_OVERALL_DICE_DROP_VS_STRICT
        ),
    }
    return {
        "development_gate_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "min_section_dice_mean": MIN_SECTION_DICE_MEAN,
            "min_overall_precision_mean": MIN_OVERALL_PRECISION_MEAN,
            "min_overall_recall_mean": MIN_OVERALL_RECALL_MEAN,
            "max_overall_abs_area_error_mean": MAX_OVERALL_ABS_AREA_ERROR_MEAN,
            "max_overall_dice_drop_vs_strict": MAX_OVERALL_DICE_DROP_VS_STRICT,
        },
        "scope": "development_calibration_only",
        "publication_ready": False,
        "publication_blocker": (
            "These masks were inspected and used during algorithm development; "
            "an independent frozen test set is still required."
        ),
    }


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"VIDEO_PATH does not exist: {VIDEO_PATH}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"MANIFEST_PATH does not exist: {MANIFEST_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH).sort_values("frame").reset_index(drop=True)
    missing_masks = [
        int(frame_idx)
        for frame_idx in manifest["frame"]
        if not (MANUAL_MASK_DIR / f"frame_{int(frame_idx):06d}_mask.png").exists()
    ]
    if missing_masks:
        raise FileNotFoundError(f"Missing manual masks for frames: {missing_masks}")

    temporal_params, temporal_halo = candidate._normalized_temporal_params(
        candidate.TEMPORAL_PARAMS
    )
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")
    source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_total <= 0:
        cap.release()
        raise ValueError("Video frame count is unavailable")

    print("Development manual-mask comparison:")
    print(f"  VIDEO_PATH = {VIDEO_PATH}")
    print(f"  n_manual_masks = {len(manifest)}")
    print(f"  CPU_WORKERS = {CPU_WORKERS}")
    print(f"  temporal_halo = {temporal_halo}")
    print(f"  temporal_params = {temporal_params}")
    print("  WARNING: this is development calibration, not an independent test.")

    rows = []
    executor_context = (
        ProcessPoolExecutor(max_workers=CPU_WORKERS)
        if CPU_WORKERS > 1
        else nullcontext(None)
    )
    try:
        with executor_context as executor:
            for item in tqdm(
                manifest.itertuples(index=False),
                total=len(manifest),
                desc="Manual strict/temporal comparison",
            ):
                frame_idx = int(item.frame)
                frame, strict_mask, temporal_mask, temporal_candidate, temporal_valid = (
                    _predict_target(
                        cap,
                        executor,
                        frame_idx,
                        source_total,
                        temporal_params,
                        temporal_halo,
                    )
                )
                manual_mask = _binary_mask(
                    MANUAL_MASK_DIR / f"frame_{frame_idx:06d}_mask.png"
                )
                if manual_mask.shape != strict_mask.shape:
                    raise ValueError(
                        f"Mask shape mismatch at frame {frame_idx}: "
                        f"manual={manual_mask.shape}, prediction={strict_mask.shape}"
                    )
                strict_metrics = _segmentation_metrics(strict_mask, manual_mask)
                temporal_metrics = _segmentation_metrics(temporal_mask, manual_mask)
                row = {
                    "section": str(item.section),
                    "frame": frame_idx,
                    "temporal_valid": bool(temporal_valid),
                    "temporal_candidate_area_px": int(temporal_candidate.sum()),
                    "temporal_added_area_px": int(
                        np.sum(temporal_mask & (~strict_mask))
                    ),
                    "temporal_removed_area_px": int(
                        np.sum(strict_mask & (~temporal_mask))
                    ),
                    **{f"strict_{key}": value for key, value in strict_metrics.items()},
                    **{
                        f"temporal_{key}": value
                        for key, value in temporal_metrics.items()
                    },
                }
                row["temporal_minus_strict_dice"] = (
                    row["temporal_dice"] - row["strict_dice"]
                )
                row["temporal_minus_strict_recall"] = (
                    row["temporal_recall"] - row["strict_recall"]
                )
                rows.append(row)
                if SAVE_REVIEW_IMAGES:
                    _save_review(
                        frame_idx,
                        frame,
                        manual_mask,
                        strict_mask,
                        temporal_mask,
                        strict_metrics,
                        temporal_metrics,
                    )
    finally:
        cap.release()

    metrics = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    summary = _summarize(metrics)
    decision = _development_decision(summary)

    metrics_path = OUTPUT_DIR / "manual_strict_temporal_metrics.csv"
    summary_path = OUTPUT_DIR / "manual_strict_temporal_summary.csv"
    decision_path = OUTPUT_DIR / "development_gate.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    decision_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Saved per-frame metrics: {metrics_path}")
    print(f"Saved section summary: {summary_path}")
    print(f"Saved development decision: {decision_path}")
    print(f"development_gate_passed = {decision['development_gate_passed']}")
    print("Publication accuracy remains NOT ESTABLISHED by this development check.")


if __name__ == "__main__":
    freeze_support()
    main()
