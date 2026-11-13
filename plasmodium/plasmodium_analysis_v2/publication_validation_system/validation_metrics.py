"""Binary-mask and temporal agreement metrics used by the validation report."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology


def safe_ratio(numerator: float, denominator: float, empty_value: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float(empty_value)


def confusion_counts(prediction: np.ndarray, reference: np.ndarray) -> dict[str, int]:
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    return {
        "tp": int(np.sum(pred & ref)),
        "fp": int(np.sum(pred & (~ref))),
        "fn": int(np.sum((~pred) & ref)),
        "tn": int(np.sum((~pred) & (~ref))),
    }


def overlap_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    counts = confusion_counts(prediction, reference)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    both_empty = tp + fp + fn == 0
    dice = 1.0 if both_empty else safe_ratio(2 * tp, 2 * tp + fp + fn, 0.0)
    iou = 1.0 if both_empty else safe_ratio(tp, tp + fp + fn, 0.0)
    precision = 1.0 if tp + fp == 0 else safe_ratio(tp, tp + fp, 0.0)
    recall = 1.0 if tp + fn == 0 else safe_ratio(tp, tp + fn, 0.0)
    return {
        **counts,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = morphology.binary_erosion(mask, morphology.disk(1))
    return mask & (~eroded)


def boundary_f1(
    prediction: np.ndarray,
    reference: np.ndarray,
    tolerance_px: int,
) -> dict[str, float]:
    pred_boundary = mask_boundary(prediction)
    ref_boundary = mask_boundary(reference)
    pred_count = int(pred_boundary.sum())
    ref_count = int(ref_boundary.sum())
    if pred_count == 0 and ref_count == 0:
        return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f1": 1.0}
    radius = max(0, int(tolerance_px))
    footprint = morphology.disk(radius) if radius > 0 else np.ones((1, 1), dtype=bool)
    pred_matches = pred_boundary & morphology.binary_dilation(ref_boundary, footprint)
    ref_matches = ref_boundary & morphology.binary_dilation(pred_boundary, footprint)
    precision = safe_ratio(int(pred_matches.sum()), pred_count, 0.0)
    recall = safe_ratio(int(ref_matches.sum()), ref_count, 0.0)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall, 0.0)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


def hd95(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred_boundary = mask_boundary(prediction)
    ref_boundary = mask_boundary(reference)
    if not np.any(pred_boundary) and not np.any(ref_boundary):
        return 0.0
    if not np.any(pred_boundary) or not np.any(ref_boundary):
        return float("nan")
    distance_to_ref = ndi.distance_transform_edt(~ref_boundary)
    distance_to_pred = ndi.distance_transform_edt(~pred_boundary)
    distances = np.concatenate(
        [distance_to_ref[pred_boundary], distance_to_pred[ref_boundary]]
    )
    return float(np.percentile(distances, 95))


def skeleton_summary(mask: np.ndarray, pixel_size: float = 1.0) -> dict[str, float | int]:
    skeleton = morphology.skeletonize(mask.astype(bool))
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    neighbors = ndi.convolve(skeleton.astype(np.uint8), neighbor_kernel, mode="constant")
    endpoint_pixels = skeleton & (neighbors == 1)
    branch_pixels = skeleton & (neighbors >= 3)
    _, endpoints = ndi.label(endpoint_pixels, structure=np.ones((3, 3), dtype=np.uint8))
    _, branchpoints = ndi.label(branch_pixels, structure=np.ones((3, 3), dtype=np.uint8))
    _, components = ndi.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    return {
        "skeleton_length_px": int(skeleton.sum()),
        "skeleton_length": float(skeleton.sum() * pixel_size),
        "n_endpoints": int(endpoints),
        "n_branchpoints": int(branchpoints),
        "n_skeleton_components": int(components),
    }


def cldice(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    pred_skeleton = morphology.skeletonize(pred)
    ref_skeleton = morphology.skeletonize(ref)
    pred_count = int(pred_skeleton.sum())
    ref_count = int(ref_skeleton.sum())
    if pred_count == 0 and ref_count == 0:
        return 1.0
    topology_precision = safe_ratio(int(np.sum(pred_skeleton & ref)), pred_count, 0.0)
    topology_sensitivity = safe_ratio(int(np.sum(ref_skeleton & pred)), ref_count, 0.0)
    return safe_ratio(
        2.0 * topology_precision * topology_sensitivity,
        topology_precision + topology_sensitivity,
        0.0,
    )


def circular_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - int(center[0])) ** 2 + (yy - int(center[1])) ** 2 <= int(radius) ** 2


def artifact_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    dish_center: tuple[int, int],
    dish_radius: int,
    rim_width_px: int,
) -> dict[str, float | int]:
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    outer = circular_mask(pred.shape, dish_center, dish_radius)
    inner_radius = max(0, int(dish_radius) - max(0, int(rim_width_px)))
    inner = circular_mask(pred.shape, dish_center, inner_radius)
    ring = outer & (~inner)
    false_positive = pred & (~ref)
    rim_fp_px = int(np.sum(false_positive & ring))
    internal_holes = ndi.binary_fill_holes(ref) & (~ref)
    internal_hole_fp_px = int(np.sum(pred & internal_holes))
    pred_area = int(pred.sum())
    return {
        "rim_fp_px": rim_fp_px,
        "rim_fp_fraction_of_prediction": safe_ratio(rim_fp_px, pred_area, 0.0),
        "rim_fp_rate_in_ring": safe_ratio(rim_fp_px, int(ring.sum()), 0.0),
        "internal_hole_fp_px": internal_hole_fp_px,
        "internal_hole_fp_fraction_of_prediction": safe_ratio(
            internal_hole_fp_px, pred_area, 0.0
        ),
    }


def evaluate_mask_pair(
    prediction: np.ndarray,
    reference: np.ndarray,
    pixel_size: float,
    dish_center: tuple[int, int],
    dish_radius: int,
    boundary_tolerance_px: int,
    rim_width_px: int,
) -> dict[str, float | int]:
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    pred_area_px = int(pred.sum())
    ref_area_px = int(ref.sum())
    signed_area_error = safe_ratio(pred_area_px - ref_area_px, ref_area_px, 0.0)
    pred_skeleton = skeleton_summary(pred, pixel_size)
    ref_skeleton = skeleton_summary(ref, pixel_size)
    ref_length = float(ref_skeleton["skeleton_length_px"])
    length_error = safe_ratio(
        float(pred_skeleton["skeleton_length_px"]) - ref_length,
        ref_length,
        0.0,
    )
    return {
        **overlap_metrics(pred, ref),
        **boundary_f1(pred, ref, boundary_tolerance_px),
        "hd95_px": hd95(pred, ref),
        "cldice": cldice(pred, ref),
        "prediction_area_px": pred_area_px,
        "reference_area_px": ref_area_px,
        "prediction_area": float(pred_area_px * pixel_size**2),
        "reference_area": float(ref_area_px * pixel_size**2),
        "signed_area_error": signed_area_error,
        "absolute_area_error": abs(signed_area_error),
        "prediction_skeleton_length_px": pred_skeleton["skeleton_length_px"],
        "reference_skeleton_length_px": ref_skeleton["skeleton_length_px"],
        "skeleton_length_signed_error": length_error,
        "skeleton_length_absolute_error": abs(length_error),
        "prediction_endpoints": pred_skeleton["n_endpoints"],
        "reference_endpoints": ref_skeleton["n_endpoints"],
        "endpoint_count_error": int(pred_skeleton["n_endpoints"])
        - int(ref_skeleton["n_endpoints"]),
        "prediction_branchpoints": pred_skeleton["n_branchpoints"],
        "reference_branchpoints": ref_skeleton["n_branchpoints"],
        "branchpoint_count_error": int(pred_skeleton["n_branchpoints"])
        - int(ref_skeleton["n_branchpoints"]),
        "prediction_skeleton_components": pred_skeleton["n_skeleton_components"],
        "reference_skeleton_components": ref_skeleton["n_skeleton_components"],
        **artifact_metrics(pred, ref, dish_center, dish_radius, rim_width_px),
    }


def front_motion(mask_previous: np.ndarray, mask_current: np.ndarray, dt: float) -> dict[str, float | int]:
    previous = mask_previous.astype(bool)
    current = mask_current.astype(bool)
    expansion = current & (~previous)
    retraction = previous & (~current)
    distance_from_previous = ndi.distance_transform_edt(~previous)
    distance_from_current = ndi.distance_transform_edt(~current)
    expansion_distance = (
        float(np.mean(distance_from_previous[expansion])) if np.any(expansion) else 0.0
    )
    retraction_distance = (
        float(np.mean(distance_from_current[retraction])) if np.any(retraction) else 0.0
    )
    return {
        "expansion_area_px": int(expansion.sum()),
        "retraction_area_px": int(retraction.sum()),
        "net_area_change_px": int(current.sum()) - int(previous.sum()),
        "expansion_front_distance_px_mean": expansion_distance,
        "retraction_front_distance_px_mean": retraction_distance,
        "expansion_front_speed_px_per_s": expansion_distance / float(dt),
        "retraction_front_speed_px_per_s": retraction_distance / float(dt),
    }


def evaluate_temporal_pair(
    pred_previous: np.ndarray,
    pred_current: np.ndarray,
    ref_previous: np.ndarray,
    ref_current: np.ndarray,
    frame_interval_s: float,
) -> dict[str, float | int]:
    pred_expansion = pred_current.astype(bool) & (~pred_previous.astype(bool))
    pred_retraction = pred_previous.astype(bool) & (~pred_current.astype(bool))
    ref_expansion = ref_current.astype(bool) & (~ref_previous.astype(bool))
    ref_retraction = ref_previous.astype(bool) & (~ref_current.astype(bool))
    pred_motion = front_motion(pred_previous, pred_current, frame_interval_s)
    ref_motion = front_motion(ref_previous, ref_current, frame_interval_s)
    expansion = overlap_metrics(pred_expansion, ref_expansion)
    retraction = overlap_metrics(pred_retraction, ref_retraction)
    return {
        "expansion_dice": expansion["dice"],
        "expansion_precision": expansion["precision"],
        "expansion_recall": expansion["recall"],
        "retraction_dice": retraction["dice"],
        "retraction_precision": retraction["precision"],
        "retraction_recall": retraction["recall"],
        "prediction_net_area_change_px": pred_motion["net_area_change_px"],
        "reference_net_area_change_px": ref_motion["net_area_change_px"],
        "net_area_change_error_px": int(pred_motion["net_area_change_px"])
        - int(ref_motion["net_area_change_px"]),
        "prediction_expansion_front_speed_px_per_s": pred_motion[
            "expansion_front_speed_px_per_s"
        ],
        "reference_expansion_front_speed_px_per_s": ref_motion[
            "expansion_front_speed_px_per_s"
        ],
        "expansion_front_speed_absolute_error_px_per_s": abs(
            float(pred_motion["expansion_front_speed_px_per_s"])
            - float(ref_motion["expansion_front_speed_px_per_s"])
        ),
        "prediction_retraction_front_speed_px_per_s": pred_motion[
            "retraction_front_speed_px_per_s"
        ],
        "reference_retraction_front_speed_px_per_s": ref_motion[
            "retraction_front_speed_px_per_s"
        ],
        "retraction_front_speed_absolute_error_px_per_s": abs(
            float(pred_motion["retraction_front_speed_px_per_s"])
            - float(ref_motion["retraction_front_speed_px_per_s"])
        ),
    }


def finite_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def relative_error(value: float, reference: float) -> float:
    if not math.isfinite(reference) or reference == 0:
        return float("nan")
    return float((value - reference) / reference)
