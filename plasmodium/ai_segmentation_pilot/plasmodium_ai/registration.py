"""Dish-rim registration for separating camera motion from biological change."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    matrix: np.ndarray
    method: str
    applied: bool
    score: float
    translation_px: float
    rotation_deg: float


def _rim_feature(frame, center, radius, max_dimension=720):
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    small = (
        cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else frame
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    feature = cv2.magnitude(gx, gy)
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    cx, cy = float(center[0]) * scale, float(center[1]) * scale
    rr = float(radius) * scale
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    annulus = ((distance >= 0.72 * rr) & (distance <= 1.18 * rr)).astype(np.uint8)
    values = feature[annulus > 0]
    if values.size:
        feature = (feature - np.median(values)) * annulus
    feature = cv2.GaussianBlur(feature.astype(np.float32), (3, 3), 0)
    return feature, (annulus * 255).astype(np.uint8), scale


def estimate_registration(
    reference,
    moving,
    center,
    radius,
    *,
    max_dimension=720,
    max_translation_px=35.0,
    max_rotation_deg=2.0,
    min_ecc_score=0.35,
):
    """Estimate a Euclidean moving-to-reference warp with safe fallbacks."""
    ref, mask, scale = _rim_feature(reference, center, radius, max_dimension)
    cur, _, current_scale = _rim_feature(moving, center, radius, max_dimension)
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    if ref.shape != cur.shape or abs(scale - current_scale) > 1e-9:
        return RegistrationResult(identity, "identity_shape_mismatch", False, np.nan, 0.0, 0.0)

    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    window *= mask.astype(np.float32) / 255.0
    try:
        (phase_x, phase_y), phase_score = cv2.phaseCorrelate(ref, cur, window)
    except cv2.error:
        phase_x = phase_y = 0.0
        phase_score = np.nan

    # ECC's warp maps template coordinates to the input; initialize it with
    # the phase estimate and invert it before returning a direct warp.
    ecc_warp = np.float32([[1, 0, phase_x], [0, 1, phase_y]])
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        ecc_score, ecc_warp = cv2.findTransformECC(
            ref, cur, ecc_warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 3
        )
        direct_small = cv2.invertAffineTransform(ecc_warp)
        direct = direct_small.astype(np.float32)
        direct[:, 2] /= scale
        rotation = float(np.degrees(np.arctan2(direct[1, 0], direct[0, 0])))
        translation = float(np.hypot(direct[0, 2], direct[1, 2]))
        valid = (
            np.isfinite(ecc_score)
            and float(ecc_score) >= float(min_ecc_score)
            and translation <= float(max_translation_px)
            and abs(rotation) <= float(max_rotation_deg)
        )
        if valid:
            return RegistrationResult(
                direct, "ecc_euclidean", True, float(ecc_score), translation, rotation
            )
    except cv2.error:
        pass

    shift_x = -float(phase_x) / scale
    shift_y = -float(phase_y) / scale
    translation = float(np.hypot(shift_x, shift_y))
    phase_valid = (
        np.isfinite(phase_score)
        and float(phase_score) >= 0.30
        and translation <= float(max_translation_px)
    )
    if phase_valid:
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return RegistrationResult(
            matrix, "phase_translation", True, float(phase_score), translation, 0.0
        )
    return RegistrationResult(identity, "identity_low_confidence", False, float(phase_score), 0.0, 0.0)


def warp_to_reference(frame, result):
    if not result.applied:
        return frame.copy()
    return cv2.warpAffine(
        frame,
        result.matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

