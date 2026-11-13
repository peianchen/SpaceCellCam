"""Small binary-mask helpers shared by segmentation and temporal recovery."""

from __future__ import annotations

import cv2
import numpy as np
from skimage import measure


def as_gray_uint8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype == np.uint8:
        return gray
    gray = gray.astype(np.float32)
    finite = np.isfinite(gray)
    if not np.any(finite):
        return np.zeros(gray.shape, dtype=np.uint8)
    low, high = np.nanpercentile(gray[finite], [1, 99])
    if high <= low:
        low, high = np.nanmin(gray[finite]), np.nanmax(gray[finite])
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return (np.clip((gray - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def circular_roi_mask(shape: tuple[int, int], center, radius) -> np.ndarray:
    height, width = shape[:2]
    center_x, center_y = center
    yy, xx = np.ogrid[:height, :width]
    return ((xx - int(center_x)) ** 2 + (yy - int(center_y)) ** 2 <= int(radius) ** 2)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return np.zeros_like(mask, dtype=bool)
    selected = max(props, key=lambda region: region.area)
    return labels == selected.label


def keep_components_touching_mask(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if candidate.shape != reference.shape:
        raise ValueError(f"Mask shape mismatch: {candidate.shape} vs {reference.shape}")
    if not np.any(candidate) or not np.any(reference):
        return np.zeros_like(candidate, dtype=bool)
    labels = measure.label(candidate)
    touching = np.unique(labels[reference & (labels > 0)])
    return np.isin(labels, touching) if touching.size else np.zeros_like(candidate, dtype=bool)

