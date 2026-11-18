"""Mask and ROI contracts, independent of quantitative analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RoiSpec:
    center: tuple[float, float]
    radius: float
    margin_px: float = 8.0

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        return (xx - self.center[0]) ** 2 + (yy - self.center[1]) ** 2 <= max(0., self.radius - self.margin_px) ** 2


@dataclass
class SegmentationResult:
    """mask is visible occupied tissue; food and true network holes are excluded.

    score is feature evidence in [0,1], NOT a calibrated probability.
    clean must not silently fill network holes; envelope is diagnostic only.
    """
    method: str
    mask: np.ndarray
    preprocessed: np.ndarray
    score: np.ndarray
    candidate: np.ndarray
    roi_mask: np.ndarray
    excluded_food: np.ndarray
    qc_flags: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        for name in ("mask", "candidate", "roi_mask", "excluded_food"):
            arr = getattr(self, name)
            if arr.shape != shape or arr.dtype != np.bool_:
                raise ValueError(f"{name} must be bool with image shape")
        if self.score.shape != shape or not np.isfinite(self.score).all() or np.any((self.score < 0) | (self.score > 1)):
            raise ValueError("score must be a finite per-pixel feature evidence map")
        if self.preprocessed.shape[:2] != shape:
            raise ValueError("preprocessed must use the original image coordinates")
        if np.any(self.mask & ~self.roi_mask):
            raise ValueError("mask extends beyond analysis ROI")
        if np.any(self.mask & self.excluded_food):
            raise ValueError("occupied tissue overlaps excluded food")
        if np.any(self.excluded_food & ~self.roi_mask):
            raise ValueError("excluded food extends beyond analysis ROI")
