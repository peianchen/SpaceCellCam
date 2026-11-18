"""Synthetic controls for segmentation behaviour, not biological validation.

The painted pixels below are known construction targets. These checks cannot
establish accuracy on real Physarum, food appearance, or camera illumination.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from segmentation_lab.contracts import RoiSpec
from segmentation_lab.segmenter import segment


METHODS = ("adaptive_chroma", "local_lab")
ROI = RoiSpec((128, 128), 110, 4)
YELLOW_BGR = (24, 180, 200)


def _network():
    """One thin loop with branches, plus a genuinely disconnected branch."""
    truth = np.zeros((256, 256), np.uint8)
    cv2.circle(truth, (105, 110), 30, 1, 2)
    cv2.line(truth, (130, 125), (186, 150), 1, 2)
    cv2.line(truth, (150, 134), (175, 101), 1, 1)
    disconnected = np.zeros_like(truth)
    cv2.line(disconnected, (70, 180), (126, 188), 1, 2)
    truth |= disconnected
    image = np.full((256, 256, 3), 85, np.uint8)
    image[truth.astype(bool)] = YELLOW_BGR
    hole = np.zeros_like(truth)
    cv2.circle(hole, (105, 110), 25, 1, -1)
    return image, truth.astype(bool), disconnected.astype(bool), hole.astype(bool)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("background", ("black", "gray", "low_noise"))
def test_empty_background_does_not_invent_tissue(method, background):
    if background == "black":
        image = np.zeros((256, 256, 3), np.uint8)
    elif background == "gray":
        image = np.full((256, 256, 3), 85, np.uint8)
    else:
        noise = np.random.default_rng(20261118).normal(0, 1.5, (256, 256, 3))
        image = np.clip(np.rint(85 + noise), 0, 255).astype(np.uint8)
    result = segment(image, ROI, method=method)
    assert not result.mask.any(), f"{background}: invented {result.mask.sum()} foreground pixels"
    assert not result.excluded_food.any()
    assert "empty_no_supported_tissue" in result.qc_flags


@pytest.mark.parametrize("method", METHODS)
def test_thin_network_disconnected_branch_and_true_hole_are_preserved(method):
    image, tissue, disconnected, hole = _network()
    result = segment(image, ROI, method=method)
    recall = float((result.mask & tissue).sum() / tissue.sum())
    detached_recall = float((result.mask & disconnected).sum() / disconnected.sum())
    assert recall >= 0.95, f"Painted thin-network recall was {recall:.4f}"
    assert detached_recall >= 0.95, f"Disconnected branch recall was {detached_recall:.4f}"
    assert not result.mask[hole].any(), "True central network hole was filled"
    assert not (result.mask & ~tissue).any(), "Neutral pixels were added around thin tissue"


@pytest.mark.parametrize("method", METHODS)
def test_yellow_marker_outside_roi_is_excluded_without_moving_coordinates(method):
    # An off-centre ROI forces an internal crop and restoration to source coordinates.
    image = np.full((600, 820, 3), 85, np.uint8)
    roi = RoiSpec((230, 370), 100, 4)
    tissue = np.zeros(image.shape[:2], np.uint8)
    cv2.line(tissue, (195, 330), (275, 404), 1, 2)
    cv2.line(tissue, (239, 371), (283, 355), 1, 1)
    marker = np.zeros_like(tissue)
    cv2.circle(marker, (720, 60), 24, 1, -1)
    image[(tissue | marker).astype(bool)] = YELLOW_BGR
    before = image.copy()
    result = segment(image, roi, method=method)
    result.validate(image.shape[:2])
    assert np.array_equal(image, before), "The input image was modified"
    assert result.mask[tissue.astype(bool)].mean() >= 0.95
    assert not result.mask[marker.astype(bool)].any()
    assert not (result.mask & ~roi.mask(image.shape[:2])).any()
    assert result.preprocessed.shape == image.shape
    assert np.isfinite(result.score).all() and result.score.min() >= 0 and result.score.max() <= 1


@pytest.mark.parametrize("method", METHODS)
def test_moderate_global_exposure_changes_keep_same_constructed_network(method):
    image, tissue, _, _ = _network()
    base = segment(image, ROI, method=method).mask
    for gain in (0.45, 0.7, 1.2):
        exposed = np.clip(np.rint(image.astype(np.float32) * gain), 0, 255).astype(np.uint8)
        result = segment(exposed, ROI, method=method)
        union = (base | result.mask).sum()
        overlap = float((base & result.mask).sum() / max(1, union))
        recall = float((result.mask & tissue).sum() / tissue.sum())
        assert overlap >= 0.95, f"gain={gain}: synthetic mask overlap={overlap:.4f}"
        assert recall >= 0.95, f"gain={gain}: synthetic painted-network recall={recall:.4f}"


@pytest.mark.parametrize("method", METHODS)
def test_central_brown_food_is_excluded_without_removing_yellow_surround(method):
    image = np.full((256, 256, 3), 85, np.uint8)
    surrounding = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(surrounding, (128, 128), 44, 1, -1)
    cv2.line(surrounding, (170, 128), (207, 140), 1, 2)
    food = np.zeros_like(surrounding)
    cv2.circle(food, (128, 128), 16, 1, -1)
    image[surrounding.astype(bool)] = YELLOW_BGR
    image[food.astype(bool)] = (18, 85, 150)
    tissue = surrounding.astype(bool) & ~food.astype(bool)
    result = segment(image, ROI, method=method)
    excluded_recall = float(result.excluded_food[food.astype(bool)].mean())
    retained_tissue = float(result.mask[tissue].mean())
    assert excluded_recall >= 0.90, f"Central brown disk excluded fraction={excluded_recall:.4f}"
    assert not result.mask[food.astype(bool)].any(), "Food was reported as occupied tissue"
    assert retained_tissue >= 0.90, f"Yellow surround retained fraction={retained_tissue:.4f}"
    assert "food_boundary_requires_review" in result.qc_flags


@pytest.mark.parametrize("method", METHODS)
def test_achromatic_glare_is_not_a_yellow_branch(method):
    image, tissue, _, _ = _network()
    glare = np.zeros(image.shape[:2], np.uint8)
    cv2.line(glare, (165, 60), (195, 95), 1, 4)
    assert not (glare.astype(bool) & tissue).any()
    image[glare.astype(bool)] = (255, 255, 255)
    result = segment(image, ROI, method=method)
    assert not result.mask[glare.astype(bool)].any()
    assert result.mask[tissue].mean() >= 0.95
