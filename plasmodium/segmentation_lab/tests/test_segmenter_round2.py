"""Independent synthetic controls for round 2, not biological accuracy evidence."""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from segmentation_lab.contracts import RoiSpec
from segmentation_lab.segmenter import segment as round1
from segmentation_lab.segmenter_round2 import DEFAULTS, prepare_baseline, segment


ROI = RoiSpec((128, 128), 110, 4)


def _dark_lines():
    image = np.full((256, 256, 3), 18, np.uint8)
    tissue = np.zeros(image.shape[:2], np.uint8)
    cv2.line(tissue, (50, 118), (196, 138), 1, 2)
    cv2.line(tissue, (122, 128), (154, 75), 1, 1)
    cv2.line(tissue, (77, 175), (135, 188), 1, 1)
    image[tissue.astype(bool)] = (8, 36, 42)
    return image, tissue.astype(bool)


@pytest.mark.parametrize("level", (0, 18, 85, 220))
def test_empty_gray_does_not_gain_foreground(level):
    image = np.full((256, 256, 3), level, np.uint8)
    result = segment(image, ROI)
    assert not result.mask.any()
    assert not result.excluded_food.any()
    assert result.diagnostics["temporal_mask_editing"] is False


@pytest.mark.parametrize("line_level", (110, 245, 255))
def test_achromatic_gray_and_bright_ridges_are_rejected(line_level):
    image = np.full((256, 256, 3), 45, np.uint8)
    cv2.line(image, (55, 80), (190, 175), (line_level,) * 3, 2, cv2.LINE_AA)
    cv2.circle(image, (110, 130), 25, (line_level,) * 3, 1, cv2.LINE_AA)
    assert not segment(image, ROI).mask.any()


def test_dark_yellow_thin_and_disconnected_lines_are_detected():
    image, tissue = _dark_lines()
    result = segment(image, ROI)
    retained = float(result.mask[tissue].mean())
    assert retained >= .90, f"Synthetic dark-yellow raster retained fraction={retained:.4f}"
    assert not (result.mask & ~ROI.mask(image.shape[:2])).any()


def test_deterministic_additive_and_inputs_immutable():
    image, _ = _dark_lines()
    image_before = image.copy()
    base = prepare_baseline(image, ROI)
    arrays_before = {name: getattr(base, name).copy() for name in
                     ("mask", "candidate", "score", "preprocessed", "roi_mask", "excluded_food")}
    first = segment(image, ROI, baseline=base)
    second = segment(image, ROI, baseline=base)
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.score, second.score)
    assert np.array_equal(image, image_before)
    for name, before in arrays_before.items():
        assert np.array_equal(getattr(base, name), before), f"Cached baseline {name} mutated"
    assert not (base.mask & ~first.mask).any(), "An additive branch path removed round-1 pixels"
    assert np.array_equal(first.roi_mask, base.roi_mask)
    assert np.array_equal(first.excluded_food, base.excluded_food)
    first.validate(image.shape[:2])


def test_food_disjoint_roi_outside_and_cached_geometry_are_preserved():
    image = np.full((300, 420, 3), 85, np.uint8)
    roi = RoiSpec((170, 150), 105, 5)
    cv2.circle(image, (170, 150), 43, (24, 180, 200), -1)
    cv2.circle(image, (170, 150), 16, (18, 85, 150), -1)
    cv2.line(image, (205, 135), (237, 96), (8, 36, 42), 2)
    cv2.circle(image, (360, 55), 18, (24, 180, 200), -1)
    base = prepare_baseline(image, roi)
    assert base.excluded_food.any(), "Constructed brown core was not detected by the frozen baseline"
    result = segment(image, roi, baseline=base)
    assert not (result.mask & result.excluded_food).any()
    assert not (result.mask & ~roi.mask(image.shape[:2])).any()
    assert not result.mask[30:80, 335:385].any()
    assert np.array_equal(result.roi_mask, base.roi_mask)
    assert np.array_equal(result.excluded_food, base.excluded_food)
    assert not (base.mask & ~result.mask).any()


def test_cached_and_uncached_results_match_for_same_inputs():
    image, _ = _dark_lines()
    base_config = {"food_exclusion": False, "refine_roi": False}
    base = prepare_baseline(image, ROI, base_config=base_config)
    cached = segment(image, ROI, base_config=base_config, baseline=base)
    direct = segment(image, ROI, base_config=base_config)
    assert np.array_equal(cached.mask, direct.mask)
    assert np.array_equal(cached.score, direct.score)


@pytest.mark.parametrize("config", [
    {"unknown_round2_setting": 1}, {"noise_floor": 0}, {"noise_floor": -1},
    {"noise_floor": float("nan")}, {"ridge_floor": float("inf")},
    {"contrast_floor": True}, {"min_component_px": "24"}, {"noise_tile_px": 7},
    {"noise_tile_px": 8.5}, {"min_seed_px": 1.5},
])
def test_unknown_or_invalid_parameters_rejected(config):
    assert "noise_floor" in DEFAULTS  # The public config contract tested below.
    with pytest.raises(ValueError):
        segment(np.full((256, 256, 3), 85, np.uint8), ROI, config=config)


@pytest.mark.parametrize("cached", (False, True))
def test_float_image_is_rejected_consistently_with_or_without_cached_baseline(cached):
    image, _ = _dark_lines()
    base = prepare_baseline(image, ROI) if cached else None
    with pytest.raises(ValueError):
        segment(image.astype(np.float32), ROI, baseline=base)


@pytest.mark.parametrize("cached", (False, True))
def test_invalid_roi_is_rejected_consistently_with_or_without_cached_baseline(cached):
    image, _ = _dark_lines()
    base = prepare_baseline(image, ROI) if cached else None
    with pytest.raises(ValueError):
        segment(image, RoiSpec((128, 128), -1, 0), baseline=base)


def test_cached_mask_wrong_shape_or_dtype_is_rejected():
    image, _ = _dark_lines()
    base = prepare_baseline(image, ROI)
    for invalid in (base.mask[:-1], base.mask.astype(np.uint8)):
        with pytest.raises(ValueError):
            segment(image, ROI, baseline=replace(base, mask=invalid))


@pytest.mark.parametrize("mismatch", ("same_shape_image", "roi", "base_config"))
def test_bound_cache_rejects_different_source_geometry_or_parameters(mismatch):
    image, _ = _dark_lines()
    config = {"food_exclusion": False, "refine_roi": False}
    base = prepare_baseline(image, ROI, base_config=config)
    requested_image, requested_roi, requested_config = image.copy(), ROI, dict(config)
    if mismatch == "same_shape_image":
        requested_image[100, 100, 0] += 1
    elif mismatch == "roi":
        requested_roi = RoiSpec((129, 128), 110, 4)
    else:
        requested_config["food_exclusion"] = True
    with pytest.raises(ValueError, match="Cached baseline does not match"):
        segment(requested_image, requested_roi, baseline=base, base_config=requested_config)


def test_unbound_round1_result_cannot_be_passed_as_a_trusted_cache():
    image, _ = _dark_lines()
    with pytest.raises(ValueError, match="prepare_baseline"):
        segment(image, ROI, baseline=round1(image, ROI))


def test_mutating_callers_config_after_baseline_creation_invalidates_reuse():
    image, _ = _dark_lines()
    config = {"low_evidence": .045, "high_evidence": .105}
    base = prepare_baseline(image, ROI, base_config=config)
    config["low_evidence"] = .080
    with pytest.raises(ValueError, match="Cached baseline does not match"):
        segment(image, ROI, baseline=base, base_config=config)


def test_added_line_near_roi_boundary_recomputes_final_mask_qc():
    image = np.full((256, 256, 3), 85, np.uint8)
    # This weak 2px line has no round-1 high-threshold seed, but supports a ridge.
    # It sits 1..4 px inside the left edge of the effective circular ROI.
    cv2.line(image, (25, 108), (25, 148), (78, 90, 94), 2)
    config = {"food_exclusion": False, "refine_roi": False}
    base = prepare_baseline(image, ROI, base_config=config)
    assert not base.mask.any(), "The intended new-branch control must start with an empty baseline"
    assert base.diagnostics["rim_fraction"] == 0
    assert "tissue_near_roi_boundary" not in base.qc_flags
    result = segment(image, ROI, base_config=config, baseline=base)
    assert result.mask.any(), "The round-2 path failed to recover the constructed near-boundary line"
    assert result.diagnostics["added_branch_area_px"] == int(result.mask.sum())
    assert result.diagnostics["occupied_area_px"] == int(result.mask.sum())
    assert result.diagnostics["area_fraction"] == pytest.approx(result.mask.sum() / result.roi_mask.sum())
    assert result.diagnostics["rim_fraction"] >= .90
    assert result.diagnostics["quality_score"] < base.diagnostics["quality_score"]
    assert "tissue_near_roi_boundary" in result.qc_flags
    assert "empty_no_supported_tissue" not in result.qc_flags
    assert "tissue_near_roi_boundary" not in base.qc_flags, "Round-1 QC was mutated"
