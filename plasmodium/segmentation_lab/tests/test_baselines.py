"""Small synthetic checks of baseline provenance and comparison semantics."""
import json

import cv2
import numpy as np
import pytest

from segmentation_lab.baselines import _extract_main_hsv_params, run_baseline
from segmentation_lab.contracts import RoiSpec


def test_parameter_extraction_uses_entry_scope_and_never_evaluates_code():
    source = '''
hsv_params = dict(s_min=99)
if __name__ == "__main__":
    raise RuntimeError("must not execute")
    hsv_params = dict(s_min=54, keep_largest=True)
def run(settings):
    hsv_params = dict(s_min=55)
'''
    assert _extract_main_hsv_params(source, "__main__")[0] == {
        "s_min": 54, "keep_largest": True,
    }
    assert _extract_main_hsv_params(source, "run")[0] == {"s_min": 55}
    with pytest.raises(ValueError):
        _extract_main_hsv_params('def run(settings):\n    hsv_params = dict(s_min=danger())', "run")


def test_original_uses_main_saturation_threshold_and_preserves_input():
    image = np.full((96, 96, 3), 80, np.uint8)
    low_saturation_yellow = cv2.cvtColor(
        np.array([[[15, 57, 200]]], np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]
    image[28:68, 28:68] = low_saturation_yellow
    before = image.copy()
    result = run_baseline(image, RoiSpec((48, 48), 44), "legacy_original")
    assert result.mask[48, 48]  # main S>=54; function default S>=60 would miss it.
    assert result.diagnostics["hsv_params"]["s_min"] == 54
    assert np.array_equal(image, before)
    assert np.array_equal(result.preprocessed, before)
    assert not result.excluded_food.any()
    assert "baseline_food_not_excluded" in result.qc_flags
    assert set(np.unique(result.score)) <= {0., 1.}
    json.dumps(result.diagnostics)


def test_baseline_keeps_true_holes_and_does_not_apply_requested_roi_margin():
    image = np.full((96, 96, 3), (30, 170, 190), np.uint8)
    image[40:56, 40:56] = 80
    result = run_baseline(image, RoiSpec((48.9, 48.9), 40.9, margin_px=20), "legacy_original")
    assert not result.mask[48, 48]
    assert result.mask[48, 82]  # Inside old radius 40, outside requested margin-shrunken ROI.
    assert result.diagnostics["effective_roi"] == {"center": [48, 48], "radius": 40, "margin_px": 0}
    assert result.diagnostics["clean_area_px"] == result.diagnostics["occupied_area_px"]
    assert result.diagnostics["envelope_area_px"] > result.diagnostics["occupied_area_px"]
    assert not result.diagnostics["complete_original_run_replayed"]


def test_v2_uses_actual_chroma_extension_and_records_all_sources():
    image = np.full((96, 96, 3), 80, np.uint8)
    image[32:64, 32:64] = (30, 170, 190)
    image[44:52, 64:78] = (125, 150, 150)  # S<54 but yellow chroma=25.
    roi = RoiSpec((48, 48), 44)
    original = run_baseline(image, roi, "legacy_original")
    v2 = run_baseline(image, roi, "v2_spatial")
    assert not original.mask[48, 67]
    assert v2.mask[48, 67]
    assert v2.diagnostics["hsv_params"]["use_vesselness_recovery"] is True
    assert len(v2.diagnostics["source_files"]) == 3
    assert all(len(item["sha256"]) == 64 for item in v2.diagnostics["source_files"])
    assert not v2.diagnostics["food_exclusion_applied"]
    json.dumps(v2.diagnostics)


@pytest.mark.parametrize("method", ["legacy_original", "v2_spatial"])
def test_gray_background_stays_empty_and_is_flagged(method):
    result = run_baseline(np.full((64, 64, 3), 100, np.uint8), RoiSpec((32, 32), 28), method)
    assert not result.mask.any()
    assert "baseline_empty_mask" in result.qc_flags


def test_invalid_inputs_and_unknown_method_fail_explicitly():
    roi = RoiSpec((16, 16), 12)
    with pytest.raises(ValueError, match="uint8 BGR"):
        run_baseline(np.zeros((32, 32), np.uint8), roi, "legacy_original")
    with pytest.raises(ValueError, match="Unknown baseline"):
        run_baseline(np.zeros((32, 32, 3), np.uint8), roi, "unknown")
