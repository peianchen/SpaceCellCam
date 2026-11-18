"""Small synthetic regressions for review findings; never research accuracy tests."""
from dataclasses import replace
import json

import cv2
import numpy as np
import pytest

from segmentation_lab import segmenter
from segmentation_lab.benchmark import _load_cached_raw, _raw_cache, pixel_sha256
from segmentation_lab.contracts import RoiSpec, SegmentationResult


def _food_with_visible_yellow_patch():
    image = np.full((128, 128, 3), 90, np.uint8)
    cv2.circle(image, (64, 64), 13, (20, 70, 140), -1)
    cv2.circle(image, (64, 64), 4, (10, 210, 220), -1)
    return image, RoiSpec((64, 64), 52, 4)


@pytest.mark.parametrize("refine", [False, True])
def test_food_hole_filling_preserves_visible_yellow_surface(refine):
    image, roi = _food_with_visible_yellow_patch()
    result = segmenter.segment(image, roi, config={"food_refine": refine})
    assert result.excluded_food.any(), "The brown food still needs exclusion"
    yellow_surface = np.all(image == (10, 210, 220), axis=2)
    assert not result.excluded_food[yellow_surface].any()
    assert result.mask[62:67, 62:67].all()
    assert not (result.mask & result.excluded_food).any()


def test_food_refinement_failure_keeps_yellow_surface_rule(monkeypatch):
    image, roi = _food_with_visible_yellow_patch()

    def fail_graphcut(*args, **kwargs):
        raise cv2.error("synthetic graph-cut failure")

    monkeypatch.setattr(segmenter.cv2, "grabCut", fail_graphcut)
    result = segmenter.segment(image, roi, config={"food_refine": True})
    assert "food_refinement_failed" in result.qc_flags
    yellow_surface = np.all(image == (10, 210, 220), axis=2)
    assert not result.excluded_food[yellow_surface].any()
    assert result.mask[62:67, 62:67].all()


@pytest.mark.parametrize("config", [
    {"high_evidence": float("inf")}, {"low_evidence": float("nan")},
    {"food_seed_rg": float("nan")}, {"noise_multiplier": 0},
    {"saturation_clip": -1}, {"saturation_clip": 256},
    {"background_scale": 1.5}, {"food_max_center_fraction": 0},
    {"food_exclusion": "false"}, {"food_refine": 1},
])
def test_invalid_parameters_are_errors_not_successful_empty_masks(config):
    image, roi = _food_with_visible_yellow_patch()
    with pytest.raises(ValueError):
        segmenter.segment(image, roi, config=config)


def test_refinement_cannot_reduce_roi_below_exclusion_margin():
    image = np.full((256, 256, 3), 70, np.uint8)
    cv2.circle(image, (128, 128), 84, (220, 220, 220), 5)
    roi = RoiSpec((128, 128), 100, 94)
    assert roi.mask(image.shape[:2]).sum() >= 64
    with pytest.raises(ValueError, match="Refined ROI|Refined analysis ROI"):
        segmenter.segment(image, roi, config={"refine_roi": True})


def test_partially_visible_roi_is_explicitly_flagged():
    image = np.full((128, 128, 3), 70, np.uint8)
    result = segmenter.segment(image, RoiSpec((-30, 64), 60, 4))
    assert "roi_clipped_by_image" in result.qc_flags


def _valid_contract():
    empty = np.zeros((4, 4), bool)
    tissue = empty.copy()
    tissue[1, 1] = True
    return SegmentationResult("synthetic", tissue, np.zeros((4, 4, 3), np.uint8),
                              tissue.astype(np.float32), tissue.copy(),
                              np.ones((4, 4), bool), empty)


@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf")])
def test_contract_rejects_out_of_range_or_nonfinite_feature_score(value):
    result = replace(_valid_contract(), score=np.full((4, 4), value, np.float32))
    with pytest.raises(ValueError, match="score"):
        result.validate((4, 4))


def test_contract_rejects_food_and_tissue_overlap():
    result = _valid_contract()
    with pytest.raises(ValueError, match="overlaps excluded food"):
        replace(result, excluded_food=result.mask.copy()).validate((4, 4))


def _cache(tmp_path):
    # This is a metadata-only fixture. The byte file is deliberately not a video,
    # because this test must exercise cache validation without decoding footage.
    video = tmp_path / "synthetic_source.bin"
    video.write_bytes(b"synthetic source identity")
    image = np.full((24, 32, 3), 70, np.uint8)
    image[8:14, 10:20] = (10, 190, 210)
    cache = tmp_path / "cache"
    cache.mkdir()
    cv2.imwrite(str(cache / "original.png"), image)
    stat = video.stat()
    row = {"sample_id": "synthetic_cache_01", "source_video": str(video),
           "frame_index": "5", "image_path": "original.png",
           "source_pixel_sha256": pixel_sha256(image), "decoded_frame_index": 5,
           "source_file_bytes": stat.st_size, "source_file_mtime_ns": stat.st_mtime_ns}
    (cache / "samples.json").write_text(json.dumps({"samples": [row]}))
    (cache / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "source_mode": "raw", "code_changed_during_run": False}))
    return cache, row, image


def test_raw_cache_reuses_exact_original_pixels_without_registration(tmp_path):
    cache, row, expected = _cache(tmp_path)
    indexed = _raw_cache(cache)
    actual, details = _load_cached_raw(row, indexed[row["sample_id"]])
    assert np.array_equal(actual, expected)
    assert details["source_pixel_sha256"] == pixel_sha256(expected)
    assert details["reused_verified_raw_pixels"] is True
    assert details["registration_reapplied"] is False


def test_raw_cache_rejects_changed_decoded_pixels(tmp_path):
    cache, row, image = _cache(tmp_path)
    indexed = _raw_cache(cache)
    image[0, 0] = (1, 2, 3)
    cv2.imwrite(str(cache / "original.png"), image)
    with pytest.raises(ValueError, match="Cached raw image changed"):
        _load_cached_raw(row, indexed[row["sample_id"]])


@pytest.mark.parametrize("field,value", [("frame_index", "6"), ("source_video", "different_source.bin")])
def test_raw_cache_rejects_a_different_video_or_frame(tmp_path, field, value):
    cache, row, _ = _cache(tmp_path)
    indexed = _raw_cache(cache)
    changed = {**row, field: str(tmp_path / value) if field == "source_video" else value}
    with pytest.raises(ValueError, match="different video/frame"):
        _load_cached_raw(changed, indexed[row["sample_id"]])


def test_raw_cache_rejects_changed_source_file(tmp_path):
    cache, row, _ = _cache(tmp_path)
    indexed = _raw_cache(cache)
    (tmp_path / "synthetic_source.bin").write_bytes(b"different size and identity")
    with pytest.raises(ValueError, match="Source video changed"):
        _load_cached_raw(row, indexed[row["sample_id"]])


@pytest.mark.parametrize("field,value", [
    ("status", "running"), ("status", "completed_with_errors"),
    ("source_mode", "prepared"), ("code_changed_during_run", True),
])
def test_raw_cache_rejects_incomplete_or_non_raw_runs(tmp_path, field, value):
    cache, _, _ = _cache(tmp_path)
    path = cache / "run_manifest.json"
    info = json.loads(path.read_text())
    path.write_text(json.dumps({**info, field: value}))
    with pytest.raises(ValueError, match="completed intact raw-source run"):
        _raw_cache(cache)
