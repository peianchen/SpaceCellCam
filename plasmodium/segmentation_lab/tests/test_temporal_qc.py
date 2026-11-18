"""Diagnostic alignment must not edit masks or manufacture valid results."""
import json

import cv2
import numpy as np
import pytest

from segmentation_lab.contracts import RoiSpec
from segmentation_lab.segmenter import segment
from segmentation_lab.temporal_qc import analyze_pair, run_window
from segmentation_lab import temporal_qc


def _scene(size=256):
    roi = RoiSpec((size/2, size/2), size*.37, 5)
    yy, xx = np.ogrid[:size, :size]
    radius = np.hypot(xx-roi.center[0], yy-roi.center[1])
    rim = (radius > roi.radius*.76) & (radius < roi.radius*1.15)
    rng = np.random.default_rng(164)
    gray = np.full((size, size), 70, np.uint8)
    texture = cv2.GaussianBlur(rng.integers(20, 240, (size, size), np.uint8), (3, 3), 0)
    gray[rim] = texture[rim]
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = np.zeros((size, size), bool)
    mask[size//2-18:size//2+18, size//2-25:size//2+25] = True
    image[mask] = (30, 165, 190)
    return image, mask, roi


def test_rigid_translation_and_small_rotation_improve_only_diagnostics():
    image, mask, roi = _scene()
    matrix = cv2.getRotationMatrix2D(roi.center, .8, 1)
    matrix[:, 2] += (4, -3)
    moved_image = cv2.warpAffine(image, matrix, (256, 256), borderValue=(70, 70, 70))
    moved_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (256, 256), flags=cv2.INTER_NEAREST).astype(bool)
    saved = [array.copy() for array in (image, moved_image, mask, moved_mask)]
    result = analyze_pair(image, moved_image, mask, moved_mask, roi)
    assert result["registration_valid"], result
    assert result["registered_iou"] > .95
    assert result["registered_iou"] > result["raw_iou"] + .10
    assert result["same_domain_registered_iou"] > result["same_domain_raw_iou"] + .10
    assert result["same_domain_area_px"] > 0
    assert result["registered_changed_fraction"] < result["raw_changed_fraction"]
    assert result["temporal_mask_editing"] is False
    assert all(np.array_equal(a, b) for a, b in zip(saved, (image, moved_image, mask, moved_mask)))
    json.dumps(result, allow_nan=False)


def test_real_growth_remains_after_rim_registration():
    image, mask, roi = _scene()
    grown_mask = mask.copy()
    grown_mask[120:136, 153:174] = True
    grown_image = image.copy()
    grown_image[grown_mask] = (30, 165, 190)
    result = analyze_pair(image, grown_image, mask, grown_mask, roi, area_jump_fraction=.05)
    assert result["registration_valid"], result
    assert result["registered_new_area_px"] > 250
    assert result["registered_iou"] < .90
    assert result["area_jump_flag"]
    assert np.array_equal(grown_mask[120:136, 153:174], np.ones((16, 21), bool))


def test_rigid_translation_across_roi_edge_is_not_false_retraction():
    image, _, roi = _scene()
    full = np.zeros(image.shape[:2], bool)
    full[116:140, 198:229] = True
    image[full] = (30, 165, 190)
    motion = np.float32([[1, 0, 4], [0, 1, 0]])
    moved = cv2.warpAffine(image, motion, (256, 256), borderValue=(70, 70, 70))
    valid_roi = roi.mask(full.shape)
    before = full & valid_roi
    after = cv2.warpAffine(full.astype(np.uint8), motion, (256, 256),
                           flags=cv2.INTER_NEAREST).astype(bool) & valid_roi
    before_saved, after_saved = before.copy(), after.copy()
    result = analyze_pair(image, moved, before, after, roi)
    assert result["registration_valid"], result
    assert result["registered_comparison_valid"]
    assert result["raw_retracted_area_px"] > 0
    assert result["registered_iou"] == 1
    assert result["registered_retracted_area_px"] == 0
    assert result["registered_new_area_px"] == 0
    assert result["registered_common_fov_fraction_of_previous_roi"] < 1
    assert np.array_equal(before, before_saved) and np.array_equal(after, after_saved)


def test_explicit_different_roi_masks_restrict_comparison(monkeypatch):
    image, mask, roi = _scene()
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    monkeypatch.setattr(temporal_qc, "_estimate_rim_registration", lambda *args: (
        identity, {"registration_valid": True, "registration_fallback_reason": ""}))
    previous_roi = roi.mask(mask.shape)
    current_roi = previous_roi.copy()
    current_roi[:, 128:] = False
    result = analyze_pair(image, image, mask, mask & current_roi, roi,
                          previous_roi_mask=previous_roi, current_roi_mask=current_roi)
    assert result["registered_iou"] == 1
    assert result["registered_retracted_area_px"] == 0
    # ROI-only cropping must not be presented as an alignment improvement:
    # global raw IoU differs from common-ROI IoU even for an identity transform.
    assert result["raw_iou"] < result["registered_iou"]
    assert result["same_domain_raw_iou"] == result["same_domain_registered_iou"] == 1
    assert result["same_domain_area_px"] == int((previous_roi & current_roi).sum())
    empty = analyze_pair(image, image, mask, mask, roi,
                         previous_roi_mask=previous_roi, current_roi_mask=~previous_roi)
    assert empty["registration_valid"]
    assert not empty["registered_comparison_valid"]
    assert empty["registered_comparison_reason"] == "no_common_valid_analysis_roi"
    assert empty["registered_iou"] is None
    assert empty["registered_retracted_area_px"] is None
    assert empty["same_domain_raw_iou"] is None
    assert empty["same_domain_registered_iou"] is None
    assert empty["same_domain_area_px"] == 0


def test_failed_registration_leaves_registered_metrics_unknown():
    image = np.full((128, 128, 3), 100, np.uint8)
    before = np.zeros((128, 128), bool)
    before[40:70, 40:70] = True
    after = np.roll(before, 3, axis=1)
    result = analyze_pair(image, image, before, after, RoiSpec((64, 64), 48))
    assert not result["registration_valid"]
    assert result["registration_fallback_reason"] == "insufficient_rim_texture"
    assert result["raw_iou"] < 1
    for key in ("registered_iou", "registered_changed_fraction", "registered_new_area_px", "registered_retracted_area_px"):
        assert result[key] is None
    for key in ("same_domain_raw_iou", "same_domain_registered_iou", "same_domain_area_px"):
        assert result[key] is None
    json.dumps(result, allow_nan=False)


def test_motion_outside_limit_is_not_treated_as_success():
    image, mask, roi = _scene()
    matrix = np.float32([[1, 0, 8], [0, 1, 0]])
    moved = cv2.warpAffine(image, matrix, (256, 256))
    moved_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (256, 256)).astype(bool)
    result = analyze_pair(image, moved, mask, moved_mask, roi,
                          registration_config={"max_center_shift_px": 2})
    assert not result["registration_valid"]
    assert result["registered_iou"] is None


def test_pair_rejects_shape_or_config_errors():
    image, mask, roi = _scene()
    with pytest.raises(ValueError, match="matching"):
        analyze_pair(image, image[:-1], mask, mask, roi)
    with pytest.raises(ValueError, match="Unknown registration"):
        analyze_pair(image, image, mask, mask, roi, registration_config={"typo": 2})


def test_window_writes_original_masks_and_keeps_unknown_time(tmp_path):
    image, _, roi = _scene(128)
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (128, 128))
    assert writer.isOpened()
    for shift in range(3):
        writer.write(cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, 0]]), (128, 128)))
    writer.release()
    source_bytes = video.read_bytes()
    output = tmp_path / "qc"
    manifest = run_window(video, output, count=3, roi=roi,
                           segmentation_config={"food_exclusion": False})
    assert manifest["status"] == "complete"
    assert manifest["completed_count"] == 3
    assert manifest["acquisition_interval_sec"] is None
    assert not manifest["biological_time_known"]
    assert video.read_bytes() == source_bytes
    assert (output / "joint_contact.jpg").exists()
    details = json.loads((output / "frame_000001" / "qc.json").read_text(encoding="utf-8"))
    assert details["frame"]["time_from_video_frame0_sec"] is None
    assert details["temporal_mask_editing"] is False
    assert "acquisition_interval_unknown" in details["qc_flags"]
    # Decode the same source frame and independently segment it: the saved mask
    # must be that exact spatial result, never the aligned diagnostic mask.
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok, decoded = cap.read()
    cap.release()
    assert ok
    expected = segment(decoded, roi, config={"food_exclusion": False}).mask
    saved_mask = cv2.imdecode(np.fromfile(str(output / "frame_000001" / "mask.png"), np.uint8),
                              cv2.IMREAD_GRAYSCALE).astype(bool)
    assert np.array_equal(saved_mask, expected)
    with pytest.raises(FileExistsError):
        run_window(video, output, count=1, roi=roi)


def test_decode_failure_retains_partial_outputs_and_failed_status(tmp_path):
    image, _, roi = _scene(128)
    video = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (128, 128))
    assert writer.isOpened()
    writer.write(image)
    writer.release()
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="required source frame 1"):
        run_window(video, output, count=2, roi=roi, acquisition_interval_sec=10)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["completed_count"] == 1
    frame = json.loads((output / "frame_000000" / "qc.json").read_text(encoding="utf-8"))["frame"]
    assert frame["acquisition_interval_sec"] == 10
    assert frame["sample_interval_sec"] == 10
    assert frame["registered_iou"] is None
    assert frame["raw_iou"] is None
    assert frame["same_domain_raw_iou"] is None
    assert frame["same_domain_registered_iou"] is None
    assert frame["same_domain_area_px"] is None
    assert (output / "diagnostics.csv").exists()


@pytest.mark.parametrize("seek_ok,decoded_index,message", [
    (False, 0, "Could not seek"), (True, 9, "Decoder returned frame"),
    (True, float("nan"), "Decoder returned frame"),
])
def test_seek_or_frame_identity_failure_cannot_mislabel_output(
        tmp_path, monkeypatch, seek_ok, decoded_index, message):
    image, _, roi = _scene(128)
    video = tmp_path / "fake.avi"
    video.write_bytes(b"synthetic source placeholder")

    class FakeCapture:
        def isOpened(self): return True
        def release(self): pass
        def set(self, prop, value): return seek_ok
        def read(self): return True, image
        def get(self, prop):
            return decoded_index + 1 if prop == cv2.CAP_PROP_POS_FRAMES else 30

    monkeypatch.setattr(temporal_qc.cv2, "VideoCapture", lambda path: FakeCapture())
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match=message):
        run_window(video, output, start=4, count=1, roi=roi)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["completed_count"] == 0
    assert not list(output.glob("frame_*"))
