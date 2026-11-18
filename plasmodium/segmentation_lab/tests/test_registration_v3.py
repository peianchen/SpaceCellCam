"""Spatial consistency is necessary even when global point error decreases."""
import json
import cv2
import numpy as np
import pytest

from segmentation_lab.contracts import RoiSpec
from segmentation_lab.registration_v3 import estimate_motion, regional_evidence


def scene():
    size = 384
    roi = RoiSpec((192, 192), 110, 5)
    yy, xx = np.ogrid[:size, :size]
    radius = np.hypot(xx-192, yy-192)
    ring = (radius > 1.12*110) & (radius < 1.65*110)
    rng = np.random.default_rng(155)
    texture = cv2.GaussianBlur(rng.integers(30, 220, (size, size), np.uint8), (3, 3), 0)
    gray = np.full((size, size), 80, np.uint8)
    gray[ring] = texture[ring]
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image[170:210, 170:215] = (25, 170, 190)
    return image, roi


@pytest.mark.parametrize("angle,translation", [(0, (.6, -.4)), (.6, (3, -2))])
def test_trustworthy_small_translation_and_rotation_are_still_applied(angle, translation):
    reference, roi = scene()
    forward = cv2.getRotationMatrix2D(roi.center, angle, 1)
    forward[:, 2] += translation
    moving = cv2.warpAffine(reference, forward, (384, 384), borderValue=(80, 80, 80))
    originals = reference.copy(), moving.copy()
    matrix, info = estimate_motion(reference, moving, roi)
    assert info['valid'] and info['applied'], info
    assert np.max(np.abs(matrix-cv2.invertAffineTransform(forward))) < .15
    assert abs(np.linalg.det(matrix[:, :2])-1) < 1e-5
    assert np.array_equal(reference, originals[0]) and np.array_equal(moving, originals[1])
    assert info['biological_plane_correction_verified'] is False
    json.dumps(info, allow_nan=False)


def test_interior_growth_cannot_drive_or_force_a_motion():
    reference, roi = scene()
    moving = reference.copy()
    moving[140:245, 150:230] = (20, 180, 200)
    matrix, info = estimate_motion(reference, moving, roi)
    assert info['valid'] and not info['applied']
    assert np.array_equal(matrix, np.float32([[1, 0, 0], [0, 1, 0]]))


def test_r3_rejection_returns_identity_and_null_applied_metrics():
    reference, roi = scene()
    moving = cv2.warpAffine(reference, np.float32([[1, 0, 3], [0, 1, -2]]), (384, 384))
    matrix, info = estimate_motion(reference, moving, roi, {"min_motion_to_residual": 10000})
    assert info['upstream_valid'] and not info['valid'] and not info['applied']
    assert info['reason'] == 'motion_not_resolved_above_spatial_residual'
    assert info['candidate_background_residual_after_median_px'] is not None
    assert info['background_residual_after_median_px'] is None
    assert info['heldout_background_ncc_after'] is None
    assert np.array_equal(matrix, np.float32([[1, 0, 0], [0, 1, 0]]))


def test_blank_image_is_invalid_not_checked_identity():
    image = np.full((256, 256, 3), 80, np.uint8)
    matrix, info = estimate_motion(image, image, RoiSpec((128, 128), 75))
    assert not info['valid'] and info['background_residual_after_median_px'] is None
    assert np.array_equal(matrix, np.float32([[1, 0, 0], [0, 1, 0]]))


def test_inconsistent_background_regions_are_not_warped_to_agree():
    reference, roi = scene()
    left = cv2.warpAffine(reference, np.float32([[1, 0, 4], [0, 1, 0]]), (384, 384))
    right = cv2.warpAffine(reference, np.float32([[1, 0, -4], [0, 1, 0]]), (384, 384))
    moving = reference.copy()
    moving[:, :192] = left[:, :192]
    moving[:, 192:] = right[:, 192:]
    matrix, info = estimate_motion(reference, moving, roi)
    assert not info['valid']
    assert np.array_equal(matrix, np.float32([[1, 0, 0], [0, 1, 0]]))


def point_ring():
    angles = np.linspace(.05, 2*np.pi-.05, 128)
    points = np.column_stack([np.cos(angles), np.sin(angles)])*160+192
    return points, RoiSpec((192, 192), 110, 5)


def test_global_majority_improvement_does_not_hide_a_stationary_sector():
    a, roi = point_ring()
    b = a+(2, 0)
    # Seven regions moved, one truly did not. A global translation improves
    # median residual to zero while creating a 2 px error in the stationary area.
    b[:16] = a[:16]
    m = np.float32([[1, 0, -2], [0, 1, 0]])
    ev = regional_evidence(m, a, b, roi)
    assert ev['residual_median_px'] == 0
    assert ev['worsened_regions'] == [0]
    assert ev['regions'][0]['regression_px'] == 2


def test_unresolved_motion_is_distinct_from_a_precise_small_motion():
    a, roi = point_ring()
    m = np.float32([[1, 0, -.5], [0, 1, 0]])
    noisy = a+(.5, 0)
    noisy[:, 1] += np.where(np.arange(len(a)) % 2, .4, -.4)
    ev = regional_evidence(m, a, noisy, roi)
    assert ev['motion_to_residual'] < 2
    clear = regional_evidence(m, a, a+(.5, 0), roi)
    assert clear['motion_to_residual'] >= 2


def test_wrong_rotation_model_cannot_score_as_true_translation():
    a, roi = point_ring()
    b = a+(3, -2)
    wrong = cv2.getRotationMatrix2D(roi.center, 1.5, 1)
    ev = regional_evidence(wrong, a, b, roi)
    assert ev['worsened_regions']
    correct = regional_evidence(np.float32([[1, 0, -3], [0, 1, 2]]), a, b, roi)
    assert not correct['worsened_regions']


@pytest.mark.parametrize('config', [{'min_motion_to_residual':0}, {'min_sector_points':2},
                                   {'min_evaluated_sectors':9}, {'min_sector_points':3.2},
                                   {'min_motion_to_residual':True}, {'unknown':1}])
def test_gate_configuration_is_explicit(config):
    reference, roi = scene()
    with pytest.raises(ValueError):
        estimate_motion(reference, reference, roi, config)
