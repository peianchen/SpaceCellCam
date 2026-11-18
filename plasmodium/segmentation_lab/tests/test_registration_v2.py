"""Rigid registration needs independent static evidence, not tissue overlap."""
import json

import cv2
import numpy as np
import pytest

from segmentation_lab.contracts import RoiSpec
from segmentation_lab.registration_v2 import estimate_motion, evaluate_motion


def scene():
    size=384;roi=RoiSpec((192,192),110,5)
    y,x=np.ogrid[:size,:size];r=np.hypot(x-192,y-192)
    ring=(r>1.12*110)&(r<1.65*110)
    rng=np.random.default_rng(155)
    textured=cv2.GaussianBlur(rng.integers(30,220,(size,size),np.uint8),(3,3),0)
    gray=np.full((size,size),80,np.uint8);gray[ring]=textured[ring]
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    image[170:210,170:215]=(25,170,190)
    return image,roi


def test_known_rigid_motion_passes_independent_points_and_preserves_inputs():
    reference,roi=scene()
    forward=cv2.getRotationMatrix2D(roi.center,.6,1);forward[:,2]+=(3,-2)
    moving=cv2.warpAffine(reference,forward,(384,384),borderValue=(80,80,80))
    original_ref,original_moving=reference.copy(),moving.copy()
    matrix,info=estimate_motion(reference,moving,roi)
    assert info['valid'],info
    assert info['applied']
    assert info['holdout_matches']>=8
    assert info['background_residual_after_median_px']<.2
    assert info['background_residual_after_median_px']<info['background_residual_before_median_px']/4
    assert abs(np.linalg.det(matrix[:,:2])-1)<1e-5
    assert np.allclose(matrix,cv2.invertAffineTransform(forward),atol=.15)
    assert np.array_equal(reference,original_ref) and np.array_equal(moving,original_moving)
    assert all(np.linalg.norm(np.array(p)-roi.center)>roi.radius for p in info['heldout_reference_points'])
    json.dumps(info,allow_nan=False)


def test_interior_growth_does_not_steer_motion_estimate():
    reference,roi=scene();moving=reference.copy()
    moving[140:245,150:230]=(20,180,200)
    matrix,info=estimate_motion(reference,moving,roi)
    assert info['valid'],info
    assert not info['applied']
    assert info['reason']=='motion_below_resolution_identity'
    assert np.array_equal(matrix,np.float32([[1,0,0],[0,1,0]]))


def test_blank_background_is_rejected_not_successful_identity():
    reference=np.full((256,256,3),80,np.uint8)
    matrix,info=estimate_motion(reference,reference,RoiSpec((128,128),75))
    assert not info['valid'] and not info['applied']
    assert info['background_residual_after_median_px'] is None
    assert np.array_equal(matrix,np.float32([[1,0,0],[0,1,0]]))
    json.dumps(info,allow_nan=False)


def test_motion_outside_allowed_range_keeps_candidate_separate():
    reference,roi=scene()
    moving=cv2.warpAffine(reference,np.float32([[1,0,4],[0,1,0]]),(384,384))
    matrix,info=estimate_motion(reference,moving,roi,{'max_center_shift_px':1})
    assert not info['valid']
    assert info['reason']=='motion_outside_allowed_rigid_range'
    assert info['background_residual_after_median_px'] is None
    assert info['candidate_background_residual_after_median_px']<.5
    assert np.array_equal(matrix,np.float32([[1,0,0],[0,1,0]]))


def test_inconsistent_background_regions_are_not_nonrigidly_repaired():
    reference,roi=scene();moving=reference.copy()
    left=cv2.warpAffine(reference,np.float32([[1,0,4],[0,1,0]]),(384,384))
    right=cv2.warpAffine(reference,np.float32([[1,0,-4],[0,1,0]]),(384,384))
    moving[:,:192]=left[:,:192];moving[:,192:]=right[:,192:]
    matrix,info=estimate_motion(reference,moving,roi)
    assert not info['valid'],info
    assert np.array_equal(matrix,np.float32([[1,0,0],[0,1,0]]))


def test_residual_evaluation_uses_supplied_points_without_refitting():
    a=np.float32([[0,0],[10,0],[0,10]])
    b=a+(3,4)
    residual=evaluate_motion(np.float32([[1,0,-3],[0,1,-4]]),a,b)
    assert residual['background_residual_before_median_px']==5
    assert residual['background_residual_after_median_px']==0


def test_bad_shape_or_configuration_raises():
    reference,roi=scene()
    with pytest.raises(ValueError,match='matching'):
        estimate_motion(reference,reference[:20],roi)
    with pytest.raises(ValueError,match='Unknown'):
        estimate_motion(reference,reference,roi,{'typo':1})
    with pytest.raises(ValueError,match='geometry'):
        estimate_motion(reference,reference,roi,{'inner_radius_ratio':.7})
