"""Synthetic regression/negative controls, NOT accuracy on real plasmodium."""
import cv2
import numpy as np
import pytest
from segmentation_lab.contracts import RoiSpec
from segmentation_lab.segmenter_round2 import prepare_baseline, segment as r2
from segmentation_lab.segmenter_round3 import segment

ROI=RoiSpec((128,128),112,4)
BASE={'food_exclusion':False,'refine_roi':False}


def blurred_line(amplitude=1.):
    field=np.zeros((256,256),np.float32)
    cv2.line(field,(45,95),(215,145),1.,3)
    field=cv2.GaussianBlur(field,(0,0),3.)
    image=np.clip(np.array([60,64,67])+field[...,None]*amplitude*np.array([-5,7,9]),0,255).astype(np.uint8)
    return image,field>.2


@pytest.mark.parametrize('amplitude',[1.,.65])
def test_weak_blurred_line_recovers_supported_width(amplitude):
    image,truth=blurred_line(amplitude)
    old=r2(image,ROI,base_config=BASE)
    result=segment(image,ROI,base_config=BASE)
    assert old.mask[truth].mean()<.70, 'Reproducer no longer represents the R2 failure'
    # Require a meaningful gain on both signal strengths, not perfect recovery
    # of the arbitrary >.2 blurred-field boundary at the weaker contrast.
    assert result.mask[truth].mean()>.75
    assert result.mask[truth].mean()-old.mask[truth].mean()>.20
    assert (result.mask & ~truth).sum()/max(1,result.mask.sum())<.12


@pytest.mark.parametrize('bgr',[(0,0,0),(18,18,18),(85,85,85),(220,220,220),(60,64,67)])
def test_empty_uniform_background_is_not_forced_to_foreground(bgr):
    image=np.empty((256,256,3),np.uint8);image[:]=bgr
    assert not segment(image,ROI,base_config=BASE).mask.any()


def test_gray_lines_and_reflection_edges_are_not_yellow_tissue():
    image=np.full((256,256,3),60,np.uint8)
    cv2.line(image,(40,100),(210,150),(180,180,180),3)
    cv2.circle(image,(128,128),90,(250,250,250),5)
    assert not segment(image,ROI,base_config=BASE).mask.any()


def test_missing_evidence_gap_is_not_filled():
    image,truth=blurred_line()
    image[:,118:140]=(60,64,67)
    result=segment(image,ROI,base_config=BASE)
    assert result.mask[:,:115].any() and result.mask[:,143:].any()
    assert not result.mask[:,122:136].any()


def test_small_isolated_weak_spot_is_withheld_but_not_called_a_known_false_positive():
    image=np.full((256,256,3),85,np.uint8)
    cv2.circle(image,(160,150),8,(78,90,94),-1)
    old=r2(image,ROI,base_config=BASE);debug={}
    result=segment(image,ROI,base_config=BASE,debug=debug)
    assert old.mask.any(), 'This control must exercise an R2 isolated addition'
    assert not result.mask.any()
    assert debug['withheld_crop'].any()
    assert 'isolated_round2_additions_withheld_review' in result.qc_flags


def test_cache_and_input_immutable_and_same_result():
    image,_=blurred_line();saved=image.copy()
    base=prepare_baseline(image,ROI,BASE)
    arrays={k:getattr(base,k).copy() for k in ['mask','candidate','score','roi_mask','excluded_food','preprocessed']}
    cached=segment(image,ROI,base_config=BASE,baseline=base)
    direct=segment(image,ROI,base_config=BASE)
    assert np.array_equal(cached.mask,direct.mask)
    assert np.array_equal(cached.score,direct.score)
    assert np.array_equal(image,saved)
    for key,value in arrays.items():assert np.array_equal(getattr(base,key),value)
    assert not (base.mask & ~cached.mask).any()
    image[128,128,0]+=1
    with pytest.raises(ValueError,match='Cached baseline does not match'):
        segment(image,ROI,base_config=BASE,baseline=base)


def test_food_body_roi_and_network_hole_are_preserved():
    image=np.full((300,420,3),85,np.uint8);roi=RoiSpec((170,150),105,5)
    cv2.circle(image,(170,150),43,(24,180,200),-1)
    cv2.circle(image,(170,150),16,(18,85,150),-1)
    cv2.circle(image,(190,130),5,(85,85,85),-1)
    cv2.line(image,(205,135),(237,96),(8,36,42),2)
    cv2.circle(image,(360,55),18,(24,180,200),-1)
    base=prepare_baseline(image,roi);result=segment(image,roi,baseline=base)
    assert base.excluded_food.any()
    assert not (result.mask & result.excluded_food).any()
    assert not (result.mask & ~result.roi_mask).any()
    assert not (base.mask & ~result.mask).any()
    assert not result.mask[129:132,189:192].any()
    assert np.array_equal(result.excluded_food,base.excluded_food)


@pytest.mark.parametrize('cfg',[{'unknown':1},{'line_floor':0},{'line_floor':float('nan')},
    {'glare_gray':256},{'orientation_selectivity':1},{'min_component_px':1.1},
    {'seed_floor':.001},{'line_floor':True}])
def test_invalid_config_rejected(cfg):
    with pytest.raises(ValueError):segment(np.full((256,256,3),85,np.uint8),ROI,config=cfg)


def test_score_and_final_counts_describe_final_mask():
    image,_=blurred_line();debug={}
    result=segment(image,ROI,base_config=BASE,debug=debug)
    assert result.diagnostics['occupied_area_px']==int(result.mask.sum())
    assert result.diagnostics['round3_added_px']==int((result.mask&~debug['round2'].mask).sum())
    assert result.diagnostics['round3_removed_px']==int((debug['round2'].mask&~result.mask).sum())
    assert result.diagnostics['temporal_mask_editing'] is False
    assert result.diagnostics['score_interpretation']=='feature_evidence_not_probability'
    result.validate(image.shape[:2])
