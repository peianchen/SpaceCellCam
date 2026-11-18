"""Directional centre/surround evidence for dim branches, independent of time.

R1 body/food decisions are preserved. R2 additions can be withheld when they
are small isolated non-elongated patches. New weak lines require bilateral
colour contrast integrated along a line, not generated or filled-in pixels.
Feature scores are not probabilities; withheld pixels remain review evidence.
"""
from __future__ import annotations

from functools import lru_cache
import cv2
import numpy as np
from scipy import ndimage as ndi

from .contracts import RoiSpec, SegmentationResult
from .segmenter import _components
from .segmenter_round2 import prepare_baseline, segment as segment_round2, _noise_map

DEFAULTS = {
    "line_floor": .0045, "line_noise_multiplier": .8,
    "local_floor": .005, "local_noise_multiplier": 1.,
    "seed_floor": .007, "orientation_selectivity": .12,
    "min_component_px": 20, "min_seed_px": 5, "min_span_px": 28,
    "isolated_span_px": 48, "isolated_elongation": 3.,
    "glare_gray": 150, "glare_saturation": 65, "glare_radial_fraction": .82,
}


def _config(config):
    cfg = dict(DEFAULTS)
    if config:
        if set(config)-set(cfg):
            raise ValueError(f"Unknown round3 parameters: {sorted(set(config)-set(cfg))}")
        cfg.update(config)
    for k,v in cfg.items():
        if isinstance(v,(bool,str)) or not np.isscalar(v) or not np.isfinite(v) or v<=0:
            raise ValueError(f"{k} must be a positive finite number")
    for k in ("min_component_px","min_seed_px","min_span_px","isolated_span_px","glare_gray","glare_saturation"):
        if int(cfg[k])!=cfg[k]:raise ValueError(f"{k} must be an integer")
    for k in ("orientation_selectivity","glare_radial_fraction"):
        if cfg[k]>=1:raise ValueError(f"{k} must be less than 1")
    if cfg['glare_gray']>255 or cfg['glare_saturation']>255:
        raise ValueError('Glare intensity/saturation must be <=255')
    if cfg['seed_floor']<cfg['line_floor']:
        raise ValueError('seed_floor must be >= line_floor')
    return cfg


@lru_cache(maxsize=1)
def _line_kernels():
    bank=[]
    for width in (1.,2.,3.5):
        length=max(6.,width*3)
        rad=int(np.ceil(max(length,width*4)*2.5))
        yy,xx=np.mgrid[-rad:rad+1,-rad:rad+1]
        group=[]
        for angle in np.arange(0,180,15):
            a=np.deg2rad(angle)
            u=xx*np.cos(a)+yy*np.sin(a)
            v=-xx*np.sin(a)+yy*np.cos(a)
            c=np.exp(-.5*((u/length)**2+(v/width)**2))
            left=np.exp(-.5*((u/length)**2+((v-width*3)/width)**2))
            right=np.exp(-.5*((u/length)**2+((v+width*3)/width)**2))
            c/=c.sum();left/=left.sum();right/=right.sum()
            group.append(((c-left).astype(np.float32),(c-right).astype(np.float32)))
        bank.append(group)
    return bank


def _directional_lines(yellow):
    """Compare a centre stripe to BOTH flanks; a step edge fails one flank."""
    strength=np.zeros_like(yellow)
    selectivity=np.zeros_like(yellow)
    for group in _line_kernels():
        maximum=np.zeros_like(yellow);mean=np.zeros_like(yellow)
        for kl,kr in group:
            left=cv2.filter2D(yellow,-1,kl,borderType=cv2.BORDER_REFLECT101)
            right=cv2.filter2D(yellow,-1,kr,borderType=cv2.BORDER_REFLECT101)
            response=np.maximum(0,np.minimum(left,right))
            maximum=np.maximum(maximum,response);mean+=response/len(group)
        chosen=maximum>strength
        selectivity[chosen]=((maximum-mean)/(maximum+1e-8))[chosen]
        strength=np.maximum(strength,maximum)
    return strength,selectivity


def _supported_extras(extra, anchor, cfg):
    """Retain long separated branches, but withhold small round isolated spots."""
    count,labels,stats,_=cv2.connectedComponentsWithStats(extra.astype(np.uint8),8)
    near=cv2.dilate(anchor.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    attached=np.bincount(labels[near].ravel(),minlength=count)>0
    keep=np.zeros(count,bool)
    for i in range(1,count):
        w,h=stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT]
        if attached[i] or np.hypot(w,h)>=cfg['isolated_span_px']:
            keep[i]=True;continue
        if np.hypot(w,h)<cfg['min_span_px']:continue
        x,y,w,h,area=stats[i]
        yy,xx=np.nonzero(labels[y:y+h,x:x+w]==i)
        cov=np.cov(np.stack([xx,yy])) if area>=3 else np.zeros((2,2))
        eig=np.linalg.eigvalsh(cov)
        keep[i]=eig[1]/max(.25,eig[0])>=cfg['isolated_elongation']
    return keep[labels]


def segment(image_bgr: np.ndarray, roi: RoiSpec, config: dict|None=None,
            base_config: dict|None=None, baseline: SegmentationResult|None=None,
            debug: dict|None=None) -> SegmentationResult:
    cfg=_config(config)
    # R2 validates input, ROI and the source-bound R1 cache before use.
    if not isinstance(image_bgr,np.ndarray) or image_bgr.dtype!=np.uint8 or image_bgr.ndim!=3 or image_bgr.shape[2]!=3:
        raise ValueError('Expected uint8 BGR image')
    if not np.isfinite([*roi.center,roi.radius,roi.margin_px]).all() or roi.radius<=roi.margin_px or roi.margin_px<0:
        raise ValueError('Invalid circular ROI')
    if not roi.mask(image_bgr.shape[:2]).any():raise ValueError('ROI does not intersect image')
    base=baseline if baseline is not None else prepare_baseline(image_bgr,roi,base_config)
    old=segment_round2(image_bgr,roi,base_config=base_config,baseline=base)
    yy,xx=np.nonzero(base.roi_mask)
    y0,y1=max(0,yy.min()-64),min(image_bgr.shape[0],yy.max()+65)
    x0,x1=max(0,xx.min()-64),min(image_bgr.shape[1],xx.max()+65)
    sl=np.s_[y0:y1,x0:x1]
    crop=image_bgr[sl];b,g,r=cv2.split(crop.astype(np.float32)/255.)
    yellow=(np.minimum(r,g)-b)/(np.minimum(r,g)+b+.08)
    usable=base.roi_mask[sl]&~base.excluded_food[sl]&(crop.max(axis=2)>12)&(crop.min(axis=2)<250)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    chromatic=(hsv[...,0]>=10)&(hsv[...,0]<=58)&(hsv[...,1]>=8)
    noise=_noise_map(yellow,usable,64,.003)
    smooth=cv2.GaussianBlur(yellow,(0,0),.7)
    strength,selectivity=_directional_lines(smooth)
    local=smooth-cv2.GaussianBlur(yellow,(0,0),12.)
    threshold=np.maximum(cfg['line_floor'],noise*cfg['line_noise_multiplier'])
    candidate=usable&chromatic&(local>np.maximum(cfg['local_floor'],noise*cfg['local_noise_multiplier']))&(strength>threshold)&(selectivity>cfg['orientation_selectivity'])
    seed=candidate&(strength>np.maximum(cfg['seed_floor'],threshold*1.4))
    branches=_components(candidate,seed,int(cfg['min_component_px']),int(cfg['min_seed_px']))
    count,labels,stats,_=cv2.connectedComponentsWithStats(branches.astype(np.uint8),8)
    keep=np.hypot(stats[:,cv2.CC_STAT_WIDTH],stats[:,cv2.CC_STAT_HEIGHT])>=cfg['min_span_px'];keep[0]=False
    branches=keep[labels]
    gy,gx=np.ogrid[y0:y1,x0:x1]
    radial=np.hypot(gx-roi.center[0],gy-roi.center[1])/roi.radius
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    glare=(radial>cfg['glare_radial_fraction'])&(gray>cfg['glare_gray'])&(hsv[...,1]<cfg['glare_saturation'])
    # Withhold only newly proposed R3 pixels in this ambiguous bright rim zone.
    proposed=branches&~old.mask[sl]
    review_glare=proposed&glare
    extras=(old.mask[sl]|(branches&~glare))&~base.mask[sl]
    supported=_supported_extras(extras,base.mask[sl],cfg)
    removed_small=extras&~supported
    mask=base.mask.copy();mask[sl]|=supported
    added=mask&~old.mask;removed=old.mask&~mask
    all_candidate=old.candidate.copy();all_candidate[sl]|=candidate
    score=old.score.copy();score[sl]=np.maximum(score[sl],np.where(usable,np.clip(strength/(2*threshold),0,1),0))
    flags=[f for f in old.qc_flags if f not in ('empty_no_supported_tissue','tissue_near_roi_boundary','large_coverage_check_background')]
    if added.any():flags.append('directional_line_additions_require_review')
    if removed.any():flags.append('isolated_round2_additions_withheld_review')
    if review_glare.any():flags.append('bright_rim_candidates_withheld_review')
    if not mask.any():flags.append('empty_no_supported_tissue')
    border=base.roi_mask&~cv2.erode(base.roi_mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))).astype(bool)
    rim_fraction=float((mask&border).sum()/max(1,mask.sum()))
    if (mask&border).any():flags.append('tissue_near_roi_boundary')
    fraction=float(mask.sum()/base.roi_mask.sum())
    if fraction>.65:flags.append('large_coverage_check_background')
    diag={**old.diagnostics,'round3_parameters':dict(cfg),'base_method':old.method,'round2_area_px':int(old.mask.sum()),'round3_added_px':int(added.sum()),'round3_removed_px':int(removed.sum()),'occupied_area_px':int(mask.sum()),'clean_area_px':int(mask.sum()),'envelope_area_px':int(ndi.binary_fill_holes(mask).sum()),'area_fraction':fraction,'rim_fraction':rim_fraction,'line_candidate_px':int(branches.sum()),'withheld_bright_rim_px':int(review_glare.sum()),'withheld_isolated_extra_px':int(removed_small.sum()),'temporal_mask_editing':False,'score_interpretation':'feature_evidence_not_probability','quality_interpretation':'heuristic_not_accuracy','branch_method':'bilateral_directional_color_contrast_with_explicit_abstention'}
    # Do not label inherited R2-only counts as counts of the R3 result.
    for key in ('added_branch_area_px','line_seed_area_px','local_noise_median','local_noise_p90'):
        if key in diag:diag['round2_'+key]=diag.pop(key)
    diag['round1_area_px']=diag.pop('base_area_px')
    diag['quality_score']=float(np.clip(1-.2*('heterogeneous_background' in flags)-.2*('glare_in_roi' in flags)-min(.3,rim_fraction*2),0,1))
    result=SegmentationResult('directional_ridge_round3',mask,base.preprocessed.copy(),score,all_candidate,base.roi_mask.copy(),base.excluded_food.copy(),list(dict.fromkeys(flags)),diag)
    result.validate(image_bgr.shape[:2])
    if debug is not None:
        debug.update({'round2':old,'crop_xyxy':[int(x0),int(y0),int(x1),int(y1)],'directional_strength':strength,'orientation_selectivity':selectivity,'withheld_crop':review_glare|removed_small,'line_candidate_crop':branches})
    return result
