"""Independent-frame Round 3 segmentation plus conservative motion diagnostics.

python -m segmentation_lab.round3_sequence --video VIDEO --start 100 --count 8
  --step 1 --roi-config segmentation_lab/config_round1_v2.json
  --video-id rotating_02_b3ec4905 --output NEW_DIRECTORY

No temporal mask edits; no playback-FPS substitution for acquisition time.
Outer candidate-static feature agreement does not verify the biological plane.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .contracts import RoiSpec
from .registration_v3 import DEFAULTS as MOTION_DEFAULTS, estimate_motion
from .segmenter_round3 import DEFAULTS as SEGMENT_DEFAULTS, segment
from .visualize import mask_overlay, write_image


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def _warp(mask, matrix):
    return cv2.warpAffine(mask.astype(np.uint8),matrix,(mask.shape[1],mask.shape[0]),
                          flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).astype(bool)


def _empty_overlap():
    return {name:None for name in ("same_domain_area_px","same_domain_raw_iou",
        "same_domain_registered_iou","registered_new_area_px","registered_retracted_area_px",
        "registered_changed_fraction")}


def _same_domain_overlap(previous,current,matrix):
    domain=(previous.roi_mask & current.roi_mask & _warp(current.roi_mask,matrix)
            & _warp(np.ones(current.mask.shape,bool),matrix))
    if not domain.any():
        return {**_empty_overlap(),"same_domain_area_px":0}
    before=previous.mask & domain
    unaligned=current.mask & domain
    aligned=_warp(current.mask,matrix) & domain
    def iou(a,b):
        union=int((a|b).sum())
        return float((a&b).sum()/union) if union else None
    new=int((aligned&~before).sum());retracted=int((before&~aligned).sum())
    union=int((aligned|before).sum())
    return dict(same_domain_area_px=int(domain.sum()),same_domain_raw_iou=iou(before,unaligned),
                same_domain_registered_iou=iou(before,aligned),registered_new_area_px=new,
                registered_retracted_area_px=retracted,
                registered_changed_fraction=(new+retracted)/union if union else None)


def run_sequence(video,output,*,roi_config,video_id,start=0,count=8,step=1,
                 acquisition_interval_sec=None,segmentation_config=None,registration_config=None):
    """Write a fresh bounded sequence run; reject unsupported frame identity."""
    video,output,roi_config=Path(video).resolve(),Path(output).resolve(),Path(roi_config).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if any(isinstance(v,bool) or int(v)!=v for v in (start,count,step)) or start<0 or count<1 or step<1:
        raise ValueError("start>=0, count>=1, step>=1 must be integers")
    if acquisition_interval_sec is not None and (not np.isfinite(acquisition_interval_sec) or acquisition_interval_sec<=0):
        raise ValueError("Acquisition interval must be positive or omitted")
    cfg=json.loads(roi_config.read_text(encoding="utf-8-sig"))
    if video_id not in cfg.get("new_method_roi_by_video",{}):
        raise ValueError(f"No video-specific ROI for {video_id!r}")
    geometry={**cfg.get("roi",{}),**cfg["new_method_roi_by_video"][video_id]}
    roi=RoiSpec(tuple(geometry["center"]),float(geometry["radius"]),float(geometry.get("margin_px",8)))
    base_config={"refine_roi":True,**cfg.get("methods",{}).get("adaptive_chroma",{})}
    segmentation_config=dict(segmentation_config or {})
    registration_config=dict(registration_config or {})
    if set(segmentation_config)-set(SEGMENT_DEFAULTS) or set(registration_config)-set(MOTION_DEFAULTS):
        raise ValueError("Unknown segmentation or registration configuration key")
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release();raise ValueError(f"Could not open video {video}")
    try:
        output.mkdir(parents=True,exist_ok=False)
    except BaseException:
        cap.release();raise
    here=Path(__file__).resolve().parent
    sources=[here/name for name in ("round3_sequence.py","segmenter_round3.py","segmenter.py",
                                    "registration_v3.py","registration_v2.py","segmenter_round2.py","contracts.py","visualize.py")]
    stat=video.stat()
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),
      video=str(video),video_id=video_id,source_size_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
      roi=asdict(roi),roi_config=str(roi_config),roi_config_sha256=_sha(roi_config),
      segmentation_parameters={**SEGMENT_DEFAULTS,**segmentation_config},base_config=base_config,
      registration_parameters={**MOTION_DEFAULTS,**registration_config},
      start_frame=int(start),requested_count=int(count),step=int(step),
      acquisition_interval_sec=acquisition_interval_sec,biological_time_known=acquisition_interval_sec is not None,
      playback_fps_metadata_only=fps if np.isfinite(fps) else None,
      temporal_mask_editing=False,biological_plane_correction_verified=False,
      registration_interpretation="candidate_static_outer_feature_alignment_only",
      comparison_domain="previous_roi & current_roi & warp(current_roi) & warp_frame_support",
      untrusted_pair_overlap_metrics="null",
      code_files=[dict(path=str(p),sha256=_sha(p)) for p in sources],
      versions=dict(python=sys.version.split()[0],opencv=cv2.__version__,numpy=np.__version__))
    rows=[];previous_image=previous_result=None;error=None
    try:
        _save_json(output/"run_manifest.json",manifest)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES,int(start)):
            raise ValueError(f"Could not seek required frame {start}")
        for sample in range(int(count)):
            index=int(start+sample*step)
            ok,image=cap.read()
            if not ok or image is None:
                raise ValueError(f"Could not decode required frame {index}")
            decoded=float(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            if not np.isfinite(decoded) or abs(decoded-index)>.5:
                raise ValueError(f"Decoded frame {decoded}, expected {index}")
            raw_hash=hashlib.sha256(image.tobytes()).hexdigest()
            result=segment(image,roi,config=segmentation_config,base_config=base_config)
            result.validate(image.shape[:2])
            overlap=_empty_overlap();pair=None
            if previous_image is not None:
                matrix,diagnostics=estimate_motion(previous_image,image,roi,registration_config)
                if diagnostics["valid"]:
                    overlap=_same_domain_overlap(previous_result,result,matrix)
                pair=dict(previous_frame=index-step,frame=index,matrix=matrix.tolist(),
                          diagnostics=diagnostics,overlap=overlap,
                          biological_plane_correction_verified=False,temporal_mask_editing=False)
            if hashlib.sha256(image.tobytes()).hexdigest()!=raw_hash:
                raise RuntimeError("Source pixels were modified during processing")
            folder=output/f"frame_{index:06d}";folder.mkdir()
            write_image(folder/"source.png",image)
            write_image(folder/"mask.png",result.mask.astype(np.uint8)*255)
            write_image(folder/"overlay.jpg",mask_overlay(image,result.mask,excluded_food=result.excluded_food))
            write_image(folder/"roi_mask.png",result.roi_mask.astype(np.uint8)*255)
            flags=list(result.qc_flags)
            if acquisition_interval_sec is None:
                flags.append("acquisition_interval_unknown")
            if pair is not None and not pair["diagnostics"]["valid"]:
                flags.append("motion_untrusted_overlap_metrics_unavailable")
            diag=pair["diagnostics"] if pair else {}
            row=dict(frame_index=index,decoded_frame_index=decoded,
                previous_frame_index=index-step if pair else None,
                time_from_source_frame0_sec=index*acquisition_interval_sec if acquisition_interval_sec is not None else None,
                acquisition_interval_sec=acquisition_interval_sec,
                sample_interval_sec=step*acquisition_interval_sec if acquisition_interval_sec is not None else None,
                occupied_area_px=int(result.mask.sum()),
                occupied_fraction_of_roi=float(result.mask.sum()/max(1,result.roi_mask.sum())),
                food_excluded_px=int(result.excluded_food.sum()),
                source_frame_sha256=raw_hash,mask_sha256=hashlib.sha256(result.mask.tobytes()).hexdigest(),
                registration_valid=diag.get("valid",False),registration_applied=diag.get("applied",False),
                registration_reason=diag.get("reason","no_previous_sample"),
                registration_matches=diag.get("matches"),registration_holdout_matches=diag.get("holdout_matches"),
                candidate_center_shift_px=diag.get("center_shift_px"),candidate_rotation_deg=diag.get("rotation_deg"),
                background_residual_before_median_px=diag.get("background_residual_before_median_px"),
                background_residual_after_median_px=diag.get("background_residual_after_median_px"),
                temporal_mask_editing=False,biological_plane_correction_verified=False,
                qc_flags=json.dumps(flags,ensure_ascii=False),**overlap)
            _save_json(folder/"qc.json",dict(frame=row,segmentation=result.diagnostics,qc_flags=flags,pair=pair))
            if pair:
                _save_json(folder/"pair_motion.json",pair)
            rows.append(row)
            previous_image,previous_result=image,result
            if sample+1<count:
                for skipped in range(step-1):
                    if not cap.grab():
                        raise ValueError(f"Could not advance required frame {index+skipped+1}")
    except BaseException as exc:
        error=exc;manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        cap.release()
        if rows:
            with (output/"diagnostics.csv").open("w",newline="",encoding="utf-8-sig") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        changed=[p["path"] for p in manifest["code_files"] if _sha(p["path"])!=p["sha256"]]
        manifest.update(completed_count=len(rows),finished_utc=datetime.now(timezone.utc).isoformat(),
                        code_files_changed_during_run=changed)
        if error is None:
            manifest["status"]="complete" if not changed else "complete_source_changed_review_required"
        _save_json(output/"run_manifest.json",manifest)
    if error is not None:
        raise error
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--count",type=int,default=8)
    p.add_argument("--step",type=int,default=1)
    p.add_argument("--roi-config",type=Path,required=True)
    p.add_argument("--video-id",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--acquisition-interval-sec",type=float)
    args=p.parse_args(argv)
    cv2.setNumThreads(2)
    result=run_sequence(args.video,args.output,roi_config=args.roi_config,video_id=args.video_id,
                       start=args.start,count=args.count,step=args.step,
                       acquisition_interval_sec=args.acquisition_interval_sec)
    print(json.dumps(dict(status=result["status"],completed_count=result["completed_count"],
                          output=str(args.output.resolve())),ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
