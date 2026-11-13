"""Read-only regression comparison against the legacy source on real frames."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


project = Path(__file__).resolve().parents[1]
workspace = project.parent
sys.path.insert(0, str(project))
from plasmodium_v2.segmentation import segment_hsv_circle as new_segment


spec = importlib.util.spec_from_file_location(
    "legacy_plasmodium_comparison", workspace / "plasmodium_brightfield_analysis_optimized.py"
)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

params = dict(
    h_min=15, h_max=49, s_min=54, v_min=30, min_obj_size=150,
    min_hole_size=0, morph_open_size=1, morph_close_size=0,
    keep_largest=True, fill_small_body_holes=False,
    body_hole_area_ratio=0.0005, body_min_hole_size=50, body_max_hole_size=150,
    use_yellow_chroma_extension=True, yellow_chroma_min=23,
    yellow_chroma_max_value=225, yellow_chroma_min_channel=55,
    yellow_chroma_rim_exclusion_px=20, yellow_chroma_connect_radius=0,
    yellow_chroma_connect_iterations=3, use_relaxed_body_mask=False,
    relaxed_h_min=14, relaxed_h_max=55, relaxed_s_min=25, relaxed_v_min=35,
    relaxed_connect_radius=30, relaxed_min_obj_size=50, rim_exclusion_px=40,
    use_vesselness_recovery=True, vesselness_percentile=95.0,
    vesselness_min_response=0.008, vesselness_connect_radius=12,
    vesselness_min_obj_size=20, vesselness_max_added_fraction=0.02,
    vesselness_chroma_min=4, vesselness_min_channel=28, vesselness_max_value=245,
    vesselness_sigma_min=1.0, vesselness_sigma_max=4.0,
    vesselness_black_ridges=False,
)


def compare(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(set([10, total // 2, max(0, total - 11)]))
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read {index}")
        _, old_mask, old_debug = legacy.segment_hsv_circle(
            frame, dish_center=(1280, 920), dish_radius=680, **params
        )
        _, new_mask, new_debug = new_segment(
            frame, dish_center=(1280, 920), dish_radius=680, **params
        )
        intersection = int((old_mask & new_mask).sum())
        union = int((old_mask | new_mask).sum())
        iou = intersection / union if union else 1.0
        print(
            f"frame={index} iou={iou:.8f} old_area={int(old_mask.sum())} "
            f"new_area={int(new_mask.sum())} "
            f"old_vessel={old_debug['vesselness_added_area_px']} "
            f"new_vessel={new_debug['vesselness_added_area_px']}"
        )
    cap.release()


if __name__ == "__main__":
    compare(Path(r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\output2.mp4"))

