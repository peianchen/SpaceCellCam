"""Generate review-only weak labels using the current V2 spatial mask."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np


CURRENT_PARAMS = dict(
    h_min=15, h_max=49, s_min=54, v_min=30,
    min_obj_size=150, min_hole_size=0, morph_open_size=1,
    morph_close_size=0, keep_largest=True, fill_small_body_holes=False,
    use_yellow_chroma_extension=True, yellow_chroma_min=23,
    yellow_chroma_max_value=225, yellow_chroma_min_channel=55,
    yellow_chroma_rim_exclusion_px=20, yellow_chroma_connect_radius=0,
    yellow_chroma_connect_iterations=3, body_hole_area_ratio=0.0005,
    body_min_hole_size=50, body_max_hole_size=150,
    use_relaxed_body_mask=False, relaxed_h_min=14, relaxed_h_max=55,
    relaxed_s_min=25, relaxed_v_min=35, relaxed_connect_radius=30,
    relaxed_min_obj_size=50, rim_exclusion_px=40,
    use_vesselness_recovery=True, vesselness_percentile=95.0,
    vesselness_min_response=0.008, vesselness_connect_radius=12,
    vesselness_min_obj_size=20, vesselness_max_added_fraction=0.02,
    vesselness_chroma_min=4, vesselness_min_channel=28,
    vesselness_max_value=245, vesselness_sigma_min=1.0,
    vesselness_sigma_max=4.0, vesselness_black_ridges=False,
)


def _load_v2_segmenter():
    project = Path(__file__).resolve().parents[1] / "plasmodium_analysis_v2" / "plasmodium_v2"
    package = types.ModuleType("plasmodium_v2")
    package.__path__ = [str(project)]
    sys.modules.setdefault("plasmodium_v2", package)
    return importlib.import_module("plasmodium_v2.segmentation").segment_hsv_circle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--center", nargs=2, type=int, default=(1280, 920))
    parser.add_argument("--radius", type=int, default=680)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    with (dataset / "dataset_manifest.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[:args.limit]
    mask_dir = dataset / "masks_weak_v2"
    overlay_dir = dataset / "weak_v2_overlays"
    mask_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    segment = _load_v2_segmenter()
    for number, row in enumerate(rows, 1):
        image = cv2.imread(str(dataset / row["image_path"]), cv2.IMREAD_COLOR)
        _, mask, _ = segment(image, tuple(args.center), args.radius, **CURRENT_PARAMS)
        cv2.imwrite(str(mask_dir / f"{row['sample_id']}.png"), mask.astype(np.uint8) * 255)
        overlay = image.copy()
        overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(str(overlay_dir / f"{row['sample_id']}.jpg"), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{number}/{len(rows)} {row['sample_id']}", flush=True)


if __name__ == "__main__":
    main()
