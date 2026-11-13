"""Render the current production mask on representative frames from many videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from plasmodium_v2.segmentation import segment_hsv_circle


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--center", nargs=2, type=int, default=(1280, 920))
    parser.add_argument("--radius", type=int, default=680)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for group_index, folder in enumerate(args.inputs):
        for video_index, path in enumerate(sorted(folder.glob("*.mp4"))):
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total - 1, args.samples, dtype=int)
            thumbs = []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    continue
                _, mask, debug = segment_hsv_circle(
                    frame, tuple(args.center), args.radius, **CURRENT_PARAMS
                )
                overlay = frame.copy()
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * 0.45
                    + np.array([0, 255, 0], np.float32) * 0.55
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 4)
                label = (
                    f"f={index}/{total - 1} area={int(mask.sum())} "
                    f"seed={debug['seed_area_px']} yellow={debug['yellow_chroma_added_area_px']} "
                    f"vessel={debug['vesselness_added_area_px']}"
                )
                thumb_w = 520
                thumb = cv2.resize(
                    overlay, (thumb_w, int(round(frame.shape[0] * thumb_w / frame.shape[1])))
                )
                cv2.putText(
                    thumb, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                thumbs.append(thumb)
            cap.release()
            if not thumbs:
                continue
            columns = 3
            blank = np.zeros_like(thumbs[0])
            rows = []
            for start in range(0, len(thumbs), columns):
                row = thumbs[start : start + columns]
                row.extend([blank] * (columns - len(row)))
                rows.append(np.hstack(row))
            cv2.imwrite(
                str(args.output / f"group{group_index}_video{video_index}_{path.stem}_mask.jpg"),
                np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            print(f"Rendered {folder.name}/{path.name}", flush=True)


if __name__ == "__main__":
    main()
