"""Compare temporal recovery with and without dish-rim motion compensation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from plasmodium_v2.segmentation import segment_hsv_circle
from plasmodium_v2.temporal import (
    apply_temporal_body_recovery_to_sequence,
    temporal_pale_branch_candidate,
)
from tools.cross_video_mask_diagnostics import CURRENT_PARAMS


def _overlay(frame, strict, recovered, title):
    image = frame.copy()
    addition = recovered & ~strict
    image[strict] = (0.45 * image[strict] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    image[addition] = (0.25 * image[addition] + 0.75 * np.array([255, 255, 0])).astype(np.uint8)
    cv2.putText(image, title, (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center", nargs=2, type=int, default=(1280, 920))
    parser.add_argument("--radius", type=int, default=680)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = np.linspace(total * 0.2, total * 0.8, 3, dtype=int)
    rows = []
    sheets = []
    for target in targets:
        items = []
        for frame_index in range(target - 2, target + 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index}")
            _, strict, _ = segment_hsv_circle(
                frame, tuple(args.center), args.radius, **CURRENT_PARAMS
            )
            candidate = temporal_pale_branch_candidate(
                frame, tuple(args.center), args.radius,
                chroma_min=10, min_channel=35, max_value=240,
                rim_exclusion_px=30, min_obj_size=20,
            )
            items.append({
                "frame_idx": frame_index, "frame": frame, "mask": strict,
                "candidate": candidate, "dish_center": tuple(args.center),
                "dish_radius": args.radius,
            })
        plain = apply_temporal_body_recovery_to_sequence(
            items, window=5, min_hits=4, connect_radius=8,
            motion_compensation=False,
        )[2]
        compensated_sequence = apply_temporal_body_recovery_to_sequence(
            items, window=5, min_hits=4, connect_radius=8,
            motion_compensation=True,
        )
        compensated = compensated_sequence[2]
        corrections = [
            float(np.hypot(item["motion_compensation_dx_px"], item["motion_compensation_dy_px"]))
            for item in compensated_sequence if item["motion_compensation_applied"]
        ]
        strict = items[2]["mask"]
        plain_added = int((plain["mask"] & ~strict).sum())
        comp_added = int((compensated["mask"] & ~strict).sum())
        rows.append({
            "frame": int(target), "strict_area": int(strict.sum()),
            "uncompensated_added": plain_added,
            "compensated_added": comp_added,
            "center_dx_px": compensated["motion_compensation_dx_px"],
            "center_dy_px": compensated["motion_compensation_dy_px"],
            "center_response": compensated["motion_compensation_response"],
            "neighbor_max_correction_px": max(corrections, default=0.0),
        })
        left = _overlay(items[2]["frame"], strict, plain["mask"], f"No compensation | added {plain_added}")
        right = _overlay(items[2]["frame"], strict, compensated["mask"], f"Motion compensated | added {comp_added}")
        pair = np.hstack([left, right])
        pair = cv2.resize(pair, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
        sheets.append(pair)
    cap.release()
    cv2.imwrite(str(args.output / "comparison.jpg"), np.vstack(sheets), [cv2.IMWRITE_JPEG_QUALITY, 94])
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
