"""Measure consecutive-frame camera translation on the dish rim."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def _feature(frame: np.ndarray, scale: float, center: tuple[int, int], radius: int) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    feature = cv2.magnitude(gx, gy)
    yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
    cx, cy = center[0] * scale, center[1] * scale
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    annulus = ((rr >= radius * scale * 0.70) & (rr <= radius * scale * 1.18)).astype(np.float32)
    window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F) * annulus
    feature = (feature - np.median(feature[annulus > 0])) * annulus
    return feature.astype(np.float32), window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames-per-window", type=int, default=120)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--center", nargs=2, type=int, default=(1280, 920))
    parser.add_argument("--radius", type=int, default=680)
    args = parser.parse_args()
    rows = []
    for folder in args.inputs:
        for path in sorted(folder.glob("*.mp4")):
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            starts = np.linspace(
                0, max(0, total - args.frames_per_window - 1), args.windows, dtype=int
            )
            shifts = []
            responses = []
            for start in starts:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
                ok, previous = cap.read()
                if not ok:
                    continue
                previous_feature, window = _feature(
                    previous, args.scale, tuple(args.center), args.radius
                )
                for _ in range(args.frames_per_window):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    current_feature, _ = _feature(
                        frame, args.scale, tuple(args.center), args.radius
                    )
                    (dx, dy), response = cv2.phaseCorrelate(
                        previous_feature, current_feature, window
                    )
                    shifts.append((dx / args.scale, dy / args.scale))
                    responses.append(response)
                    previous_feature = current_feature
            cap.release()
            shift = np.asarray(shifts, dtype=float)
            magnitude = np.linalg.norm(shift, axis=1)
            row = {
                "group": folder.name, "video": path.name, "pairs": len(shift),
                "dx_median": float(np.median(shift[:, 0])),
                "dy_median": float(np.median(shift[:, 1])),
                "step_p50_px": float(np.percentile(magnitude, 50)),
                "step_p90_px": float(np.percentile(magnitude, 90)),
                "step_p99_px": float(np.percentile(magnitude, 99)),
                "step_max_px": float(np.max(magnitude)),
                "response_p10": float(np.percentile(responses, 10)),
                "response_median": float(np.median(responses)),
            }
            rows.append(row)
            print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
