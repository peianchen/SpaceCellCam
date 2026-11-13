"""Sample long videos and quantify appearance variation and camera jitter.

This is a read-only diagnostic helper.  It writes contact sheets and a CSV to
the requested output directory without changing the source videos.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def _sample_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    return np.linspace(0, total - 1, min(count, total), dtype=int).tolist()


def _read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {index}")
    return frame


def _estimate_euclidean(reference: np.ndarray, frame: np.ndarray) -> tuple[float, float, float, float]:
    """Return dx, dy, rotation degrees, and ECC score at reduced resolution."""
    scale = min(1.0, 720.0 / max(reference.shape[:2]))
    size = (int(round(reference.shape[1] * scale)), int(round(reference.shape[0] * scale)))
    ref = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
    cur = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Ignore the image boundary, where codec padding and black edges dominate.
    margin = max(8, int(round(min(size) * 0.05)))
    mask = np.zeros(ref_gray.shape, np.uint8)
    mask[margin:-margin, margin:-margin] = 255
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        score, warp = cv2.findTransformECC(
            ref_gray, cur_gray, warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 3
        )
        angle = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
        return (
            float(warp[0, 2] / scale),
            float(warp[1, 2] / scale),
            angle,
            float(score),
        )
    except cv2.error:
        return np.nan, np.nan, np.nan, np.nan


def inspect_video(path: Path, sample_count: int, output_dir: Path) -> list[dict[str, object]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = _sample_indices(total, sample_count)
    reference = _read_frame(cap, indices[0])
    thumbs = []
    rows = []
    for index in indices:
        frame = _read_frame(cap, index)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dx, dy, angle, score = _estimate_euclidean(reference, frame)
        rows.append(
            {
                "group": path.parent.name,
                "video": path.name,
                "frame": index,
                "seconds": index / fps if fps > 0 else np.nan,
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total,
                "b_mean": float(frame[..., 0].mean()),
                "g_mean": float(frame[..., 1].mean()),
                "r_mean": float(frame[..., 2].mean()),
                "h_median": float(np.median(hsv[..., 0])),
                "s_median": float(np.median(hsv[..., 1])),
                "v_median": float(np.median(hsv[..., 2])),
                "dx_px": dx,
                "dy_px": dy,
                "rotation_deg": angle,
                "ecc_score": score,
            }
        )
        thumb_w = 420
        thumb = cv2.resize(frame, (thumb_w, int(round(height * thumb_w / width))))
        cv2.putText(
            thumb,
            f"{index}/{total - 1}  dx={dx:.1f} dy={dy:.1f} a={angle:.2f}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        thumbs.append(thumb)
    cap.release()
    columns = 3
    blank = np.zeros_like(thumbs[0])
    grid_rows = []
    for start in range(0, len(thumbs), columns):
        row = thumbs[start : start + columns]
        row.extend([blank] * (columns - len(row)))
        grid_rows.append(np.hstack(row))
    sheet = np.vstack(grid_rows)
    safe_name = f"{path.parent.name}_{path.stem}_samples.jpg"
    cv2.imwrite(str(output_dir / safe_name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for folder in args.inputs:
        for path in sorted(folder.glob("*.mp4")):
            print(f"Inspecting {path}", flush=True)
            all_rows.extend(inspect_video(path, args.samples, args.output))
    if all_rows:
        with (args.output / "video_variation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
