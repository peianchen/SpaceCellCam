"""Test low-cost temporal recovery of pale plasmodium branches."""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology

from plasmodium_brightfield_analysis import segment_hsv_circle


VIDEO_PATH = Path(r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\output2.mp4")
OUTPUT_DIR = Path("results_plasmodium/temporal_recovery_test_hits4")

OUTPUT_SCALE = 0.5
SECTION_FRAMES = 300
SKIP_EDGE_FRAMES = 3

TEMPORAL_WINDOW = 5
TEMPORAL_MIN_HITS = 4
TEMPORAL_CHROMA_MIN = 10
TEMPORAL_MIN_CHANNEL = 35
TEMPORAL_MAX_VALUE = 240
TEMPORAL_RIM_EXCLUSION_PX = 30
TEMPORAL_CONNECT_RADIUS = 8
TEMPORAL_MIN_OBJECT_SIZE = 20

DISH_CENTER = (1280, 960)
DISH_RADIUS = 750

HSV_PARAMS = {
    "h_min": 15,
    "h_max": 49,
    "s_min": 54,
    "v_min": 30,
    "min_obj_size": 150,
    "min_hole_size": 0,
    "morph_open_size": 1,
    "morph_close_size": 0,
    "keep_largest": True,
    "fill_small_body_holes": False,
    "use_yellow_chroma_extension": True,
    "yellow_chroma_min": 23,
    "yellow_chroma_max_value": 225,
    "yellow_chroma_min_channel": 55,
    "yellow_chroma_rim_exclusion_px": 20,
    "yellow_chroma_connect_radius": 0,
    "yellow_chroma_connect_iterations": 3,
    "use_relaxed_body_mask": False,
    "rim_exclusion_px": 40,
}


def _scaled_hsv_params(scale):
    params = dict(HSV_PARAMS)
    params["min_obj_size"] = max(
        1, int(round(params["min_obj_size"] * scale * scale))
    )
    params["morph_open_size"] = max(
        0, int(round(params["morph_open_size"] * scale))
    )
    params["yellow_chroma_rim_exclusion_px"] = max(
        1, int(round(params["yellow_chroma_rim_exclusion_px"] * scale))
    )
    params["rim_exclusion_px"] = max(
        1, int(round(params["rim_exclusion_px"] * scale))
    )
    return params


def _pale_candidate(frame, center, radius, scale):
    blue, green, red = cv2.split(frame.astype(np.int16))
    yellow_chroma = np.minimum(red - blue, green - blue)
    max_channel = np.maximum(np.maximum(red, green), blue)
    height, width = frame.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    inner_radius = max(
        1,
        int(round(radius - TEMPORAL_RIM_EXCLUSION_PX * scale)),
    )
    inner_dish = (
        (xx - int(center[0])) ** 2 + (yy - int(center[1])) ** 2
        <= inner_radius**2
    )
    candidate = (
        (yellow_chroma >= TEMPORAL_CHROMA_MIN)
        & (red >= TEMPORAL_MIN_CHANNEL)
        & (green >= TEMPORAL_MIN_CHANNEL)
        & (max_channel <= TEMPORAL_MAX_VALUE)
        & inner_dish
    )
    min_size = max(
        1, int(round(TEMPORAL_MIN_OBJECT_SIZE * scale * scale))
    )
    return morphology.remove_small_objects(candidate, min_size=min_size)


def _components_touching(candidate, reference):
    labels = measure.label(candidate.astype(bool), connectivity=2)
    selected = np.unique(labels[reference.astype(bool) & (labels > 0)])
    if selected.size == 0:
        return np.zeros_like(candidate, dtype=bool)
    return np.isin(labels, selected)


def _render_panel(frame, mask, addition, label):
    overlay = frame.copy().astype(np.float32)
    mask = mask.astype(bool)
    overlay[mask] = 0.58 * overlay[mask] + 0.42 * np.array(
        [0, 255, 0], dtype=np.float32
    )
    if addition is not None:
        addition = addition.astype(bool)
        overlay[addition] = (
            0.25 * overlay[addition]
            + 0.75 * np.array([255, 255, 0], dtype=np.float32)
        )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.putText(
        overlay,
        label,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def run_temporal_recovery_test():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    still_dir = OUTPUT_DIR / "stills"
    still_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    start_frame = max(SKIP_EDGE_FRAMES, total_frames - SECTION_FRAMES)
    end_frame = total_frames - SKIP_EDGE_FRAMES - 1
    half_window = TEMPORAL_WINDOW // 2
    output_start = start_frame + half_window
    output_end = end_frame - half_window
    still_frames = set(
        np.linspace(output_start, output_end, 8, dtype=int).tolist()
    )

    scale = OUTPUT_SCALE
    center = (
        int(round(DISH_CENTER[0] * scale)),
        int(round(DISH_CENTER[1] * scale)),
    )
    radius = int(round(DISH_RADIUS * scale))
    hsv_params = _scaled_hsv_params(scale)
    connect_radius = max(
        1, int(round(TEMPORAL_CONNECT_RADIUS * scale))
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    buffer = deque(maxlen=TEMPORAL_WINDOW)
    writer = None
    rows = []
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            _, strict_mask, _ = segment_hsv_circle(
                frame,
                dish_center=center,
                dish_radius=radius,
                **hsv_params,
            )
            candidate = _pale_candidate(frame, center, radius, scale)
            buffer.append(
                {
                    "frame": frame_idx,
                    "image": frame,
                    "strict": strict_mask.astype(bool),
                    "candidate": candidate.astype(bool),
                }
            )
            if len(buffer) < TEMPORAL_WINDOW:
                continue

            center_item = buffer[half_window]
            hits = np.sum(
                np.stack([item["candidate"] for item in buffer], axis=0),
                axis=0,
            )
            consensus = hits >= TEMPORAL_MIN_HITS
            near_strict = morphology.binary_dilation(
                center_item["strict"],
                morphology.disk(connect_radius),
            )
            trusted = _components_touching(consensus, near_strict)
            addition = trusted & (~center_item["strict"])
            final_mask = center_item["strict"] | addition

            original_panel = center_item["image"].copy()
            cv2.putText(
                original_panel,
                f"original | frame {center_item['frame']}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            strict_panel = _render_panel(
                center_item["image"],
                center_item["strict"],
                None,
                "current HSV",
            )
            temporal_panel = _render_panel(
                center_item["image"],
                final_mask,
                addition,
                "temporal | cyan=added",
            )
            comparison = np.hstack(
                [original_panel, strict_panel, temporal_panel]
            )
            if writer is None:
                height, width = comparison.shape[:2]
                writer = cv2.VideoWriter(
                    str(OUTPUT_DIR / "rear_temporal_comparison.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps if source_fps > 0 else 30.0,
                    (width, height),
                    True,
                )
                if not writer.isOpened():
                    raise ValueError("Could not create temporal comparison video.")
            writer.write(comparison)

            if center_item["frame"] in still_frames:
                cv2.imwrite(
                    str(
                        still_dir
                        / f"frame_{center_item['frame']:06d}_comparison.jpg"
                    ),
                    comparison,
                    [cv2.IMWRITE_JPEG_QUALITY, 94],
                )
            strict_area = int(center_item["strict"].sum())
            addition_area = int(addition.sum())
            rows.append(
                {
                    "frame": center_item["frame"],
                    "strict_area_px": strict_area,
                    "temporal_added_px": addition_area,
                    "temporal_added_fraction": (
                        addition_area / strict_area if strict_area else np.nan
                    ),
                    "final_area_px": int(final_mask.sum()),
                }
            )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "rear_temporal_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config = pd.DataFrame(
        [
            {
                "total_frames": total_frames,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "output_start": output_start,
                "output_end": output_end,
                "skip_edge_frames": SKIP_EDGE_FRAMES,
                "temporal_window": TEMPORAL_WINDOW,
                "temporal_min_hits": TEMPORAL_MIN_HITS,
                "temporal_chroma_min": TEMPORAL_CHROMA_MIN,
                "temporal_rim_exclusion_px": TEMPORAL_RIM_EXCLUSION_PX,
                "temporal_connect_radius": TEMPORAL_CONNECT_RADIUS,
            }
        ]
    )
    config.to_csv(
        OUTPUT_DIR / "temporal_test_config.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved temporal comparison: {OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    run_temporal_recovery_test()

