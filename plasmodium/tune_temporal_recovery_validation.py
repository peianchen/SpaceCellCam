"""Rear-section temporal recovery parameter sweep for validation only."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

import plasmodium_brightfield_analysis as analysis


VIDEO_PATH = Path(r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\output2.mp4")
OUTPUT_DIR = Path("results_plasmodium") / "temporal_parameter_sweep"
OUTPUT_SCALE = 0.5
SECTION_FRAMES = 300
SKIP_EDGE_FRAMES = 3
STILLS_PER_SECTION = 6
N_JOBS = 8
CORE_BATCH_FRAMES = 48
TARGET_ADDED_FRACTION = 0.050

DISH_CENTER = (1280, 960)
DISH_RADIUS = 750

TEMPORAL_WINDOW = 5
TEMPORAL_MIN_HITS = 3
TEMPORAL_CHROMA_MIN = 9
TEMPORAL_MIN_CHANNEL = 40
TEMPORAL_MAX_VALUE = 240
TEMPORAL_RIM_EXCLUSION_PX = 30
TEMPORAL_MIN_OBJ_SIZE = 15
TEMPORAL_EXTERIOR_ONLY = False
TEMPORAL_FINAL_SMOOTH_WINDOW = 5
TEMPORAL_FINAL_SMOOTH_MIN_HITS = 3

CONFIGS = [
    {"name": "connect10_block1500", "connect_radius": 10, "block_hole_max_area": 1500},
    {"name": "connect12_block1500", "connect_radius": 12, "block_hole_max_area": 1500},
    {"name": "connect14_block1500", "connect_radius": 14, "block_hole_max_area": 1500},
    {"name": "connect12_block1000", "connect_radius": 12, "block_hole_max_area": 1000},
    {"name": "connect12_block2000", "connect_radius": 12, "block_hole_max_area": 2000},
    {"name": "connect12_block2500", "connect_radius": 12, "block_hole_max_area": 2500},
]

HSV_PARAMS = dict(
    h_min=15,
    h_max=49,
    s_min=54,
    v_min=30,
    min_obj_size=150,
    min_hole_size=0,
    morph_open_size=1,
    morph_close_size=0,
    keep_largest=True,
    fill_small_body_holes=False,
    use_yellow_chroma_extension=True,
    yellow_chroma_min=23,
    yellow_chroma_max_value=225,
    yellow_chroma_min_channel=55,
    yellow_chroma_rim_exclusion_px=20,
    yellow_chroma_connect_radius=0,
    yellow_chroma_connect_iterations=3,
    body_hole_area_ratio=0.0005,
    body_min_hole_size=50,
    body_max_hole_size=150,
    use_relaxed_body_mask=False,
    relaxed_h_min=14,
    relaxed_h_max=55,
    relaxed_s_min=25,
    relaxed_v_min=35,
    relaxed_connect_radius=30,
    relaxed_min_obj_size=50,
    rim_exclusion_px=40,
)


def pack_mask(mask):
    return np.packbits(mask.astype(np.uint8).reshape(-1))


def unpack_mask(packed, shape):
    count = int(shape[0] * shape[1])
    return np.unpackbits(packed, count=count).reshape(shape).astype(bool)


def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if total <= 0:
        raise ValueError(f"Invalid video frame count: {video_path}")
    return total, fps


def prepare_mask_cache(video_path, start_frame, end_frame):
    scaled_center = tuple(int(round(v * OUTPUT_SCALE)) for v in DISH_CENTER)
    scaled_radius = int(round(DISH_RADIUS * OUTPUT_SCALE))
    scaled_hsv = analysis._scale_temporal_hsv_params(HSV_PARAMS, OUTPUT_SCALE)
    scaled_rim = max(1, int(round(TEMPORAL_RIM_EXCLUSION_PX * OUTPUT_SCALE)))
    scaled_min_obj = max(
        1,
        int(round(TEMPORAL_MIN_OBJ_SIZE * OUTPUT_SCALE * OUTPUT_SCALE)),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    executor = ProcessPoolExecutor(max_workers=N_JOBS) if N_JOBS > 1 else None
    cache = []
    mask_shape = None
    candidate_areas = []
    recoverable_areas = []
    try:
        next_frame = start_frame
        with tqdm(total=end_frame - start_frame + 1, desc="Preparing rear strict/candidate cache") as pbar:
            while next_frame <= end_frame:
                work = []
                while len(work) < CORE_BATCH_FRAMES and next_frame <= end_frame:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Could not read frame {next_frame}")
                    frame_small = cv2.resize(
                        frame,
                        None,
                        fx=OUTPUT_SCALE,
                        fy=OUTPUT_SCALE,
                        interpolation=cv2.INTER_AREA,
                    )
                    work.append(
                        (
                            next_frame,
                            frame_small,
                            scaled_center,
                            scaled_radius,
                            scaled_hsv,
                            TEMPORAL_CHROMA_MIN,
                            TEMPORAL_MIN_CHANNEL,
                            TEMPORAL_MAX_VALUE,
                            scaled_rim,
                            scaled_min_obj,
                        )
                    )
                    next_frame += 1
                if executor is not None:
                    prepared = list(
                        executor.map(
                            analysis._prepare_temporal_validation_frame_for_parallel,
                            work,
                        )
                    )
                else:
                    prepared = [
                        analysis._prepare_temporal_validation_frame_for_parallel(item)
                        for item in work
                    ]
                prepared.sort(key=lambda item: item["frame_idx"])
                for item in prepared:
                    strict = item["mask"].astype(bool)
                    candidate = item["candidate"].astype(bool)
                    if mask_shape is None:
                        mask_shape = strict.shape
                    candidate_areas.append(int(candidate.sum()))
                    recoverable_areas.append(int((candidate & (~strict)).sum()))
                    cache.append(
                        {
                            "frame_idx": int(item["frame_idx"]),
                            "strict": pack_mask(strict),
                            "candidate": pack_mask(candidate),
                        }
                    )
                pbar.update(len(prepared))
    finally:
        cap.release()
        if executor is not None:
            executor.shutdown(wait=True)

    return {
        "records": cache,
        "shape": mask_shape,
        "candidate_area_px_mean": float(np.mean(candidate_areas)),
        "recoverable_candidate_area_px_mean": float(np.mean(recoverable_areas)),
        "scaled_center": scaled_center,
        "scaled_radius": scaled_radius,
        "scaled_rim": scaled_rim,
    }


def decode_records(cache, first_pos, last_pos):
    shape = cache["shape"]
    items = []
    for record in cache["records"][first_pos:last_pos + 1]:
        items.append(
            {
                "frame_idx": record["frame_idx"],
                "frame": None,
                "mask": unpack_mask(record["strict"], shape),
                "candidate": unpack_mask(record["candidate"], shape),
            }
        )
    return items


def iter_config_results(cache, config):
    count = len(cache["records"])
    halo = TEMPORAL_WINDOW // 2 + TEMPORAL_FINAL_SMOOTH_WINDOW // 2
    scaled_connect = max(1, int(round(config["connect_radius"] * OUTPUT_SCALE)))
    scaled_block = max(
        0,
        int(round(config["block_hole_max_area"] * OUTPUT_SCALE * OUTPUT_SCALE)),
    )
    for core_first in range(0, count, CORE_BATCH_FRAMES):
        core_last = min(count - 1, core_first + CORE_BATCH_FRAMES - 1)
        read_first = max(0, core_first - halo)
        read_last = min(count - 1, core_last + halo)
        items = decode_records(cache, read_first, read_last)
        recovered = analysis.apply_temporal_body_recovery_to_sequence(
            items,
            window=TEMPORAL_WINDOW,
            min_hits=TEMPORAL_MIN_HITS,
            connect_radius=scaled_connect,
            exterior_only=TEMPORAL_EXTERIOR_ONLY,
            block_hole_max_area=scaled_block,
            final_smooth_window=TEMPORAL_FINAL_SMOOTH_WINDOW,
            final_smooth_min_hits=TEMPORAL_FINAL_SMOOTH_MIN_HITS,
        )
        core_start_frame = cache["records"][core_first]["frame_idx"]
        core_end_frame = cache["records"][core_last]["frame_idx"]
        for item in recovered:
            frame_idx = int(item["frame_idx"])
            if core_start_frame <= frame_idx <= core_end_frame:
                yield item


def evaluate_config(cache, config, still_frames):
    shape = cache["shape"]
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    cx, cy = cache["scaled_center"]
    radius = cache["scaled_radius"]
    inner_radius = max(0, radius - cache["scaled_rim"])
    radius_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    rim_band = (radius_sq <= radius ** 2) & (radius_sq > inner_radius ** 2)

    previous = None
    ious = []
    areas = []
    added_areas = []
    added_fractions = []
    rim_added = 0
    total_added = 0
    still_results = {}
    for item in tqdm(
        iter_config_results(cache, config),
        total=len(cache["records"]),
        desc=f"Evaluating {config['name']}",
    ):
        mask = item["mask"].astype(bool)
        strict = item["strict_mask"].astype(bool)
        added = item["temporal_added_mask"].astype(bool)
        area = int(mask.sum())
        added_area = int(added.sum())
        areas.append(area)
        added_areas.append(added_area)
        added_fractions.append(float(added_area / area) if area else 0.0)
        rim_added += int((added & rim_band).sum())
        total_added += added_area
        if previous is not None:
            union = int(np.logical_or(previous, mask).sum())
            intersection = int(np.logical_and(previous, mask).sum())
            ious.append(float(intersection / union) if union else 1.0)
        previous = mask
        frame_idx = int(item["frame_idx"])
        if frame_idx in still_frames:
            still_results[frame_idx] = {
                "strict": strict.copy(),
                "added": added.copy(),
                "mask": mask.copy(),
            }

    ious = np.asarray(ious, dtype=float)
    areas = np.asarray(areas, dtype=float)
    added_areas = np.asarray(added_areas, dtype=float)
    added_fractions = np.asarray(added_fractions, dtype=float)
    row = {
        **config,
        "scaled_connect_radius": int(round(config["connect_radius"] * OUTPUT_SCALE)),
        "scaled_block_hole_max_area": int(
            round(config["block_hole_max_area"] * OUTPUT_SCALE * OUTPUT_SCALE)
        ),
        "mean_adjacent_iou": float(np.mean(ious)),
        "p05_adjacent_iou": float(np.percentile(ious, 5)),
        "min_adjacent_iou": float(np.min(ious)),
        "mask_area_cv": float(np.std(areas) / np.mean(areas)),
        "temporal_added_area_px_mean": float(np.mean(added_areas)),
        "temporal_added_fraction_mean": float(np.mean(added_fractions)),
        "temporal_added_fraction_p95": float(np.percentile(added_fractions, 95)),
        "frames_with_added": int(np.count_nonzero(added_areas > 0)),
        "rim_added_fraction": float(rim_added / total_added) if total_added else 0.0,
    }
    row["usable_numeric"] = bool(
        row["mean_adjacent_iou"] >= 0.985
        and row["p05_adjacent_iou"] >= 0.970
        and 0.04 <= row["temporal_added_fraction_mean"] <= 0.07
        and row["temporal_added_fraction_p95"] <= 0.085
        and row["rim_added_fraction"] <= 0.001
    )
    row["heuristic_score"] = float(
        8.0 * abs(row["temporal_added_fraction_mean"] - TARGET_ADDED_FRACTION)
        + 5.0 * max(0.0, 0.990 - row["mean_adjacent_iou"])
        + 5.0 * max(0.0, 0.975 - row["p05_adjacent_iou"])
        + 20.0 * row["rim_added_fraction"]
    )
    return row, still_results


def read_scaled_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not read source frame {frame_idx}")
    return cv2.resize(
        frame,
        None,
        fx=OUTPUT_SCALE,
        fy=OUTPUT_SCALE,
        interpolation=cv2.INTER_AREA,
    )


def render_overlay(frame, mask, added, title, frame_idx, total_frames):
    overlay = analysis._render_temporal_validation_overlay(
        frame,
        mask,
        added,
        title,
        frame_idx,
        total_frames,
        0.40,
    )
    cv2.putText(
        overlay,
        title,
        (30, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def save_comparison_sheets(video_path, total_frames, configs, all_stills, still_frames):
    comparison_dir = OUTPUT_DIR / "comparison_sheets"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not reopen video: {video_path}")
    try:
        for frame_idx in sorted(still_frames):
            frame = read_scaled_frame(cap, frame_idx)
            panels = []
            for config in configs:
                result = all_stills[config["name"]][frame_idx]
                panel = render_overlay(
                    frame,
                    result["mask"],
                    result["added"],
                    config["name"],
                    frame_idx,
                    total_frames,
                )
                panel = cv2.resize(panel, (640, 480), interpolation=cv2.INTER_AREA)
                panels.append(panel)
            sheet = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:6])))
            cv2.imwrite(
                str(comparison_dir / f"frame_{frame_idx:06d}_comparison.jpg"),
                sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )
    finally:
        cap.release()


def save_best_outputs(video_path, total_frames, fps, cache, config, still_frames):
    best_dir = OUTPUT_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = best_dir / "debug_masks"
    debug_dir.mkdir(parents=True, exist_ok=True)
    video_path_out = best_dir / "best_rear_overlay.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not reopen video: {video_path}")
    writer = None
    try:
        for item in tqdm(
            iter_config_results(cache, config),
            total=len(cache["records"]),
            desc=f"Saving best video {config['name']}",
        ):
            frame_idx = int(item["frame_idx"])
            frame = read_scaled_frame(cap, frame_idx)
            mask = item["mask"].astype(bool)
            strict = item["strict_mask"].astype(bool)
            added = item["temporal_added_mask"].astype(bool)
            overlay = render_overlay(
                frame,
                mask,
                added,
                config["name"],
                frame_idx,
                total_frames,
            )
            if writer is None:
                height, width = overlay.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps if fps > 0 else 30.0,
                    (width, height),
                    True,
                )
                if not writer.isOpened():
                    raise ValueError(f"Could not open writer: {video_path_out}")
            writer.write(overlay)
            if frame_idx in still_frames:
                prefix = debug_dir / f"frame_{frame_idx:06d}"
                cv2.imwrite(str(prefix) + "_original.jpg", frame)
                cv2.imwrite(str(prefix) + "_strict.png", strict.astype(np.uint8) * 255)
                cv2.imwrite(str(prefix) + "_added.png", added.astype(np.uint8) * 255)
                cv2.imwrite(str(prefix) + "_final.png", mask.astype(np.uint8) * 255)
                cv2.imwrite(str(prefix) + "_overlay.jpg", overlay)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return video_path_out


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video does not exist: {VIDEO_PATH}")
    total_frames, fps = video_info(VIDEO_PATH)
    usable_end = total_frames - SKIP_EDGE_FRAMES - 1
    start_frame = max(SKIP_EDGE_FRAMES, usable_end - SECTION_FRAMES + 1)
    end_frame = usable_end
    still_frames = set(
        np.linspace(
            start_frame,
            end_frame,
            min(STILLS_PER_SECTION, end_frame - start_frame + 1),
            dtype=int,
        ).tolist()
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Video: {VIDEO_PATH}")
    print(f"Rear frames: {start_frame}-{end_frame}")
    print(f"Configs: {len(CONFIGS)}")
    cache = prepare_mask_cache(VIDEO_PATH, start_frame, end_frame)
    print(f"candidate_area_px_mean = {cache['candidate_area_px_mean']:.2f}")
    print(
        "recoverable_candidate_area_px_mean = "
        f"{cache['recoverable_candidate_area_px_mean']:.2f}"
    )

    rows = []
    all_stills = {}
    for config in CONFIGS:
        row, still_results = evaluate_config(cache, config, still_frames)
        rows.append(row)
        all_stills[config["name"]] = still_results
        print(
            f"[{config['name']}] added={row['temporal_added_fraction_mean']:.4f}, "
            f"IoU={row['mean_adjacent_iou']:.4f}, "
            f"p05={row['p05_adjacent_iou']:.4f}, "
            f"rim={row['rim_added_fraction']:.6f}, "
            f"usable={row['usable_numeric']}"
        )

    summary = pd.DataFrame(rows).sort_values(
        ["usable_numeric", "heuristic_score"],
        ascending=[False, True],
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary["candidate_area_px_mean"] = cache["candidate_area_px_mean"]
    summary["recoverable_candidate_area_px_mean"] = cache[
        "recoverable_candidate_area_px_mean"
    ]
    summary_path = OUTPUT_DIR / "temporal_parameter_sweep_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    best_name = str(summary.iloc[0]["name"])
    best_config = next(config for config in CONFIGS if config["name"] == best_name)
    best_payload = {
        "selected_by": "numeric pre-screen; requires visual review",
        "config": best_config,
        "metrics": summary.iloc[0].to_dict(),
    }
    best_json = OUTPUT_DIR / "best_config.json"
    best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    save_comparison_sheets(
        VIDEO_PATH,
        total_frames,
        CONFIGS,
        all_stills,
        still_frames,
    )
    best_video = save_best_outputs(
        VIDEO_PATH,
        total_frames,
        fps,
        cache,
        best_config,
        still_frames,
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved numeric best config: {best_json}")
    print(f"Saved numeric best video: {best_video}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
