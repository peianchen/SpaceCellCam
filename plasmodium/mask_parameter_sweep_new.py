"""
Standalone mask-parameter sweep for plasmodium HSV/temporal segmentation.

This file intentionally does not modify plasmodium_brightfield_analysis_complete.py.
It imports the current analysis functions, runs candidate parameter sets on a
small frame sample, and writes review CSV/overlay images for manual selection.

Example:
python mask_parameter_sweep_new.py --start 6126 --count 12 --stride 10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_VIDEO = r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\output2.mp4"
ANALYSIS_MODULE = "plasmodium_brightfield_analysis_complete.py"

BASE_HSV_PARAMS = dict(
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
    use_yellow_chroma_extension=False,
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

BASE_TEMPORAL_PARAMS = dict(
    temporal_window=5,
    temporal_min_hits=4,
    temporal_chroma_min=10,
    temporal_min_channel=35,
    temporal_max_value=240,
    temporal_rim_exclusion_px=30,
    temporal_connect_radius=8,
    temporal_min_obj_size=20,
    temporal_exterior_only=True,
    temporal_block_hole_max_area=0,
    temporal_final_smooth_window=1,
    temporal_final_smooth_min_hits=1,
)


@dataclass(frozen=True)
class Variant:
    name: str
    hsv_updates: dict = field(default_factory=dict)
    temporal_updates: dict = field(default_factory=dict)
    note: str = ""


def default_variants() -> list[Variant]:
    return [
        Variant("baseline", note="Current conservative HSV-only temporal profile."),
        Variant(
            "lower_saturation_48",
            hsv_updates={"s_min": 48},
            note="Recover paler yellow body pixels; watch for background uptake.",
        ),
        Variant(
            "lower_saturation_42",
            hsv_updates={"s_min": 42},
            note="More aggressive pale-body recovery; higher false-positive risk.",
        ),
        Variant(
            "yellow_chroma_extension",
            hsv_updates={"use_yellow_chroma_extension": True},
            note="Adds constrained RGB/yellow chroma extension to strict HSV seed.",
        ),
        Variant(
            "yellow_chroma_plus_lower_s48",
            hsv_updates={"s_min": 48, "use_yellow_chroma_extension": True},
            note="Combined pale HSV and yellow-chroma extension.",
        ),
        Variant(
            "temporal_more_connected",
            temporal_updates={"temporal_connect_radius": 12, "temporal_min_hits": 3},
            note="Allows more temporal recovery around persistent pale branches.",
        ),
        Variant(
            "temporal_paler_candidate",
            temporal_updates={"temporal_chroma_min": 8, "temporal_min_channel": 32},
            note="Makes temporal candidate generation more sensitive to pale tubes.",
        ),
    ]


def load_analysis_module(path: Path):
    spec = importlib.util.spec_from_file_location("plas_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analysis module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_sampled_frames(video_path: Path, start: int, count: int, stride: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(count):
        frame_idx = start + i * stride
        if total > 0 and frame_idx >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        frames.append((frame_idx, frame, f"frame_{frame_idx:06d}"))
    cap.release()
    if not frames:
        raise ValueError("No frames were sampled.")
    return frames, total


def overlay_mask(frame_bgr, mask, label):
    mask = np.asarray(mask, dtype=bool)
    out = frame_bgr.copy()
    green = np.zeros_like(out)
    green[..., 1] = 255
    out[mask] = cv2.addWeighted(out[mask], 0.55, green[mask], 0.45, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    cv2.putText(out, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def run_variant(module, frames, variant: Variant, scale: float, dish_center, dish_radius):
    hsv = dict(BASE_HSV_PARAMS)
    hsv.update(variant.hsv_updates)
    temporal = dict(BASE_TEMPORAL_PARAMS)
    temporal.update(variant.temporal_updates)

    items = []
    for frame_idx, frame, name in frames:
        item = module._prepare_formal_temporal_item(
            frame_idx,
            frame,
            name,
            dish_center=dish_center,
            dish_radius=dish_radius,
            hsv_params=hsv,
            temporal_chroma_min=temporal["temporal_chroma_min"],
            temporal_min_channel=temporal["temporal_min_channel"],
            temporal_max_value=temporal["temporal_max_value"],
            temporal_rim_exclusion_px=temporal["temporal_rim_exclusion_px"],
            temporal_min_obj_size=temporal["temporal_min_obj_size"],
            frame_scale=scale,
        )
        items.append(item)

    recovered = module.apply_temporal_body_recovery_to_sequence(
        items,
        window=temporal["temporal_window"],
        min_hits=temporal["temporal_min_hits"],
        connect_radius=module._scale_radius_param_for_formal_analysis(temporal["temporal_connect_radius"], scale),
        exterior_only=temporal["temporal_exterior_only"],
        block_hole_max_area=module._scale_area_param_for_formal_analysis(temporal["temporal_block_hole_max_area"], scale),
        final_smooth_window=temporal["temporal_final_smooth_window"],
        final_smooth_min_hits=temporal["temporal_final_smooth_min_hits"],
    )
    return recovered, hsv, temporal


def summarize_item(module, variant_name, item, scale):
    factor = (1.0 / scale) ** 2
    mask = item["mask"].astype(bool)
    strict = item.get("strict_mask", mask).astype(bool)
    candidate = item["candidate"].astype(bool)
    added = item.get("temporal_added_mask", np.zeros_like(mask, dtype=bool)).astype(bool)
    skel = module.skeleton_metrics(mask, pixel_size=(1.0 / scale))
    return {
        "variant": variant_name,
        "frame": int(item["frame_idx"]),
        "scale": float(scale),
        "processed_width": int(item["frame"].shape[1]),
        "processed_height": int(item["frame"].shape[0]),
        "strict_area_orig_px": float(strict.sum() * factor),
        "candidate_area_orig_px": float(candidate.sum() * factor),
        "recoverable_candidate_orig_px": float((candidate & (~strict)).sum() * factor),
        "recovered_area_orig_px": float(mask.sum() * factor),
        "temporal_added_orig_px": float(added.sum() * factor),
        "temporal_added_fraction": float(added.sum() / max(mask.sum(), 1)),
        "skeleton_length_orig_px": float(skel["skeleton_length"]),
        "n_endpoints": int(skel["n_endpoints"]),
        "n_branchpoints": int(skel["n_branchpoints"]),
        "touches_border": bool(module.mask_qc_metrics(mask)["touch_roi_border"]),
    }


def write_contact_sheet(out_path: Path, source_frames, variant_outputs, scale: float, max_frames: int):
    panels = []
    for variant_name, items in variant_outputs.items():
        row = []
        for (frame_idx, frame_bgr, _), item in zip(source_frames[:max_frames], items[:max_frames]):
            mask = item["mask"].astype(np.uint8) * 255
            if scale != 1.0:
                mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            label = f"{variant_name} | {frame_idx}"
            overlay = overlay_mask(frame_bgr, mask.astype(bool), label)
            overlay = cv2.resize(overlay, None, fx=0.22, fy=0.22, interpolation=cv2.INTER_AREA)
            row.append(overlay)
        panels.append(np.hstack(row))
    sheet = np.vstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def mean_summary(rows):
    keys = [
        "strict_area_orig_px",
        "recovered_area_orig_px",
        "temporal_added_orig_px",
        "temporal_added_fraction",
        "skeleton_length_orig_px",
        "n_endpoints",
        "n_branchpoints",
    ]
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    summary = []
    for variant, vrows in by_variant.items():
        out = {"variant": variant, "n_frames": len(vrows)}
        for key in keys:
            vals = np.asarray([r[key] for r in vrows], dtype=float)
            out[f"{key}_mean"] = float(np.nanmean(vals))
        out["border_touch_frames"] = int(sum(bool(r["touches_border"]) for r in vrows))
        summary.append(out)
    summary.sort(key=lambda r: (r["skeleton_length_orig_px_mean"], r["recovered_area_orig_px_mean"]), reverse=True)
    return summary


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Standalone mask-parameter sweep for plasmodium videos.")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--analysis-module", default=ANALYSIS_MODULE)
    parser.add_argument("--out", default="mask_parameter_sweep_output")
    parser.add_argument("--start", type=int, default=6126)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dish-center-x", type=int, default=1280)
    parser.add_argument("--dish-center-y", type=int, default=920)
    parser.add_argument("--dish-radius", type=int, default=680)
    parser.add_argument("--contact-frames", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    video_path = Path(args.video)
    module = load_analysis_module(Path(args.analysis_module))
    frames, total_frames = read_sampled_frames(video_path, args.start, args.count, args.stride)
    variants = default_variants()

    all_rows = []
    variant_outputs = {}
    metadata = {
        "video": str(video_path),
        "total_frames": total_frames,
        "sampled_frames": [idx for idx, _, _ in frames],
        "scale": args.scale,
        "dish_center": [args.dish_center_x, args.dish_center_y],
        "dish_radius": args.dish_radius,
        "note": "Higher skeleton/branch counts can indicate better tube retention or more false positives; review overlays before choosing.",
    }

    for variant in variants:
        items, hsv, temporal = run_variant(
            module,
            frames,
            variant,
            scale=args.scale,
            dish_center=(args.dish_center_x, args.dish_center_y),
            dish_radius=args.dish_radius,
        )
        variant_outputs[variant.name] = items
        metadata.setdefault("variants", {})[variant.name] = {
            "note": variant.note,
            "hsv_updates": variant.hsv_updates,
            "temporal_updates": variant.temporal_updates,
            "resolved_hsv_params": hsv,
            "resolved_temporal_params": temporal,
        }
        for item in items:
            all_rows.append(summarize_item(module, variant.name, item, args.scale))

    frame_csv = out_dir / "frame_metrics.csv"
    summary_csv = out_dir / "variant_summary.csv"
    metadata_json = out_dir / "sweep_metadata.json"
    sheet_path = out_dir / "overlay_contact_sheet.jpg"

    write_csv(frame_csv, all_rows)
    summary = mean_summary(all_rows)
    write_csv(summary_csv, summary)
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_contact_sheet(sheet_path, frames, variant_outputs, args.scale, max(1, args.contact_frames))

    print(f"Sampled {len(frames)} frames from {video_path}")
    print(f"Wrote {frame_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {metadata_json}")
    print(f"Wrote {sheet_path}")
    print("Top variants by skeleton_length_orig_px_mean:")
    for row in summary[:5]:
        print(
            f"  {row['variant']}: skeleton={row['skeleton_length_orig_px_mean']:.1f}, "
            f"area={row['recovered_area_orig_px_mean']:.1f}, "
            f"added_fraction={row['temporal_added_fraction_mean']:.4f}, "
            f"border_touch={row['border_touch_frames']}/{row['n_frames']}"
        )


if __name__ == "__main__":
    main()
