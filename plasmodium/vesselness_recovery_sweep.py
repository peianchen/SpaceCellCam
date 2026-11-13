"""
Standalone vesselness recovery sweep.

Tests whether Frangi vesselness can conservatively recover thin plasmodium tubes
missed by the HSV/yellow-chroma body mask. It does not modify production code.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology

warnings.filterwarnings("ignore", category=FutureWarning)

VIDEO = Path(r"C:\Users\chen8\Pictures\研究\實驗圖\年菌\output2.mp4")
ANALYSIS_MODULE = Path("plasmodium_brightfield_analysis_optimized.py")
OUT_DIR = Path("vesselness_recovery_sweep_output")
DISH_CENTER = (1280, 920)
DISH_RADIUS = 680

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

@dataclass(frozen=True)
class Variant:
    name: str
    percentile: float
    min_response: float
    connect_radius: int
    min_obj_size: int
    max_added_fraction: float
    chroma_min: int
    min_channel: int
    black_ridges: bool = False
    sigma_min: float = 1.0
    sigma_max: float = 4.0


def variants():
    return [
        Variant("vessel_p98_r8_color", 98.0, 0.01, 8, 15, 0.020, 8, 35),
        Variant("vessel_p97_r10_color", 97.0, 0.01, 10, 15, 0.025, 8, 35),
        Variant("vessel_p96_r12_pale", 96.0, 0.008, 12, 20, 0.030, 4, 28),
        Variant("vessel_p95_r12_pale_cap", 95.0, 0.008, 12, 20, 0.020, 4, 28),
        Variant("vessel_p97_r16_pale", 97.0, 0.008, 16, 20, 0.030, 4, 28),
    ]


def load_module():
    spec = importlib.util.spec_from_file_location("plas", ANALYSIS_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_frames(start, count, stride):
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")
    frames = []
    for i in range(count):
        frame_idx = start + i * stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx}")
        frames.append((frame_idx, frame))
    cap.release()
    return frames


def circular_roi_mask(shape, center, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    cx, cy = center
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= int(radius) ** 2


def color_gate(frame_bgr, dish_mask, chroma_min, min_channel):
    b, g, r = cv2.split(frame_bgr.astype(np.int16))
    chroma = np.minimum(r - b, g - b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    max_rgb = np.maximum(np.maximum(r, g), b)
    return (chroma >= chroma_min) & (min_rgb >= min_channel) & (max_rgb <= 245) & dish_mask


def frangi_response_near_mask(module, frame_bgr, base_mask, variant):
    dish = circular_roi_mask(base_mask.shape, DISH_CENTER, DISH_RADIUS)
    near = morphology.binary_dilation(base_mask, morphology.disk(int(variant.connect_radius))) & dish
    if not np.any(near):
        return np.zeros_like(base_mask, dtype=float), near

    ys, xs = np.where(near)
    pad = max(12, int(np.ceil(variant.sigma_max * 5)))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(base_mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(base_mask.shape[1], int(xs.max()) + pad + 1)

    gray = module._as_gray_uint8(frame_bgr[y0:y1, x0:x1]).astype(np.float32) / 255.0
    near_crop = near[y0:y1, x0:x1]
    work = exposure.equalize_adapthist(gray, clip_limit=0.02)
    fill = float(np.nanmedian(work[near_crop])) if np.any(near_crop) else float(np.nanmedian(work))
    work[~near_crop] = fill
    sigmas = np.linspace(variant.sigma_min, variant.sigma_max, num=max(2, int(np.ceil(variant.sigma_max - variant.sigma_min)) + 1))
    resp_crop = filters.frangi(work, sigmas=sigmas, black_ridges=variant.black_ridges)
    resp_crop = np.nan_to_num(resp_crop, nan=0.0, posinf=0.0, neginf=0.0)
    response = np.zeros(base_mask.shape, dtype=float)
    response[y0:y1, x0:x1] = resp_crop
    return response, near


def recover(module, frame_bgr, base_mask, variant):
    response, near = frangi_response_near_mask(module, frame_bgr, base_mask, variant)
    search = near & (~base_mask)
    if not np.any(search):
        return base_mask.copy(), np.zeros_like(base_mask, dtype=bool), 0.0
    thresh = float(np.nanpercentile(response[search], variant.percentile))
    thresh = max(thresh, variant.min_response)
    gate = color_gate(frame_bgr, circular_roi_mask(base_mask.shape, DISH_CENTER, DISH_RADIUS), variant.chroma_min, variant.min_channel)
    candidate = (response >= thresh) & search & gate
    if variant.min_obj_size > 0:
        candidate = morphology.remove_small_objects(candidate.astype(bool), min_size=variant.min_obj_size)
    added_fraction = float(candidate.sum() / max(base_mask.sum(), 1))
    if added_fraction > variant.max_added_fraction:
        # Keep only strongest connected additions until the cap is respected.
        labeled, n = ndi.label(candidate)
        if n > 0:
            comp_scores = []
            for comp_id in range(1, n + 1):
                comp = labeled == comp_id
                comp_scores.append((float(np.nanmean(response[comp])), int(comp.sum()), comp_id))
            comp_scores.sort(reverse=True)
            capped = np.zeros_like(candidate, dtype=bool)
            max_add = int(round(base_mask.sum() * variant.max_added_fraction))
            total = 0
            for _, size, comp_id in comp_scores:
                if total + size > max_add and total > 0:
                    continue
                comp = labeled == comp_id
                capped |= comp
                total += size
                if total >= max_add:
                    break
            candidate = capped
            added_fraction = float(candidate.sum() / max(base_mask.sum(), 1))
    return base_mask | candidate, candidate, thresh


def overlay(frame, mask, added, label):
    out = frame.copy()
    green = np.zeros_like(out); green[..., 1] = 255
    blue = np.zeros_like(out); blue[..., 0] = 255
    mask = mask.astype(bool); added = added.astype(bool)
    out[mask] = cv2.addWeighted(out[mask], 0.55, green[mask], 0.45, 0)
    if np.any(added):
        out[added] = cv2.addWeighted(out[added], 0.35, blue[added], 0.65, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    cv2.putText(out, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    return cv2.resize(out, (512, 384), interpolation=cv2.INTER_AREA)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    module = load_module()
    phases = {"early": (0, 3, 90), "middle": (3063, 3, 90), "late": (6126, 3, 90)}
    rows = []
    sheets = []
    for phase, (start, count, stride) in phases.items():
        frames = read_frames(start, count, stride)
        phase_panels = []
        for variant in variants():
            row_panels = []
            for frame_idx, frame in frames:
                _, base, _debug = module.segment_hsv_circle(frame, dish_center=DISH_CENTER, dish_radius=DISH_RADIUS, **HSV_PARAMS)
                recovered, added, threshold = recover(module, frame, base.astype(bool), variant)
                base_skel = module.skeleton_metrics(base.astype(bool), pixel_size=1.0)
                rec_skel = module.skeleton_metrics(recovered.astype(bool), pixel_size=1.0)
                rows.append({
                    "phase": phase,
                    "frame": frame_idx,
                    "variant": variant.name,
                    "base_area_px": int(base.sum()),
                    "recovered_area_px": int(recovered.sum()),
                    "added_area_px": int(added.sum()),
                    "added_fraction": float(added.sum() / max(base.sum(), 1)),
                    "base_skeleton_px": int(base_skel["skeleton_length_px"]),
                    "recovered_skeleton_px": int(rec_skel["skeleton_length_px"]),
                    "skeleton_gain_ratio": float(rec_skel["skeleton_length_px"] / max(base_skel["skeleton_length_px"], 1)),
                    "base_endpoints": int(base_skel["n_endpoints"]),
                    "recovered_endpoints": int(rec_skel["n_endpoints"]),
                    "base_branchpoints": int(base_skel["n_branchpoints"]),
                    "recovered_branchpoints": int(rec_skel["n_branchpoints"]),
                    "threshold": threshold,
                    "border_touch": bool(module.mask_qc_metrics(recovered)["touch_roi_border"]),
                })
                if len(row_panels) < 3:
                    row_panels.append(overlay(frame, recovered, added, f"{phase} {variant.name} f{frame_idx}"))
            phase_panels.append(np.hstack(row_panels))
        sheets.append(np.vstack(phase_panels))

    csv_path = OUT_DIR / "vesselness_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # Mean summary.
    grouped = {}
    for r in rows:
        grouped.setdefault((r["phase"], r["variant"]), []).append(r)
    summary = []
    for (phase, variant), vrows in grouped.items():
        out = {"phase": phase, "variant": variant, "n_frames": len(vrows)}
        for key in ["added_fraction", "skeleton_gain_ratio", "recovered_area_px", "added_area_px", "recovered_skeleton_px"]:
            out[f"{key}_mean"] = float(np.mean([x[key] for x in vrows]))
        out["border_touch_frames"] = int(sum(bool(x["border_touch"]) for x in vrows))
        summary.append(out)
    summary_path = OUT_DIR / "vesselness_variant_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)

    sheet = np.vstack(sheets)
    sheet_path = OUT_DIR / "vesselness_overlay_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Robust ranking across phases.
    by_variant = {}
    for r in summary:
        by_variant.setdefault(r["variant"], []).append(r)
    ranked = []
    for variant, items in by_variant.items():
        mean_gain = float(np.mean([x["skeleton_gain_ratio_mean"] for x in items]))
        min_gain = float(np.min([x["skeleton_gain_ratio_mean"] for x in items]))
        max_added = float(np.max([x["added_fraction_mean"] for x in items]))
        border = int(sum(x["border_touch_frames"] for x in items))
        score = mean_gain + 0.5 * min_gain - 3.0 * max(0.0, max_added - 0.02) - 0.2 * border
        ranked.append((score, variant, mean_gain, min_gain, max_added, border))
    ranked.sort(reverse=True)
    print("Wrote", csv_path)
    print("Wrote", summary_path)
    print("Wrote", sheet_path)
    print("Ranking:")
    for score, variant, mean_gain, min_gain, max_added, border in ranked:
        print(f"{variant}: score={score:.4f}, mean_gain={mean_gain:.4f}, min_gain={min_gain:.4f}, max_added={max_added:.4f}, border={border}")

if __name__ == "__main__":
    main()

