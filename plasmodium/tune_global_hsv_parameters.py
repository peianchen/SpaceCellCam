"""Tune one HSV parameter set against all completed manual masks."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from plasmodium_brightfield_analysis import segment_hsv_circle


PACKAGE_DIR = Path("results_plasmodium/manual_validation_package")
DISH_CENTER = (1280, 960)
DISH_RADIUS = 750
SEARCH_SCALE = 0.25
RANDOM_SEED = 20260623
N_RANDOM_CANDIDATES = 120
N_FULL_RES_FINALISTS = 8

BASE_PARAMS = {
    "h_min": 16,
    "h_max": 48,
    "s_min": 60,
    "v_min": 45,
    "yellow_chroma_min": 20,
    "yellow_chroma_max_value": 225,
    "yellow_chroma_min_channel": 55,
    "yellow_chroma_rim_exclusion_px": 8,
}

FIXED_PARAMS = {
    "min_obj_size": 150,
    "min_hole_size": 0,
    "morph_open_size": 1,
    "morph_close_size": 0,
    "keep_largest": True,
    "fill_small_body_holes": False,
    "use_yellow_chroma_extension": True,
    "use_relaxed_body_mask": False,
    "rim_exclusion_px": 40,
}


def _metrics(predicted, manual):
    intersection = int(np.sum(predicted & manual))
    predicted_area = int(predicted.sum())
    manual_area = int(manual.sum())
    union = predicted_area + manual_area - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "dice": (
            2.0 * intersection / (predicted_area + manual_area)
            if predicted_area + manual_area
            else 1.0
        ),
        "precision": intersection / predicted_area if predicted_area else 1.0,
        "recall": intersection / manual_area if manual_area else 1.0,
        "absolute_area_error": (
            abs(predicted_area - manual_area) / manual_area if manual_area else 0.0
        ),
    }


def _load_samples(scale):
    manifest = pd.read_csv(PACKAGE_DIR / "manual_validation_manifest.csv")
    samples = []
    for row in manifest.itertuples(index=False):
        manual_path = Path(row.expected_manual_mask)
        if not manual_path.exists():
            continue
        source = cv2.imread(str(row.source_image), cv2.IMREAD_COLOR)
        manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
        if source is None or manual is None:
            raise FileNotFoundError(f"Missing validation input for frame {row.frame}.")
        if scale != 1.0:
            width = max(1, int(round(source.shape[1] * scale)))
            height = max(1, int(round(source.shape[0] * scale)))
            source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            manual = cv2.resize(
                manual, (width, height), interpolation=cv2.INTER_NEAREST
            )
        samples.append(
            {
                "section": str(row.section),
                "frame": int(row.frame),
                "source": source,
                "manual": manual > 0,
            }
        )
    if not samples:
        raise ValueError("No completed manual masks were found.")
    return samples


def _scaled_parameters(params, scale):
    merged = {**FIXED_PARAMS, **params}
    if scale != 1.0:
        merged["min_obj_size"] = max(
            1, int(round(FIXED_PARAMS["min_obj_size"] * scale * scale))
        )
        merged["morph_open_size"] = max(
            0, int(round(FIXED_PARAMS["morph_open_size"] * scale))
        )
        merged["yellow_chroma_rim_exclusion_px"] = max(
            1, int(round(params["yellow_chroma_rim_exclusion_px"] * scale))
        )
        merged["rim_exclusion_px"] = max(
            1, int(round(FIXED_PARAMS["rim_exclusion_px"] * scale))
        )
    return merged


def _evaluate(params, samples, scale):
    rows = []
    center = (
        int(round(DISH_CENTER[0] * scale)),
        int(round(DISH_CENTER[1] * scale)),
    )
    radius = int(round(DISH_RADIUS * scale))
    call_params = _scaled_parameters(params, scale)
    for sample in samples:
        _, predicted, _ = segment_hsv_circle(
            sample["source"],
            dish_center=center,
            dish_radius=radius,
            **call_params,
        )
        rows.append(
            {
                "section": sample["section"],
                "frame": sample["frame"],
                **_metrics(predicted > 0, sample["manual"]),
            }
        )
    detail = pd.DataFrame(rows)
    section_dice = detail.groupby("section")["dice"].mean()
    mean_dice = float(detail["dice"].mean())
    min_section_dice = float(section_dice.min())
    precision = float(detail["precision"].mean())
    recall = float(detail["recall"].mean())
    area_error = float(detail["absolute_area_error"].mean())
    score = (
        0.55 * mean_dice
        + 0.30 * min_section_dice
        + 0.075 * precision
        + 0.075 * recall
        - 0.05 * area_error
    )
    summary = {
        **params,
        "score": score,
        "dice_mean": mean_dice,
        "dice_min": float(detail["dice"].min()),
        "min_section_dice": min_section_dice,
        "precision_mean": precision,
        "recall_mean": recall,
        "absolute_area_error_mean": area_error,
    }
    for section, value in section_dice.items():
        summary[f"dice_{section}"] = float(value)
    return summary, detail


def _random_candidates():
    rng = np.random.default_rng(RANDOM_SEED)
    candidates = [BASE_PARAMS.copy()]
    for _ in range(N_RANDOM_CANDIDATES):
        h_min = int(rng.integers(12, 20))
        h_max = int(rng.integers(max(45, h_min + 25), 59))
        candidates.append(
            {
                "h_min": h_min,
                "h_max": h_max,
                "s_min": int(rng.integers(25, 71)),
                "v_min": int(rng.integers(25, 61)),
                "yellow_chroma_min": int(rng.integers(10, 31)),
                "yellow_chroma_max_value": int(rng.integers(210, 246)),
                "yellow_chroma_min_channel": int(rng.integers(35, 71)),
                "yellow_chroma_rim_exclusion_px": int(rng.integers(6, 31)),
            }
        )
    return candidates


def tune_global_hsv_parameters():
    search_samples = _load_samples(SEARCH_SCALE)
    search_rows = []
    for index, candidate in enumerate(_random_candidates(), start=1):
        summary, _ = _evaluate(candidate, search_samples, SEARCH_SCALE)
        search_rows.append(summary)
        if index % 10 == 0:
            print(f"Global HSV search: {index}/{N_RANDOM_CANDIDATES + 1}")
    search = pd.DataFrame(search_rows).sort_values("score", ascending=False)
    search.to_csv(
        PACKAGE_DIR / "global_hsv_parameter_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_samples = _load_samples(1.0)
    finalist_rows = []
    finalist_details = []
    for _, candidate_row in search.head(N_FULL_RES_FINALISTS).iterrows():
        candidate = {
            key: int(candidate_row[key])
            for key in BASE_PARAMS
        }
        summary, detail = _evaluate(candidate, full_samples, 1.0)
        finalist_rows.append(summary)
        detail = detail.copy()
        for key, value in candidate.items():
            detail[key] = value
        finalist_details.append(detail)
    finalists = pd.DataFrame(finalist_rows).sort_values("score", ascending=False)
    finalists.to_csv(
        PACKAGE_DIR / "global_hsv_parameter_finalists.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best = finalists.iloc[0]
    best_params = {key: int(best[key]) for key in BASE_PARAMS}
    best_detail = next(
        detail
        for detail in finalist_details
        if all(int(detail.iloc[0][key]) == value for key, value in best_params.items())
    )
    best_detail.to_csv(
        PACKAGE_DIR / "global_hsv_best_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Best global HSV parameters:")
    for key, value in best_params.items():
        print(f"  {key}={value}")
    print(
        f"Dice mean={best['dice_mean']:.4f}, "
        f"minimum section Dice={best['min_section_dice']:.4f}, "
        f"precision={best['precision_mean']:.4f}, "
        f"recall={best['recall_mean']:.4f}"
    )
    return best_params, finalists, best_detail


if __name__ == "__main__":
    tune_global_hsv_parameters()
