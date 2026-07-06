"""Evaluate frozen masks against blinded, adjudicated human annotations."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import VALID_ROLES, load_json, package_root, read_binary_mask, sha256_file, write_json
from validation_metrics import evaluate_mask_pair, evaluate_temporal_pair, finite_values


METHODS = ("strict", "temporal")
SUMMARY_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "absolute_area_error",
    "boundary_f1",
    "hd95_px",
    "cldice",
    "skeleton_length_absolute_error",
    "rim_fp_fraction_of_prediction",
    "internal_hole_fp_fraction_of_prediction",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
        fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def required_mask_paths(root: Path, rows: list[dict[str, str]], annotators: list[str]):
    required: list[tuple[str, Path]] = []
    for row in rows:
        bid = row["blind_id"]
        for annotator in annotators:
            required.append((f"{annotator}:{bid}", root / "annotators" / annotator / "masks" / f"{bid}.png"))
        required.append((f"adjudicated:{bid}", root / "adjudicated_masks" / f"{bid}.png"))
        for method in METHODS:
            required.append((f"prediction-{method}:{bid}", root / "predictions" / method / f"{bid}.png"))
    return required


def verify_inputs(root: Path, rows: list[dict[str, str]], config: dict) -> None:
    missing = [label for label, path in required_mask_paths(
        root, rows, list(config["annotation"]["annotators"])
    ) if not path.exists()]
    repeat_manifest_path = root / "private_repeat_manifest.csv"
    if repeat_manifest_path.exists():
        repeat_rows = read_csv(repeat_manifest_path)
        for row in repeat_rows:
            rid = row["repeat_blind_id"]
            for annotator in config["annotation"].get("repeat_annotators", []):
                path = root / "annotators" / f"{annotator}_repeat" / "masks" / f"{rid}.png"
                if not path.exists():
                    missing.append(f"{annotator}_repeat:{rid}")
    if missing:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"Validation inputs are incomplete:\n  {preview}{suffix}")


def identity(row: dict[str, str]) -> dict[str, object]:
    return {
        "blind_id": row["blind_id"],
        "video_id": row["video_id"],
        "biological_replicate": row["biological_replicate"],
        "condition": row["condition"],
        "phase": row["phase"],
        "window_id": row["window_id"],
        "window_position": int(row["window_position"]),
        "source_frame_idx": int(row["source_frame_idx"]),
    }


def evaluate_frame_masks(root: Path, rows: list[dict[str, str]], config: dict):
    settings = config["evaluation"]
    output: list[dict[str, object]] = []
    inter_rater: list[dict[str, object]] = []
    annotators = list(config["annotation"]["annotators"])
    if len(annotators) < 2:
        raise ValueError("At least two annotators are required")
    for index, row in enumerate(rows, start=1):
        bid = row["blind_id"]
        shape = (int(row["image_height"]), int(row["image_width"]))
        reference = read_binary_mask(root / "adjudicated_masks" / f"{bid}.png", shape)
        center = (int(row["dish_center_x"]), int(row["dish_center_y"]))
        radius = int(row["dish_radius"])
        pixel_size = float(row["pixel_size"])
        base = identity(row)
        for method in METHODS:
            prediction = read_binary_mask(root / "predictions" / method / f"{bid}.png", shape)
            output.append(
                {
                    **base,
                    "method": method,
                    **evaluate_mask_pair(
                        prediction,
                        reference,
                        pixel_size,
                        center,
                        radius,
                        int(settings["boundary_tolerance_px"]),
                        int(settings["rim_width_px"]),
                    ),
                }
            )
        first = read_binary_mask(
            root / "annotators" / annotators[0] / "masks" / f"{bid}.png", shape
        )
        second = read_binary_mask(
            root / "annotators" / annotators[1] / "masks" / f"{bid}.png", shape
        )
        inter_rater.append(
            {
                **base,
                "comparison": f"{annotators[0]}_vs_{annotators[1]}",
                **evaluate_mask_pair(
                    first,
                    second,
                    pixel_size,
                    center,
                    radius,
                    int(settings["boundary_tolerance_px"]),
                    int(settings["rim_width_px"]),
                ),
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"Evaluated frame masks: {index}/{len(rows)}")
    return output, inter_rater


def evaluate_intra_rater(root: Path, rows: list[dict[str, str]], config: dict):
    repeat_path = root / "private_repeat_manifest.csv"
    if not repeat_path.exists():
        return []
    repeats = read_csv(repeat_path)
    by_blind = {row["blind_id"]: row for row in rows}
    settings = config["evaluation"]
    output: list[dict[str, object]] = []
    for repeat in repeats:
        primary_id = repeat["primary_blind_id"]
        repeat_id = repeat["repeat_blind_id"]
        row = by_blind[primary_id]
        shape = (int(row["image_height"]), int(row["image_width"]))
        center = (int(row["dish_center_x"]), int(row["dish_center_y"]))
        radius = int(row["dish_radius"])
        for annotator in config["annotation"].get("repeat_annotators", []):
            primary = read_binary_mask(
                root / "annotators" / annotator / "masks" / f"{primary_id}.png",
                shape,
            )
            repeated = read_binary_mask(
                root / "annotators" / f"{annotator}_repeat" / "masks" / f"{repeat_id}.png",
                shape,
            )
            output.append(
                {
                    **identity(row),
                    "repeat_blind_id": repeat_id,
                    "annotator": annotator,
                    **evaluate_mask_pair(
                        repeated,
                        primary,
                        float(row["pixel_size"]),
                        center,
                        radius,
                        int(settings["boundary_tolerance_px"]),
                        int(settings["rim_width_px"]),
                    ),
                }
            )
    return output


def evaluate_temporal(root: Path, rows: list[dict[str, str]]):
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["video_id"], row["window_id"])].append(row)
    output: list[dict[str, object]] = []
    for (_, window_id), group in sorted(grouped.items()):
        group.sort(key=lambda value: int(value["window_position"]))
        for previous_row, current_row in zip(group[:-1], group[1:]):
            previous_id = previous_row["blind_id"]
            current_id = current_row["blind_id"]
            shape = (int(current_row["image_height"]), int(current_row["image_width"]))
            ref_previous = read_binary_mask(root / "adjudicated_masks" / f"{previous_id}.png", shape)
            ref_current = read_binary_mask(root / "adjudicated_masks" / f"{current_id}.png", shape)
            dt = float(current_row["frame_interval_s"])
            for method in METHODS:
                pred_previous = read_binary_mask(root / "predictions" / method / f"{previous_id}.png", shape)
                pred_current = read_binary_mask(root / "predictions" / method / f"{current_id}.png", shape)
                output.append(
                    {
                        **identity(current_row),
                        "method": method,
                        "previous_blind_id": previous_id,
                        "current_blind_id": current_id,
                        "frame_interval_s": dt,
                        **evaluate_temporal_pair(
                            pred_previous,
                            pred_current,
                            ref_previous,
                            ref_current,
                            dt,
                        ),
                    }
                )
    return output


def cluster_bootstrap(
    rows: list[dict[str, object]],
    metric: str,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    by_cluster: dict[str, np.ndarray] = {}
    cluster_rows: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            cluster_rows[str(row["biological_replicate"])].append(value)
    for key, values in cluster_rows.items():
        by_cluster[key] = np.asarray(values, dtype=float)
    clusters = sorted(by_cluster)
    if not clusters:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.concatenate([by_cluster[str(cluster)] for cluster in sampled])
        estimates[iteration] = float(np.mean(values))
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def summarize_methods(frame_metrics: list[dict[str, object]], config: dict):
    settings = config["evaluation"]
    iterations = int(settings["cluster_bootstrap_iterations"])
    confidence = float(settings["bootstrap_confidence_level"])
    base_seed = int(config["random_seed"])
    summary: list[dict[str, object]] = []
    phase_summary: list[dict[str, object]] = []
    video_summary: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        subset = [row for row in frame_metrics if row["method"] == method]
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            values = finite_values([row[metric] for row in subset])
            low, high = cluster_bootstrap(
                subset,
                metric,
                iterations,
                confidence,
                base_seed + method_index * 1000 + metric_index,
            )
            summary.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_frames": len(values),
                    "n_biological_replicates": len(
                        {str(row["biological_replicate"]) for row in subset}
                    ),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "p05": float(np.percentile(values, 5)) if len(values) else np.nan,
                    "p95": float(np.percentile(values, 95)) if len(values) else np.nan,
                    "cluster_bootstrap_ci_low": low,
                    "cluster_bootstrap_ci_high": high,
                }
            )
            for phase in sorted({str(row["phase"]) for row in subset}):
                phase_values = finite_values(
                    [row[metric] for row in subset if row["phase"] == phase]
                )
                phase_summary.append(
                    {
                        "method": method,
                        "phase": phase,
                        "metric": metric,
                        "n_frames": len(phase_values),
                        "mean": float(np.mean(phase_values)) if len(phase_values) else np.nan,
                        "median": float(np.median(phase_values)) if len(phase_values) else np.nan,
                        "p05": float(np.percentile(phase_values, 5)) if len(phase_values) else np.nan,
                        "p95": float(np.percentile(phase_values, 95)) if len(phase_values) else np.nan,
                    }
                )
        for replicate in sorted({str(row["biological_replicate"]) for row in subset}):
            replicate_rows = [row for row in subset if row["biological_replicate"] == replicate]
            for metric in SUMMARY_METRICS:
                values = finite_values([row[metric] for row in replicate_rows])
                video_summary.append(
                    {
                        "method": method,
                        "biological_replicate": replicate,
                        "video_id": replicate_rows[0]["video_id"],
                        "metric": metric,
                        "n_frames": len(values),
                        "mean": float(np.mean(values)) if len(values) else np.nan,
                        "median": float(np.median(values)) if len(values) else np.nan,
                    }
                )
    return summary, phase_summary, video_summary


def paired_comparison(frame_metrics: list[dict[str, object]], config: dict):
    by_key: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in frame_metrics:
        by_key[str(row["blind_id"])][str(row["method"])] = row
    rows: list[dict[str, object]] = []
    for metric in SUMMARY_METRICS:
        deltas = []
        paired_rows = []
        for pair in by_key.values():
            if set(pair) != set(METHODS):
                continue
            delta = float(pair["temporal"][metric]) - float(pair["strict"][metric])
            if math.isfinite(delta):
                deltas.append(delta)
                paired_rows.append(
                    {
                        "biological_replicate": pair["strict"]["biological_replicate"],
                        "delta": delta,
                    }
                )
        low, high = cluster_bootstrap(
            paired_rows,
            "delta",
            int(config["evaluation"]["cluster_bootstrap_iterations"]),
            float(config["evaluation"]["bootstrap_confidence_level"]),
            int(config["random_seed"]) + 5000 + len(rows),
        )
        rows.append(
            {
                "metric": metric,
                "delta_definition": "temporal_minus_strict",
                "n_pairs": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else np.nan,
                "median_delta": float(np.median(deltas)) if deltas else np.nan,
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
            }
        )
    return rows


def mean_metric(rows: list[dict[str, object]], metric: str) -> float:
    values = finite_values([row[metric] for row in rows])
    return float(np.mean(values)) if len(values) else float("nan")


def acceptance_results(
    summary: list[dict[str, object]],
    frame_metrics: list[dict[str, object]],
    inter_rater: list[dict[str, object]],
    config: dict,
):
    thresholds = config["acceptance"]
    lookup = {(row["method"], row["metric"]): row for row in summary}
    human_cldice = mean_metric(inter_rater, "cldice")
    results: list[dict[str, object]] = []
    for method in METHODS:
        subset = [row for row in frame_metrics if row["method"] == method]
        checks = [
            (
                "dice_mean_ci_low",
                float(lookup[(method, "dice")]["cluster_bootstrap_ci_low"]),
                ">=",
                float(thresholds["dice_mean_ci_low_min"]),
            ),
            (
                "precision_mean_ci_low",
                float(lookup[(method, "precision")]["cluster_bootstrap_ci_low"]),
                ">=",
                float(thresholds["precision_mean_ci_low_min"]),
            ),
            (
                "recall_mean_ci_low",
                float(lookup[(method, "recall")]["cluster_bootstrap_ci_low"]),
                ">=",
                float(thresholds["recall_mean_ci_low_min"]),
            ),
            (
                "absolute_area_error_median",
                float(lookup[(method, "absolute_area_error")]["median"]),
                "<=",
                float(thresholds["absolute_area_error_median_max"]),
            ),
            (
                "absolute_area_error_p95",
                float(lookup[(method, "absolute_area_error")]["p95"]),
                "<=",
                float(thresholds["absolute_area_error_p95_max"]),
            ),
            (
                "rim_fp_fraction_mean",
                mean_metric(subset, "rim_fp_fraction_of_prediction"),
                "<=",
                float(thresholds["rim_fp_fraction_mean_max"]),
            ),
            (
                "cldice_vs_human_floor",
                mean_metric(subset, "cldice"),
                ">=",
                human_cldice - float(thresholds["cldice_margin_below_interrater_max"]),
            ),
        ]
        for check, value, operator, threshold in checks:
            passed = (
                math.isfinite(value)
                and math.isfinite(threshold)
                and (value >= threshold if operator == ">=" else value <= threshold)
            )
            results.append(
                {
                    "method": method,
                    "check": check,
                    "value": value,
                    "operator": operator,
                    "threshold": threshold,
                    "pass": bool(passed),
                }
            )
    replicate_count = len(
        {str(row["biological_replicate"]) for row in frame_metrics}
    )
    minimum_replicates = int(
        thresholds["minimum_test_biological_replicates"]
    )
    results.insert(
        0,
        {
            "method": "study_design",
            "check": "minimum_test_biological_replicates",
            "value": replicate_count,
            "operator": ">=",
            "threshold": minimum_replicates,
            "pass": replicate_count >= minimum_replicates,
        },
    )
    return results


def bland_altman_plots(frame_metrics: list[dict[str, object]], output_dir: Path) -> None:
    colors = {"front": "#0072B2", "middle": "#E69F00", "rear": "#009E73"}
    for method in METHODS:
        subset = [row for row in frame_metrics if row["method"] == method]
        reference = np.asarray([row["reference_area"] for row in subset], dtype=float)
        prediction = np.asarray([row["prediction_area"] for row in subset], dtype=float)
        average = (reference + prediction) / 2.0
        difference = prediction - reference
        mean_difference = float(np.mean(difference))
        sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for phase in sorted(set(str(row["phase"]) for row in subset)):
            selector = np.asarray([row["phase"] == phase for row in subset])
            ax.scatter(
                average[selector],
                difference[selector],
                s=24,
                alpha=0.75,
                label=phase,
                color=colors.get(phase),
            )
        ax.axhline(mean_difference, color="black", linewidth=1.5, label="mean bias")
        ax.axhline(mean_difference + 1.96 * sd, color="red", linestyle="--", linewidth=1)
        ax.axhline(mean_difference - 1.96 * sd, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Bland-Altman area agreement: {method}")
        ax.set_xlabel("Mean manual and predicted area")
        ax.set_ylabel("Predicted - manual area")
        ax.legend(frameon=False)
        fig.savefig(output_dir / f"bland_altman_area_{method}.png", dpi=180)
        plt.close(fig)


def failure_gallery(
    root: Path,
    frame_metrics: list[dict[str, object]],
    count: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        worst = sorted(
            [row for row in frame_metrics if row["method"] == method],
            key=lambda row: float(row["dice"]),
        )[:count]
        for rank, row in enumerate(worst, start=1):
            bid = str(row["blind_id"])
            image = cv2.imread(str(root / "annotation_blinded" / "images" / f"{bid}.png"))
            if image is None:
                raise ValueError(f"Could not read blinded image: {bid}")
            shape = image.shape[:2]
            prediction = read_binary_mask(root / "predictions" / method / f"{bid}.png", shape)
            reference = read_binary_mask(root / "adjudicated_masks" / f"{bid}.png", shape)
            tp = prediction & reference
            fp = prediction & (~reference)
            fn = (~prediction) & reference
            color = np.zeros_like(image)
            color[tp] = (0, 180, 0)
            color[fp] = (0, 0, 255)
            color[fn] = (255, 255, 0)
            overlay = image.copy()
            occupied = tp | fp | fn
            overlay[occupied] = cv2.addWeighted(
                image[occupied], 0.45, color[occupied], 0.55, 0
            )
            cv2.putText(
                overlay,
                f"{method} rank {rank} | Dice={float(row['dice']):.3f} | green=TP red=FP cyan=FN",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(output_dir / f"{method}_{rank:02d}_{bid}.jpg"), overlay)


def write_report(
    output_dir: Path,
    config: dict,
    role: str,
    prediction_metadata: dict,
    rows: list[dict[str, str]],
    acceptance: list[dict[str, object]],
    inter_rater: list[dict[str, object]],
    intra_rater: list[dict[str, object]],
) -> None:
    all_pass = {
        method: all(bool(row["pass"]) for row in acceptance if row["method"] == method)
        for method in METHODS
    }
    publication_eligible = bool(prediction_metadata.get("publication_eligible")) and role == "test"
    primary_method = str(config["evaluation"]["primary_method"])
    lines = [
        "# Plasmodium Segmentation Validation Report",
        "",
        f"- Dataset role: `{role}`",
        f"- Independent biological replicates: `{len(set(row['biological_replicate'] for row in rows))}`",
        f"- Annotated frames: `{len(rows)}`",
        f"- Frozen prediction run eligible for publication claim: `{publication_eligible}`",
        f"- Pre-specified primary method: `{primary_method}`",
        "",
        "## Pre-specified acceptance",
        "",
    ]
    design = next(row for row in acceptance if row["method"] == "study_design")
    lines.append(
        f"- `study_design`: `{'PASS' if design['pass'] else 'FAIL'}`"
    )
    lines.append(
        f"  - {design['check']}: {int(design['value'])} "
        f"{design['operator']} {int(design['threshold'])} -> "
        f"{'PASS' if design['pass'] else 'FAIL'}"
    )
    for method in METHODS:
        lines.append(f"- `{method}`: `{'PASS' if all_pass[method] else 'FAIL'}`")
        for result in [row for row in acceptance if row["method"] == method]:
            lines.append(
                f"  - {result['check']}: {float(result['value']):.6g} "
                f"{result['operator']} {float(result['threshold']):.6g} -> "
                f"{'PASS' if result['pass'] else 'FAIL'}"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A PASS applies only to the listed mask endpoints. Topology, temporal motion, and vein/branch measurements must be interpreted from their own agreement tables.",
            "The final claim is ineligible if this is not an untouched `test` set generated under a valid algorithm lock.",
            "Any parameter change after viewing these test results requires a new independent test set.",
            "",
            "## Human annotation agreement",
            "",
            f"- Inter-rater mean Dice: `{mean_metric(inter_rater, 'dice'):.4f}`",
            f"- Inter-rater mean clDice: `{mean_metric(inter_rater, 'cldice'):.4f}`",
            f"- Intra-rater comparisons: `{len(intra_rater)}`",
            "",
            "See CSV outputs, Bland-Altman plots, and `failure_gallery/` for the complete audit trail.",
        ]
    )
    (output_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(config_path: Path, role: str, overwrite: bool) -> Path:
    config = load_json(config_path)
    root = package_root(config, config_path, role)
    manifest_path = root / "private_manifest.csv"
    metadata_path = root / "predictions" / "prediction_run_metadata.json"
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Prepare the package and generate predictions before evaluation")
    rows = read_csv(manifest_path)
    rows.sort(key=lambda row: (row["video_id"], int(row["source_frame_idx"])))
    verify_inputs(root, rows, config)
    prediction_metadata = load_json(metadata_path)
    output_dir = root / "evaluation"
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Evaluation already exists: {output_dir}. Use --overwrite to reproduce it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics, inter_rater = evaluate_frame_masks(root, rows, config)
    intra_rater = evaluate_intra_rater(root, rows, config)
    temporal_metrics = evaluate_temporal(root, rows)
    summary, phase_summary, video_summary = summarize_methods(frame_metrics, config)
    paired = paired_comparison(frame_metrics, config)
    acceptance = acceptance_results(summary, frame_metrics, inter_rater, config)

    write_csv(output_dir / "frame_metrics.csv", frame_metrics)
    write_csv(output_dir / "temporal_metrics.csv", temporal_metrics)
    write_csv(output_dir / "inter_rater_metrics.csv", inter_rater)
    write_csv(output_dir / "intra_rater_metrics.csv", intra_rater)
    write_csv(output_dir / "summary_by_method.csv", summary)
    write_csv(output_dir / "summary_by_phase.csv", phase_summary)
    write_csv(output_dir / "summary_by_biological_replicate.csv", video_summary)
    write_csv(output_dir / "paired_method_comparison.csv", paired)
    write_csv(output_dir / "acceptance_summary.csv", acceptance)
    bland_altman_plots(frame_metrics, output_dir)
    failure_gallery(
        root,
        frame_metrics,
        int(config["evaluation"]["failure_gallery_count"]),
        output_dir / "failure_gallery",
    )
    write_report(
        output_dir,
        config,
        role,
        prediction_metadata,
        rows,
        acceptance,
        inter_rater,
        intra_rater,
    )
    all_pass = {
        method: all(bool(row["pass"]) for row in acceptance if row["method"] == method)
        for method in METHODS
    }
    design_pass = all(
        bool(row["pass"]) for row in acceptance if row["method"] == "study_design"
    )
    primary_method = str(config["evaluation"]["primary_method"])
    if primary_method not in METHODS:
        raise ValueError(f"Unknown evaluation.primary_method: {primary_method}")
    write_json(
        output_dir / "evaluation_status.json",
        {
            "dataset_role": role,
            "publication_eligible_prediction_run": bool(
                prediction_metadata.get("publication_eligible")
            ),
            "study_design_acceptance": design_pass,
            "primary_method": primary_method,
            "method_acceptance": all_pass,
            "evaluation_code_sha256": sha256_file(Path(__file__).resolve()),
            "metric_code_sha256": sha256_file(
                Path(__file__).resolve().parent / "validation_metrics.py"
            ),
            "config_sha256": sha256_file(config_path),
            "overall_publication_claim_status": (
                "ELIGIBLE_AND_PASSED"
                if role == "test"
                and prediction_metadata.get("publication_eligible")
                and design_pass
                and all_pass[primary_method]
                else "NOT_ESTABLISHED"
            ),
            "important": (
                "Acceptance is endpoint-specific. A mask pass does not automatically "
                "validate skeleton, branchpoint, endpoint, vein, or biological claims."
            ),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate(args.config.resolve(), args.role, args.overwrite)
    print(f"Saved validation report: {output}")


if __name__ == "__main__":
    main()
