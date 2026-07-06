"""Run a small end-to-end software test; this is not biological validation."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import load_json, write_json
from evaluate_validation import evaluate
from freeze_algorithm import freeze
from generate_predictions import generate
from prepare_dataset import build_package


def create_video(path: Path, total_frames: int = 150) -> None:
    size = (160, 120)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        15.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic test video")
    center = (80, 60)
    try:
        for frame_idx in range(total_frames):
            image = np.full((size[1], size[0], 3), (25, 25, 25), dtype=np.uint8)
            cv2.circle(image, center, 50, (112, 118, 122), -1)
            cv2.circle(image, center, 13, (25, 200, 230), -1)
            growth = 22 + int(round(8 * frame_idx / max(total_frames - 1, 1)))
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                wobble = 0.12 * np.sin(frame_idx / 18.0 + angle * 3.0)
                endpoint = (
                    center[0] + int(round(growth * np.cos(angle + wobble))),
                    center[1] + int(round(growth * np.sin(angle + wobble))),
                )
                cv2.line(image, center, endpoint, (25, 205, 235), 2)
                cv2.circle(image, endpoint, 2, (35, 215, 240), -1)
            writer.write(image)
    finally:
        writer.release()


def write_video_table(path: Path, video_path: Path) -> None:
    fields = [
        "video_id",
        "video_path",
        "dataset_role",
        "biological_replicate",
        "condition",
        "acquisition_date",
        "dish_center_x",
        "dish_center_y",
        "dish_radius",
        "pixel_size",
        "frame_interval_s",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "synthetic_video",
                "video_path": str(video_path),
                "dataset_role": "development",
                "biological_replicate": "synthetic_rep_01",
                "condition": "software_test_only",
                "acquisition_date": "",
                "dish_center_x": 80,
                "dish_center_y": 60,
                "dish_radius": 50,
                "pixel_size": 1.0,
                "frame_interval_s": 10.0,
            }
        )


def populate_annotations(package: Path) -> None:
    with (package / "private_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    for row in manifest:
        bid = row["blind_id"]
        source = package / "predictions" / "temporal" / f"{bid}.png"
        for target in (
            package / "annotators" / "annotator_A" / "masks" / f"{bid}.png",
            package / "annotators" / "annotator_B" / "masks" / f"{bid}.png",
            package / "adjudicated_masks" / f"{bid}.png",
        ):
            shutil.copy2(source, target)
    repeat_path = package / "private_repeat_manifest.csv"
    with repeat_path.open("r", encoding="utf-8-sig", newline="") as handle:
        repeats = list(csv.DictReader(handle))
    for row in repeats:
        source_id = row["primary_blind_id"]
        repeat_id = row["repeat_blind_id"]
        for annotator in ("annotator_A", "annotator_B"):
            source = package / "annotators" / annotator / "masks" / f"{source_id}.png"
            target = (
                package
                / "annotators"
                / f"{annotator}_repeat"
                / "masks"
                / f"{repeat_id}.png"
            )
            shutil.copy2(source, target)


def run_smoke_test(workspace: Path) -> Path:
    script_dir = Path(__file__).resolve().parent
    source_config = load_json(script_dir / "config.json")
    source_config["output_root"] = str(workspace / "packages")
    source_config["algorithm"]["source_file"] = str(
        script_dir.parent / "plasmodium_brightfield_analysis.py"
    )
    source_config["algorithm"]["prediction_runner_file"] = str(
        script_dir / "generate_predictions.py"
    )
    source_config["algorithm"]["body_hsv_params"].update(
        {
            "min_obj_size": 10,
            "body_min_hole_size": 5,
            "body_max_hole_size": 20,
            "yellow_chroma_rim_exclusion_px": 3,
            "rim_exclusion_px": 5,
        }
    )
    source_config["algorithm"]["temporal_params"].update(
        {
            "rim_exclusion_px": 3,
            "connect_radius": 3,
            "min_obj_size": 2,
            "block_hole_max_area": 20,
        }
    )
    source_config["sampling"].update(
        {
            "frames_per_window": 5,
            "windows_per_phase": 2,
            "minimum_gap_between_windows": 2,
        }
    )
    source_config["evaluation"].update(
        {
            "cluster_bootstrap_iterations": 100,
            "failure_gallery_count": 2,
        }
    )
    config_path = workspace / "smoke_config.json"
    write_json(config_path, source_config)
    video_path = workspace / "synthetic.avi"
    create_video(video_path)
    videos_path = workspace / "videos.csv"
    write_video_table(videos_path, video_path)
    package = build_package(config_path, videos_path, "development", force=False)
    freeze(config_path)
    generate(
        config_path,
        "development",
        allow_development=False,
        overwrite=False,
    )
    populate_annotations(package)
    evaluation = evaluate(config_path, "development", overwrite=False)
    status = load_json(evaluation / "evaluation_status.json")
    if status["overall_publication_claim_status"] != "NOT_ESTABLISHED":
        raise AssertionError("Synthetic development data must never establish publication status")
    with (evaluation / "frame_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        metrics = list(csv.DictReader(handle))
    temporal_dice = [
        float(row["dice"]) for row in metrics if row["method"] == "temporal"
    ]
    if not temporal_dice or min(temporal_dice) < 0.999999:
        raise AssertionError("Synthetic temporal reference should reproduce exactly")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the temporary workspace and print its path.",
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parent / "_synthetic_smoke_workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        output = run_smoke_test(workspace)
        print(f"Synthetic end-to-end smoke test passed: {output}")
        print("This confirms software wiring only; it is not biological accuracy evidence.")
        if args.keep:
            print(f"Kept smoke-test workspace: {workspace}")
    finally:
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
