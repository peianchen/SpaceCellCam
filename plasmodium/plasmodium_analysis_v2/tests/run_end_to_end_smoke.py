"""Small real OpenCV + multiprocessing smoke test for the lbr environment."""

from __future__ import annotations

import shutil
from multiprocessing import freeze_support
from pathlib import Path

import cv2
import numpy as np

from plasmodium_v2 import AnalysisSettings, run_analysis


def make_video(path: Path, frame_count=14):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 128), True
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create smoke-test video")
    for index in range(frame_count):
        frame = np.full((128, 128, 3), 70, dtype=np.uint8)
        cv2.circle(frame, (52 + index // 3, 64), 24, (20, 190, 210), -1)
        cv2.line(frame, (68, 64), (95, 58 + index % 3), (25, 155, 180), 5)
        writer.write(frame)
    writer.release()


def main():
    root = Path(__file__).resolve().parent / "_smoke_output"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    video = root / "synthetic.mp4"
    make_video(video)
    settings = AnalysisSettings(
        input_path=video,
        output_root=root / "runs",
        run_mode="validation_front",
        frame_interval_sec=1.0,
        dish_center=(64, 64),
        dish_radius=58,
        validation_section_frames=5,
        validation_video_scale=0.5,
        temporal_skip_edge_frames=2,
        cpu_workers=2,
        parallel_batch_frames=4,
        max_cpu_workers=2,
    )
    output = run_analysis(settings)
    assert (output / "run_manifest.json").exists()
    assert (output / "validation_analysis" / "front" / "front_formal_metrics.csv").exists()
    assert (output / "validation_clips" / "front" / "front_color_mask_overlay.mp4").exists()
    print(f"SMOKE_OK={output}")


if __name__ == "__main__":
    freeze_support()
    main()

