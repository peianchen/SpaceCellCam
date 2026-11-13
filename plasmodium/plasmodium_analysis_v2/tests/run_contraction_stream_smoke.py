"""End-to-end smoke test for low-memory contraction ROI streaming."""
from __future__ import annotations

import builtins
from multiprocessing import freeze_support
from pathlib import Path

import cv2
import numpy as np

from plasmodium_v2 import AnalysisSettings, run_analysis
from plasmodium_v2 import engine


def main():
    root = Path(__file__).resolve().parent / "_contraction_smoke_output"
    root.mkdir(parents=True, exist_ok=True)
    video = root / "tiny_contraction.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (1024, 768))
    if not writer.isOpened():
        raise RuntimeError("Could not create contraction smoke video")
    for index in range(9):
        frame = np.full((768, 1024, 3), 35, dtype=np.uint8)
        radius = 24 + (index % 3)
        cv2.circle(frame, (512, 384), radius, (50, 155, 185), -1)
        cv2.rectangle(frame, (500 + index % 2, 372), (524 + index % 2, 396), (80, 190, 220), -1)
        writer.write(frame)
    writer.release()

    engine.cv2.selectROIs = lambda *args, **kwargs: np.array([[494, 366, 40, 40]])
    builtins.input = lambda *_args, **_kwargs: (_ for _ in ()).throw(EOFError())
    settings = AnalysisSettings(
        input_path=video, output_root=root / "runs", run_mode="contraction_roi",
        dish_center=(512, 384), dish_radius=350, cpu_workers=2,
        parallel_batch_frames=4, temporal_skip_edge_frames=2,
    )
    output = run_analysis(settings)
    assert (output / "selected_contraction_rois.json").exists()
    assert (output / "sampled_frames_all_metrics.csv").exists()
    for segment in ("early", "middle", "late"):
        assert (output / segment / f"sampled_frames_{segment}.csv").exists()
        assert (output / segment / f"overlay_{segment}.mp4").exists()
    print(f"CONTRACTION_STREAM_SMOKE_OK={output}")


if __name__ == "__main__":
    freeze_support()
    main()
