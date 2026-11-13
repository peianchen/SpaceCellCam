import cv2
import numpy as np
import pandas as pd

from plasmodium_v2 import engine


def _write_video(path, count=6, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, size)
    assert writer.isOpened()
    for index in range(count):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(frame, (12 + index, 14), (26 + index, 28), (40, 160, 210), -1)
        writer.write(frame)
    writer.release()


def test_video_frame_range_is_lazy_reopenable_and_tracker_compatible(tmp_path):
    video = tmp_path / "tiny.avi"
    _write_video(video)
    frames = engine._VideoFrameRange(video, 1, 6)
    assert len(frames) == 5
    assert frames[0].shape == (48, 64, 3)
    assert len(list(frames[:2])) == 2
    assert len(list(frames)) == 5

    df = pd.DataFrame(index=range(5))
    summary = engine.add_tracker_contraction_metrics_to_df(
        df, frames, [("roi_0", 10, 10, 18, 20)], dt=10.0,
        min_period_sec=20.0, max_period_sec=100.0, min_valid_samples=2,
        search_radius=8, min_match_score=0.0, save_plots=False,
    )
    assert "tracker_roi_0_x" in df.columns
    assert len(summary) == 4
