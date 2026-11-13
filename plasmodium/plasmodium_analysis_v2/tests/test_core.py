from pathlib import Path

import cv2
import numpy as np

from plasmodium_v2.config import AnalysisSettings
from plasmodium_v2.performance import build_performance_plan
from plasmodium_v2.segmentation import recover_vesselness_near_body
from plasmodium_v2.temporal import (
    apply_temporal_body_recovery_to_sequence,
    required_temporal_halo,
)


def test_temporal_smoothing_never_deletes_current_strict_pixel():
    items = []
    for index in range(5):
        strict = np.zeros((9, 9), dtype=bool)
        strict[4, 4] = True
        if index == 2:
            strict[4, 5] = True
        candidate = np.zeros_like(strict)
        candidate[4, 6] = True
        items.append({"frame_idx": index, "frame": np.zeros((9, 9, 3), np.uint8), "mask": strict, "candidate": candidate})
    recovered = apply_temporal_body_recovery_to_sequence(
        items, window=3, min_hits=2, final_smooth_window=3, final_smooth_min_hits=2
    )
    assert recovered[2]["mask"][4, 5]


def test_composed_temporal_halo():
    assert required_temporal_halo(5, 5) == 4
    assert required_temporal_halo(5, 1) == 2


def test_temporal_motion_compensation_recovers_shifted_thin_branch():
    size = 128
    center = (64, 64)
    radius = 44
    yy, xx = np.ogrid[:size, :size]
    distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    rng = np.random.default_rng(1234)
    base = np.zeros((size, size), np.uint8)
    annulus = (distance >= radius * 0.70) & (distance <= radius * 1.18)
    base[annulus] = rng.integers(25, 240, size=int(annulus.sum()), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)

    strict_base = np.zeros((size, size), dtype=np.uint8)
    strict_base[61:67, 58:66] = 1
    candidate_base = np.zeros_like(strict_base)
    candidate_base[62:66, 66:72] = 1
    items = []
    for index, shift_x in enumerate((-8, -4, 0, 4, 8)):
        matrix = np.float32([[1, 0, shift_x], [0, 1, 0]])
        frame_gray = cv2.warpAffine(base, matrix, (size, size))
        strict = cv2.warpAffine(
            strict_base, matrix, (size, size), flags=cv2.INTER_NEAREST
        ).astype(bool)
        candidate = cv2.warpAffine(
            candidate_base, matrix, (size, size), flags=cv2.INTER_NEAREST
        ).astype(bool)
        items.append(
            {
                "frame_idx": index,
                "frame": cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR),
                "mask": strict,
                "candidate": candidate,
                "dish_center": center,
                "dish_radius": radius,
            }
        )

    uncompensated = apply_temporal_body_recovery_to_sequence(
        items, window=5, min_hits=4, connect_radius=3,
        motion_compensation=False,
    )
    compensated = apply_temporal_body_recovery_to_sequence(
        items, window=5, min_hits=4, connect_radius=3,
        motion_compensation=True, motion_max_shift_px=12,
    )
    assert not uncompensated[2]["mask"][63, 68]
    assert compensated[2]["mask"][63, 68]
    assert compensated[2]["motion_compensation_applied"]


def test_vesselness_hard_cap_is_never_exceeded(monkeypatch):
    from plasmodium_v2 import segmentation

    monkeypatch.setattr(segmentation.exposure, "equalize_adapthist", lambda image, clip_limit: image)
    monkeypatch.setattr(
        segmentation.filters,
        "frangi",
        lambda image, sigmas, black_ridges: np.ones_like(image, dtype=float),
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:] = (40, 150, 150)
    body = np.zeros((64, 64), dtype=bool)
    body[28:36, 28:36] = True
    added, _, _, _ = recover_vesselness_near_body(
        frame,
        body,
        dish_center=(32, 32),
        dish_radius=30,
        percentile=50,
        min_response=0,
        connect_radius=8,
        min_obj_size=1,
        max_added_fraction=0.02,
        chroma_min=0,
        min_channel=0,
        max_value=255,
    )
    assert int(added.sum()) <= int(np.floor(body.sum() * 0.02))


def test_auto_performance_plan_is_bounded(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    settings = AnalysisSettings(input_path=source, max_cpu_workers=8)
    plan = build_performance_plan(settings)
    assert 1 <= plan.workers <= 8
    assert plan.batch_frames >= plan.workers
