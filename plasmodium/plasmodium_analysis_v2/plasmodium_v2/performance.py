"""CPU and memory-aware execution tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass

import psutil

from .config import AnalysisSettings


@dataclass(frozen=True)
class PerformancePlan:
    cpu_count: int
    workers: int
    batch_frames: int
    available_memory_gb: float
    memory_budget_gb: float


def build_performance_plan(settings: AnalysisSettings) -> PerformancePlan:
    cpu_count = max(1, int(os.cpu_count() or 1))
    available = int(psutil.virtual_memory().available)
    memory_budget = max(512 * 1024**2, int(available * settings.memory_fraction))

    if settings.cpu_workers > 0:
        workers = int(settings.cpu_workers)
    else:
        workers = max(1, cpu_count - max(0, int(settings.reserve_cpu_cores)))
    workers = min(workers, cpu_count, max(1, int(settings.max_cpu_workers)))

    # A decoded 2560x1920 BGR frame is about 14 MiB. During segmentation each
    # worker may temporarily hold several float/bool arrays, so reserve 6x that
    # amount per in-flight frame rather than filling RAM aggressively.
    conservative_frame_bytes = 90 * 1024**2
    memory_limited_frames = max(1, memory_budget // conservative_frame_bytes)
    if settings.parallel_batch_frames > 0:
        batch_frames = int(settings.parallel_batch_frames)
    else:
        batch_frames = min(max(workers, workers * 2), int(memory_limited_frames))
    batch_frames = max(workers, batch_frames)

    return PerformancePlan(
        cpu_count=cpu_count,
        workers=workers,
        batch_frames=batch_frames,
        available_memory_gb=available / 1024**3,
        memory_budget_gb=memory_budget / 1024**3,
    )

