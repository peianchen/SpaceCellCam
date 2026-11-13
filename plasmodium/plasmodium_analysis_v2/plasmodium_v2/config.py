"""Typed settings for Spyder and batch execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


VALID_RUN_MODES = {
    "complete",
    "formal_only",
    "formal_with_validation",
    "validation_all",
    "validation_front",
    "validation_middle",
    "validation_rear",
    "contraction_roi",
    "manual_validation_export",
    "manual_validation_annotate",
    "manual_validation_score",
}


@dataclass(frozen=True)
class AnalysisSettings:
    """User-editable settings; zero workers/batch means automatic selection."""

    input_path: Path
    output_root: Path = Path("runs")
    run_mode: str = "validation_all"
    input_mode: str = "video"
    run_label: str = ""
    unique_run_directory: bool = True
    resume_run_directory: Path | None = None

    frame_interval_sec: float = 10.0
    pixel_size: float = 1.0
    dish_center: tuple[int, int] = (1280, 920)
    dish_radius: int = 680
    formal_max_time_sec: float | None = None

    cpu_workers: int = 0
    max_cpu_workers: int = 16
    reserve_cpu_cores: int = 1
    parallel_batch_frames: int = 0
    memory_fraction: float = 0.55
    low_memory_video_stream: bool = True

    validation_section_frames: int = 300
    validation_video_scale: float = 0.5
    manual_samples_per_section: int = 10
    temporal_skip_edge_frames: int = 10

    resolved_output_dir: Path | None = field(default=None, repr=False)
    resolved_cpu_workers: int | None = field(default=None, repr=False)
    resolved_batch_frames: int | None = field(default=None, repr=False)

    def validate(self) -> None:
        if self.run_mode not in VALID_RUN_MODES:
            raise ValueError(f"Unknown run_mode={self.run_mode!r}")
        if self.input_mode not in {"video", "folder"}:
            raise ValueError("input_mode must be 'video' or 'folder'")
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Input does not exist: {self.input_path}")
        if self.frame_interval_sec <= 0:
            raise ValueError("frame_interval_sec must be positive")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.dish_radius <= 0:
            raise ValueError("dish_radius must be positive")
        if not 0.1 <= float(self.validation_video_scale) <= 1.0:
            raise ValueError("validation_video_scale must be between 0.1 and 1.0")
        if not 0.1 <= float(self.memory_fraction) <= 0.9:
            raise ValueError("memory_fraction must be between 0.1 and 0.9")
        if self.resume_run_directory is not None and not Path(self.resume_run_directory).is_dir():
            raise FileNotFoundError(
                f"resume_run_directory does not exist: {self.resume_run_directory}"
            )
        if self.run_mode in {"manual_validation_annotate", "manual_validation_score"}:
            if self.resume_run_directory is None:
                raise ValueError(
                    f"{self.run_mode} must set resume_run_directory to the previous "
                    "manual_validation_export run directory"
                )

    def with_runtime(
        self,
        *,
        output_dir: Path,
        cpu_workers: int,
        batch_frames: int,
    ) -> "AnalysisSettings":
        return replace(
            self,
            resolved_output_dir=Path(output_dir),
            resolved_cpu_workers=int(cpu_workers),
            resolved_batch_frames=int(batch_frames),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data

