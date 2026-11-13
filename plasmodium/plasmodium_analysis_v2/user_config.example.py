"""Copy to user_config.py and set the input and acquisition calibration."""
from pathlib import Path
from plasmodium_v2 import AnalysisSettings

SETTINGS = AnalysisSettings(
    input_path=Path("PATH_TO_YOUR_VIDEO.mp4"),
    output_root=Path(__file__).resolve().parent / "runs",
    run_mode="validation_all",
    resume_run_directory=None,
    # Examples only: replace with this acquisition's actual timing and geometry.
    frame_interval_sec=10.0,
    pixel_size=1.0,  # Uncalibrated: report pixel units, not physical units.
    dish_center=(1280, 920),
    dish_radius=680,
    cpu_workers=0,
    parallel_batch_frames=0,
    memory_fraction=0.55,
)
