from pathlib import Path

import pytest

from plasmodium_v2.config import AnalysisSettings
from plasmodium_v2.run_context import create_output_directory


def test_manual_score_requires_explicit_existing_run(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    settings = AnalysisSettings(input_path=source, run_mode="manual_validation_score")
    with pytest.raises(ValueError, match="resume_run_directory"):
        settings.validate()


def test_resume_uses_existing_run_directory(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    existing = tmp_path / "previous_export"
    existing.mkdir()
    settings = AnalysisSettings(
        input_path=source,
        run_mode="manual_validation_score",
        resume_run_directory=existing,
    )
    settings.validate()
    assert create_output_directory(settings) == existing.resolve()

