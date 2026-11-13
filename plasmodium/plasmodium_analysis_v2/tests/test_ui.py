import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from plasmodium_v2.config import AnalysisSettings
from plasmodium_v2.ui_app import AnalysisWindow


def test_pyqt_ui_builds_settings_and_performance_preview(tmp_path):
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"placeholder")
    defaults = AnalysisSettings(input_path=video, output_root=tmp_path / "runs")
    window = AnalysisWindow(defaults, tmp_path / "ui_settings.json")
    settings = window.form_settings(validate=True)
    assert settings.input_path == video
    assert settings.run_mode == "validation_all"
    assert "workers" in window.performance_label.text()
    window.close()
    app.processEvents()
