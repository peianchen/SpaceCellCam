"""PyQt5 desktop UI for configuring and running Plasmodium Analysis V2."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sys
from pathlib import Path

import psutil
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QCloseEvent, QFont
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from .config import AnalysisSettings, VALID_RUN_MODES
from .performance import build_performance_plan
from .run_context import finalize_metadata
from .ui_worker import run_analysis_worker

MODE_LABELS = {
    "validation_all": "三段驗證（前／中／後）",
    "validation_front": "前段驗證", "validation_middle": "中段驗證",
    "validation_rear": "後段驗證", "formal_only": "正式全影片分析",
    "formal_with_validation": "正式分析＋三段驗證", "contraction_roi": "ROI 收縮分析",
    "manual_validation_export": "匯出人工標註資料", "manual_validation_annotate": "人工遮罩標註",
    "manual_validation_score": "人工遮罩評分", "complete": "完整流程",
}


class PathRow(QWidget):
    def __init__(self, choose_callback):
        super().__init__()
        self.edit = QLineEdit()
        button = QPushButton("瀏覽…")
        button.clicked.connect(choose_callback)
        layout = QHBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1); layout.addWidget(button)


class AnalysisWindow(QMainWindow):
    def __init__(self, defaults: AnalysisSettings, settings_file: Path):
        super().__init__()
        self.settings_file = settings_file
        self.process = None
        self.message_queue = None
        self.current_output = None
        self.current_manifest = None
        self.widgets = {}
        self._setup_window()
        self._build_ui(defaults)
        self._load_settings()
        self._update_mode_hint()
        self._performance_preview()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_worker)
        self.poll_timer.start(120)

    def _setup_window(self):
        self.setWindowTitle("Plasmodium Bright-field Analysis V2")
        self.resize(1080, 780)
        self.setMinimumSize(900, 660)
        self.setStyleSheet("""
            QMainWindow { background: #f4f6f8; }
            QLabel#title { font-size: 24px; font-weight: 700; color: #17324d; }
            QLabel#subtitle { color: #5c6b78; margin-bottom: 4px; }
            QFrame#card { background: white; border: 1px solid #dce3e8; border-radius: 8px; }
            QLabel#section { font-size: 15px; font-weight: 700; color: #263b4d; }
            QLineEdit, QComboBox { min-height: 28px; padding: 2px 6px; border: 1px solid #bfcbd4; border-radius: 4px; background: white; }
            QPushButton { min-height: 30px; padding: 2px 12px; border: 1px solid #b8c5cf; border-radius: 5px; background: #ffffff; }
            QPushButton:hover { background: #edf4f8; }
            QPushButton#start { background: #1976d2; color: white; border: 1px solid #1976d2; font-weight: 700; }
            QPushButton#start:hover { background: #1565c0; }
            QPushButton#stop { color: #b3261e; }
            QTabWidget::pane { border: 1px solid #d6dfe5; background: white; }
            QTabBar::tab { padding: 9px 18px; }
            QPlainTextEdit { background: #17212b; color: #d9e2e8; font-family: Consolas; border: 0; }
        """)

    def _build_ui(self, defaults):
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central); outer.setContentsMargins(18, 16, 18, 14); outer.setSpacing(9)
        title = QLabel("Plasmodium 影像分析"); title.setObjectName("title"); outer.addWidget(title)
        subtitle = QLabel("選好影片與模式後開始；分析在獨立程序執行，介面可持續顯示日誌。")
        subtitle.setObjectName("subtitle"); outer.addWidget(subtitle)
        self.tabs = QTabWidget(); outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._basic_tab(defaults), "基本設定")
        self.tabs.addTab(self._advanced_tab(defaults), "效能與驗證")
        self.tabs.addTab(self._log_tab(), "執行日誌")
        buttons = QHBoxLayout()
        self.start_button = QPushButton("開始分析"); self.start_button.setObjectName("start"); self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("停止"); self.stop_button.setObjectName("stop"); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self._stop)
        open_button = QPushButton("開啟結果資料夾"); open_button.clicked.connect(self._open_output)
        save_button = QPushButton("儲存介面設定"); save_button.clicked.connect(self._save_settings)
        buttons.addWidget(self.start_button); buttons.addWidget(self.stop_button); buttons.addWidget(open_button); buttons.addStretch(); buttons.addWidget(save_button)
        outer.addLayout(buttons)
        self.status = QLabel("就緒"); outer.addWidget(self.status)

    def _scroll_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(); layout = QVBoxLayout(body); layout.setContentsMargins(12, 12, 12, 12); layout.setSpacing(12); layout.addStretch()
        scroll.setWidget(body)
        return scroll, layout

    def _card(self, title):
        card = QFrame(); card.setObjectName("card")
        layout = QVBoxLayout(card); layout.setContentsMargins(14, 12, 14, 14)
        heading = QLabel(title); heading.setObjectName("section"); layout.addWidget(heading)
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight); form.setHorizontalSpacing(14); form.setVerticalSpacing(8); layout.addLayout(form)
        return card, form

    def _line(self, key, value):
        widget = QLineEdit("" if value is None else str(value)); self.widgets[key] = widget; return widget

    def _check(self, key, value, text):
        widget = QCheckBox(text); widget.setChecked(bool(value)); self.widgets[key] = widget; return widget

    def _path_widget(self, key, value, callback):
        row = PathRow(callback); row.edit.setText("" if value is None else str(value)); self.widgets[key] = row.edit; return row

    def _basic_tab(self, d):
        scroll, layout = self._scroll_tab(); layout.takeAt(layout.count() - 1)
        card, form = self._card("輸入與輸出")
        mode = QComboBox(); mode.addItems(["video", "folder"]); mode.setCurrentText(d.input_mode); self.widgets["input_mode"] = mode
        form.addRow("輸入類型", mode)
        form.addRow("影片／影像資料夾", self._path_widget("input_path", d.input_path, self._browse_input))
        form.addRow("輸出根目錄", self._path_widget("output_root", d.output_root, self._browse_output))
        form.addRow("續接執行資料夾", self._path_widget("resume_run_directory", d.resume_run_directory, self._browse_resume))
        layout.addWidget(card)
        card, form = self._card("分析設定")
        run_mode = QComboBox()
        for item in sorted(VALID_RUN_MODES): run_mode.addItem(f"{item} — {MODE_LABELS.get(item, item)}", item)
        run_mode.setCurrentIndex(max(0, run_mode.findData(d.run_mode))); run_mode.currentIndexChanged.connect(self._update_mode_hint); self.widgets["run_mode"] = run_mode
        mode_box = QWidget(); mode_layout = QVBoxLayout(mode_box); mode_layout.setContentsMargins(0, 0, 0, 0); mode_layout.addWidget(run_mode)
        self.mode_hint = QLabel(); self.mode_hint.setStyleSheet("color:#647582"); mode_layout.addWidget(self.mode_hint)
        form.addRow("執行模式", mode_box)
        form.addRow("執行標籤", self._line("run_label", d.run_label))
        form.addRow("每幀間隔（秒）", self._line("frame_interval_sec", d.frame_interval_sec))
        form.addRow("像素尺寸", self._line("pixel_size", d.pixel_size))
        center = QWidget(); center_layout = QHBoxLayout(center); center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._line("dish_center_x", d.dish_center[0])); center_layout.addWidget(QLabel("/")); center_layout.addWidget(self._line("dish_center_y", d.dish_center[1]))
        form.addRow("培養皿中心 X / Y", center)
        form.addRow("培養皿半徑", self._line("dish_radius", d.dish_radius))
        form.addRow("正式分析時間上限（秒）", self._line("formal_max_time_sec", d.formal_max_time_sec))
        form.addRow("", self._check("unique_run_directory", d.unique_run_directory, "每次建立獨立結果資料夾"))
        layout.addWidget(card); layout.addStretch()
        return scroll

    def _advanced_tab(self, d):
        scroll, layout = self._scroll_tab(); layout.takeAt(layout.count() - 1)
        card, form = self._card("CPU 與記憶體")
        for label, key, value in (("CPU workers（0＝自動）", "cpu_workers", d.cpu_workers), ("最多 workers", "max_cpu_workers", d.max_cpu_workers), ("保留 CPU 核心", "reserve_cpu_cores", d.reserve_cpu_cores), ("每批幀數（0＝自動）", "parallel_batch_frames", d.parallel_batch_frames), ("可用記憶體比例", "memory_fraction", d.memory_fraction)):
            form.addRow(label, self._line(key, value))
        form.addRow("", self._check("low_memory_video_stream", d.low_memory_video_stream, "低記憶體影片串流"))
        preview_box = QWidget(); row = QHBoxLayout(preview_box); row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("更新資源預估"); button.clicked.connect(self._performance_preview); self.performance_label = QLabel(); row.addWidget(button); row.addWidget(self.performance_label, 1)
        form.addRow("", preview_box); layout.addWidget(card)
        card, form = self._card("驗證輸出")
        for label, key, value in (("每段驗證幀數", "validation_section_frames", d.validation_section_frames), ("驗證影片輸出縮放", "validation_video_scale", d.validation_video_scale), ("每段人工樣本數", "manual_samples_per_section", d.manual_samples_per_section), ("Temporal 邊緣排除幀數", "temporal_skip_edge_frames", d.temporal_skip_edge_frames)):
            form.addRow(label, self._line(key, value))
        layout.addWidget(card); layout.addStretch()
        return scroll

    def _log_tab(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(8, 8, 8, 8)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 9)); layout.addWidget(self.log, 1)
        clear = QPushButton("清除日誌"); clear.clicked.connect(self.log.clear); line = QHBoxLayout(); line.addStretch(); line.addWidget(clear); layout.addLayout(line)
        return page

    def _value(self, key):
        widget = self.widgets[key]
        if isinstance(widget, QCheckBox): return widget.isChecked()
        if isinstance(widget, QComboBox): return widget.currentData() if key == "run_mode" else widget.currentText()
        return widget.text()

    def _set_value(self, key, value):
        widget = self.widgets[key]
        if isinstance(widget, QCheckBox): widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            index = widget.findData(value) if key == "run_mode" else widget.findText(str(value))
            if index >= 0: widget.setCurrentIndex(index)
        else: widget.setText("" if value is None else str(value))

    @staticmethod
    def _number(value, label, kind=float, optional=False):
        if optional and not str(value).strip(): return None
        try: return kind(value)
        except ValueError as exc: raise ValueError(f"{label} 格式錯誤") from exc

    def form_settings(self, validate=True):
        g = self._value; resume = str(g("resume_run_directory")).strip()
        settings = AnalysisSettings(
            input_path=Path(str(g("input_path")).strip()), output_root=Path(str(g("output_root")).strip()),
            run_mode=str(g("run_mode")), input_mode=str(g("input_mode")), run_label=str(g("run_label")).strip(),
            unique_run_directory=bool(g("unique_run_directory")), resume_run_directory=Path(resume) if resume else None,
            frame_interval_sec=self._number(g("frame_interval_sec"), "每幀間隔"), pixel_size=self._number(g("pixel_size"), "像素尺寸"),
            dish_center=(self._number(g("dish_center_x"), "中心 X", int), self._number(g("dish_center_y"), "中心 Y", int)), dish_radius=self._number(g("dish_radius"), "培養皿半徑", int),
            formal_max_time_sec=self._number(g("formal_max_time_sec"), "時間上限", optional=True), cpu_workers=self._number(g("cpu_workers"), "CPU workers", int),
            max_cpu_workers=self._number(g("max_cpu_workers"), "最多 workers", int), reserve_cpu_cores=self._number(g("reserve_cpu_cores"), "保留核心", int),
            parallel_batch_frames=self._number(g("parallel_batch_frames"), "每批幀數", int), memory_fraction=self._number(g("memory_fraction"), "記憶體比例"),
            low_memory_video_stream=bool(g("low_memory_video_stream")), validation_section_frames=self._number(g("validation_section_frames"), "驗證幀數", int),
            validation_video_scale=self._number(g("validation_video_scale"), "驗證影片縮放"), manual_samples_per_section=self._number(g("manual_samples_per_section"), "人工樣本數", int),
            temporal_skip_edge_frames=self._number(g("temporal_skip_edge_frames"), "邊緣排除幀數", int))
        if validate: settings.validate()
        return settings

    def _update_mode_hint(self): self.mode_hint.setText(MODE_LABELS.get(str(self._value("run_mode")), ""))

    def _browse_input(self):
        if self._value("input_mode") == "folder": selected = QFileDialog.getExistingDirectory(self, "選擇影像資料夾")
        else: selected, _ = QFileDialog.getOpenFileName(self, "選擇影片", "", "影片 (*.mp4 *.avi *.mov *.mkv);;全部檔案 (*.*)")
        if selected: self._set_value("input_path", selected)

    def _browse_output(self):
        selected = QFileDialog.getExistingDirectory(self, "選擇輸出根目錄")
        if selected: self._set_value("output_root", selected)

    def _browse_resume(self):
        selected = QFileDialog.getExistingDirectory(self, "選擇先前的執行資料夾")
        if selected: self._set_value("resume_run_directory", selected)

    def _performance_preview(self):
        try:
            p = build_performance_plan(self.form_settings(False))
            self.performance_label.setText(f"{p.workers}/{p.cpu_count} workers；batch {p.batch_frames}；RAM {p.memory_budget_gb:.1f} GiB")
        except Exception as exc: self.performance_label.setText(str(exc))

    def _append_log(self, text):
        cursor = self.log.textCursor(); cursor.movePosition(cursor.End); cursor.insertText(text); self.log.setTextCursor(cursor); self.log.ensureCursorVisible()

    def _start(self):
        if self.process is not None and self.process.is_alive(): return
        try: settings = self.form_settings(True)
        except Exception as exc: QMessageBox.critical(self, "設定錯誤", str(exc)); return
        self._save_settings(silent=True); self.current_output = self.current_manifest = None
        ctx = mp.get_context("spawn"); self.message_queue = ctx.Queue()
        self.process = ctx.Process(target=run_analysis_worker, args=(settings, self.message_queue), name="plasmodium-analysis")
        self.process.start(); self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.status.setText("分析執行中…"); self._append_log("\n=== 開始新的分析 ===\n"); self.tabs.setCurrentIndex(2)

    def _poll_worker(self):
        if self.message_queue is not None:
            while True:
                try: message = self.message_queue.get_nowait()
                except queue.Empty: break
                self._handle_message(message)
        if self.process is not None and not self.process.is_alive(): self.process.join(timeout=.1); self._set_idle()

    def _handle_message(self, msg):
        kind = msg.get("type")
        if kind == "log": self._append_log(str(msg.get("text", "")))
        elif kind == "event" and msg.get("event") == "output_created":
            self.current_output, self.current_manifest = Path(msg["output_directory"]), Path(msg["manifest_path"]); self.status.setText(f"執行中：{self.current_output.name}")
        elif kind == "done":
            self.current_output = Path(msg["output_directory"]); self.status.setText(f"完成：{self.current_output.name}"); self._append_log("\n=== 分析完成 ===\n")
        elif kind == "error":
            self.status.setText("分析失敗"); self._append_log("\n" + str(msg.get("traceback", msg.get("error"))) + "\n"); QMessageBox.critical(self, "分析失敗", str(msg.get("error", "未知錯誤")))

    def _set_idle(self): self.start_button.setEnabled(True); self.stop_button.setEnabled(False); self.process = None

    def _stop(self):
        if self.process is None or not self.process.is_alive(): return
        if QMessageBox.question(self, "停止分析", "確定要停止目前分析嗎？") != QMessageBox.Yes: return
        self._terminate_worker(); self.status.setText("已由使用者停止"); self._append_log("\n=== 已停止分析 ===\n")

    def _terminate_worker(self):
        # multiprocessing.Process.terminate() only stops the direct worker on
        # Windows. Terminate its nested ProcessPool descendants first so they
        # cannot remain alive and keep several GiB of frame arrays allocated.
        try:
            parent = psutil.Process(self.process.pid)
            descendants = parent.children(recursive=True)
            for child in descendants:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            _, alive = psutil.wait_procs(descendants, timeout=3)
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        self.process.terminate(); self.process.join(timeout=5)
        if self.current_manifest and self.current_manifest.exists():
            try: finalize_metadata(self.current_manifest, "cancelled", "Stopped from PyQt UI")
            except Exception as exc: self._append_log(f"\n無法更新停止狀態：{exc}\n")
        self._set_idle()

    def _open_output(self):
        path = self.current_output or Path(str(self._value("output_root")))
        if path.exists(): os.startfile(str(path))
        else: QMessageBox.information(self, "資料夾不存在", str(path))

    def _save_settings(self, _checked=False, silent=False):
        try:
            self.settings_file.write_text(json.dumps({key: self._value(key) for key in self.widgets}, ensure_ascii=False, indent=2), encoding="utf-8")
            if not silent: self.status.setText(f"設定已儲存：{self.settings_file.name}")
        except Exception as exc:
            if not silent: QMessageBox.critical(self, "無法儲存設定", str(exc))

    def _load_settings(self):
        if not self.settings_file.exists(): return
        try:
            for key, value in json.loads(self.settings_file.read_text(encoding="utf-8")).items():
                if key in self.widgets: self._set_value(key, value)
        except Exception: pass

    def closeEvent(self, event: QCloseEvent):
        if self.process is not None and self.process.is_alive():
            if QMessageBox.question(self, "結束程式", "分析仍在執行，確定要停止並關閉嗎？") != QMessageBox.Yes: event.ignore(); return
            self._terminate_worker()
        event.accept()


def launch_ui(defaults: AnalysisSettings, settings_file: Path | None = None):
    existing_app = QApplication.instance()
    app = existing_app or QApplication(sys.argv)
    window = AnalysisWindow(defaults, settings_file or Path(__file__).resolve().parents[1] / "ui_settings.json")
    window.show()
    # Spyder may already own the Qt event loop; do not start a nested loop.
    if existing_app is None:
        app.exec_()
    return window
