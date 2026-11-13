# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 22:20:24 2025

@author: chen8
"""

# -*- coding: utf-8 -*-
"""
PyQt5 GUI：將分桶資料夾（如 b000, b001, ...）內的影像全部合併到單一資料夾
- 支援複製 / 搬移
- 自訂副檔名
- 進度條與日誌
"""

import sys
import re
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QCheckBox, QProgressBar, QTextEdit, QMessageBox, QFrame
)
from PyQt5.QtGui import QFont  # 調整全域字體用


# ================= 工具函式 =================

def is_image(p: Path, allowed_exts):
    return p.is_file() and p.suffix.lower() in allowed_exts

def guess_bucket_name(path: Path) -> str:
    """
    從路徑中找像 b000/b001 這種桶名（大小寫皆可）；找不到就用父層資料夾名。
    """
    for part in reversed(path.parts):
        if re.fullmatch(r"[bB]\d{3,}", part):
            return part
    return path.parent.name

def unique_dest(dest: Path) -> Path:
    """
    若目標檔名已存在，在檔名後自動加 (1), (2)...
    """
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand
        i += 1


# ================= 背景工作執行緒 =================

class FlattenWorker(QObject):
    progress = pyqtSignal(int, int)         # done, total
    log = pyqtSignal(str)                   # log text
    finished = pyqtSignal()                 # when done
    aborted = pyqtSignal(str)               # when aborted with reason

    def __init__(self, root_dir: Path, out_dir: Path, move_files: bool, exts: set):
        super().__init__()
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.move_files = move_files
        self.allowed_exts = exts
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if not self.root_dir.exists() or not self.root_dir.is_dir():
                self.aborted.emit("來源資料夾不存在或不可讀。")
                return
            self.out_dir.mkdir(parents=True, exist_ok=True)

            # 收集檔案
            files = [p for p in self.root_dir.rglob("*") if is_image(p, self.allowed_exts)]
            files.sort(key=lambda p: str(p).lower())
            total = len(files)
            done = 0

            if total == 0:
                self.log.emit("找不到符合副檔名的影像檔。")
                self.finished.emit()
                return

            mode = "搬移" if self.move_files else "複製"
            self.log.emit(f"開始處理：共 {total} 個檔案，模式：{mode}")

            for src in files:
                if self._stop:
                    self.aborted.emit("已中止。")
                    return

                try:
                    bucket = guess_bucket_name(src)
                    # 若要加桶名前綴可改成 f"{bucket}_{src.name}"
                    dest_name = f"{src.name}"
                    dest = unique_dest(self.out_dir / dest_name)

                    if self.move_files:
                        shutil.move(str(src), str(dest))
                    else:
                        shutil.copy2(str(src), str(dest))

                    self.log.emit(f"✓ {src}  →  {dest}")
                except Exception as e:
                    self.log.emit(f"[跳過] {src} 失敗：{e}")

                done += 1
                self.progress.emit(done, total)

            self.log.emit("完成。")
            self.finished.emit()

        except Exception as e:
            self.aborted.emit(f"發生錯誤：{e}")


# ================= 主視窗 =================

class MainWindow(QWidget):
    def __init__(self, ui_scale: float = 1.0):
        super().__init__()
        self.ui_scale = max(1.0, ui_scale)
        self.setWindowTitle("分桶影像合併器")

        base_w, base_h = 760, 520
        self.resize(int(base_w * self.ui_scale), int(base_h * self.ui_scale))

        # 控制項
        self.root_edit = QLineEdit()
        self.root_btn = QPushButton("選擇來源根資料夾…")

        self.out_edit = QLineEdit()
        self.out_btn = QPushButton("選擇輸出資料夾…")

        self.move_chk = QCheckBox("搬移檔案（預設：複製）")

        self.ext_edit = QLineEdit(".jpg,.jpeg,.png,.bmp,.tif,.tiff")
        self.ext_label = QLabel("副檔名（逗號分隔）：")

        self.start_btn = QPushButton("開始")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        # 版面
        layout = QVBoxLayout(self)

        # 來源
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("來源根資料夾："))
        row1.addWidget(self.root_edit)
        row1.addWidget(self.root_btn)
        layout.addLayout(row1)

        # 輸出
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("輸出資料夾："))
        row2.addWidget(self.out_edit)
        row2.addWidget(self.out_btn)
        layout.addLayout(row2)

        # 選項列
        row3 = QHBoxLayout()
        row3.addWidget(self.move_chk)
        row3.addSpacing(20)
        row3.addWidget(self.ext_label)
        row3.addWidget(self.ext_edit)
        layout.addLayout(row3)

        # 分隔線
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 控制列
        row4 = QHBoxLayout()
        row4.addWidget(self.start_btn)
        row4.addWidget(self.stop_btn)
        row4.addWidget(self.progress, 1)
        layout.addLayout(row4)

        # 日誌
        layout.addWidget(self.log_view, 1)

        # 元件最小高度/寬度跟著放大
        min_h = int(28 * self.ui_scale)
        btn_h = int(32 * self.ui_scale)
        self.root_edit.setMinimumHeight(min_h)
        self.out_edit.setMinimumHeight(min_h)
        self.ext_edit.setMinimumHeight(min_h)
        self.start_btn.setMinimumHeight(btn_h)
        self.stop_btn.setMinimumHeight(btn_h)
        self.progress.setMinimumHeight(int(18 * self.ui_scale))
        self.log_view.setMinimumHeight(int(200 * self.ui_scale))

        # 事件
        self.root_btn.clicked.connect(self.choose_root)
        self.out_btn.clicked.connect(self.choose_out)
        self.start_btn.clicked.connect(self.start_work)
        self.stop_btn.clicked.connect(self.stop_work)

        # 執行緒/工人
        self.thread = None
        self.worker = None

    def choose_root(self):
        path = QFileDialog.getExistingDirectory(self, "選擇來源根資料夾")
        if path:
            self.root_edit.setText(path)

    def choose_out(self):
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if path:
            self.out_edit.setText(path)

    def start_work(self):
        root = Path(self.root_edit.text().strip())
        out = Path(self.out_edit.text().strip())
        if not root.exists() or not root.is_dir():
            QMessageBox.warning(self, "提示", "請選擇有效的來源根資料夾。")
            return
        if not out:
            QMessageBox.warning(self, "提示", "請選擇有效的輸出資料夾。")
            return
        if root.resolve() == out.resolve():
            QMessageBox.warning(self, "提示", "輸出資料夾不可與來源根資料夾相同。")
            return

        # 副檔名集合
        exts_raw = [e.strip().lower() for e in self.ext_edit.text().split(",") if e.strip()]
        exts = set()
        for e in exts_raw:
            if not e.startswith("."):
                e = "." + e
            exts.add(e)

        self.log_view.clear()
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = FlattenWorker(root, out, self.move_chk.isChecked(), exts)
        self.worker.moveToThread(self.thread)

        # 連線
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.aborted.connect(self.on_aborted)

        # 清理
        self.worker.finished.connect(self.thread.quit)
        self.worker.aborted.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)

        self.thread.start()
        self.append_log(f"=== 來源：{root} ===")
        self.append_log(f"=== 輸出：{out} ===")
        self.append_log(f"模式：{'搬移' if self.move_chk.isChecked() else '複製'}")
        self.append_log(f"副檔名：{', '.join(sorted(exts))}")

    def stop_work(self):
        if self.worker:
            self.worker.stop()
            self.append_log("嘗試中止中…")

    def on_progress(self, done, total):
        if total > 0:
            pct = int(done * 100 / total)
            self.progress.setValue(pct)

    def append_log(self, text: str):
        self.log_view.append(text)

    def on_finished(self):
        self.append_log("✅ 全部完成。")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_aborted(self, reason: str):
        self.append_log(f"⛔ {reason}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def cleanup_thread(self):
        self.worker = None
        self.thread = None


# ---- 可選：在同一 Python 核心只套用一次全域樣式（避免反覆執行時重覆設定）----
_APPLIED_ONCE = False

def main():
    global _APPLIED_ONCE

    # ✅ 這兩個 attribute 必須在建立 QApplication 之前設定
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    UI_SCALE = 1.35     # <<< 這裡改倍率（1.2~1.5 常見）
    BASE_PT  = 10.0     # 基準字體大小（pt），避免累乘

    if not _APPLIED_ONCE:
        # 用「絕對大小」設定字體
        f = QFont()
        f.setPointSizeF(BASE_PT * UI_SCALE)
        app.setFont(f)

        # 全域樣式，讓元件 padding/高度一起放大
        pad = int(6 * UI_SCALE)
        padx = int(12 * UI_SCALE)
        pbar_h = int(18 * UI_SCALE)
        app.setStyleSheet(f"""
            QLineEdit, QTextEdit, QPlainTextEdit {{
                padding:{pad}px;
            }}
            QPushButton {{
                padding:{pad}px {padx}px;
            }}
            QProgressBar {{
                min-height:{pbar_h}px;
            }}
        """)
        _APPLIED_ONCE = True

    w = MainWindow(ui_scale=UI_SCALE)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
