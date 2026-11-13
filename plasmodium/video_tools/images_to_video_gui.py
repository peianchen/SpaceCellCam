# -*- coding: utf-8 -*-
"""
PyQt5 GUI：將資料夾中的照片做成序列影片（MP4/AVI）
- 起始/終點為「1 起算」
- 尺寸可選：自動以首張可讀影像、或自訂寬高
- letterbox 預設關閉
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QCheckBox, QProgressBar, QTextEdit, QMessageBox,
    QFrame, QSpinBox, QDoubleSpinBox, QComboBox
)

# ------------------ 公用函式 ------------------

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def load_image(path: Path):
    # 跨平台穩定讀圖；若你沒有中文路徑問題也可改回 cv2.imread
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def resize_to_fit(img, target_size: Tuple[int, int], letterbox: bool):
    tw, th = target_size
    ih, iw = img.shape[:2]
    if letterbox:
        scale = min(tw / iw, th / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        x = (tw - nw) // 2
        y = (th - nh) // 2
        canvas[y:y+nh, x:x+nw] = resized
        return canvas
    else:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA if (tw < iw or th < ih) else cv2.INTER_CUBIC)

def pick_fourcc(fmt: str):
    if fmt.upper() == "MP4":
        return cv2.VideoWriter_fourcc(*'mp4v'), ".mp4"
    else:
        return cv2.VideoWriter_fourcc(*'XVID'), ".avi"

# ------------------ 背景工作執行緒 ------------------

class EncodeWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    aborted = pyqtSignal(str)

    def __init__(self, folder: Path, out_path: Path, fps: float,
                 start_1based: int, end_1based: int, fmt: str,
                 letterbox: bool, custom_size: Optional[Tuple[int,int]]):
        super().__init__()
        self.folder = folder
        self.out_path = out_path
        self.fps = fps
        self.start_1 = start_1based
        self.end_1 = end_1based
        self.fmt = fmt
        self.letterbox = letterbox
        self.custom_size = custom_size
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            imgs = list_images(self.folder)
            if not imgs:
                self.aborted.emit("資料夾中沒有找到影像檔。")
                return

            total = len(imgs)
            # 1 起算 → 0 起算
            s = max(1, min(self.start_1, total)) - 1
            e = max(1, min(self.end_1, total)) - 1
            if e < s:
                self.aborted.emit("終點不可小於起點（1 起算）。")
                return

            sel = imgs[s:e+1]
            nframes = len(sel)
            self.log.emit(f"共找到 {total} 張影像，輸出範圍（1 起算）：{self.start_1} ~ {self.end_1}（實際 {nframes} 張），FPS={self.fps}")

            fourcc, _ = pick_fourcc(self.fmt)
            writer = None
            target_size = None
            done = 0

            # 若有自訂尺寸，先檢查合理性
            if self.custom_size is not None:
                cw, ch = self.custom_size
                if cw < 2 or ch < 2:
                    self.aborted.emit("自訂尺寸無效（寬高需 ≥ 2）。")
                    return

            for p in sel:
                if self._stop:
                    if writer: writer.release()
                    self.aborted.emit("已中止。")
                    return

                img = load_image(p)
                if img is None:
                    # 詳細記錄方便排錯
                    try:
                        sz = p.stat().st_size if p.exists() else -1
                    except Exception:
                        sz = -1
                    self.log.emit(f"[跳過] {p.name}：無法讀取（exists={p.exists()}, size={sz}）")
                    done += 1
                    self.progress.emit(done, nframes)
                    continue

                if writer is None:
                    # 決定輸出尺寸
                    if self.custom_size is not None:
                        target_size = self.custom_size
                        self.log.emit(f"已設定輸出尺寸（自訂）：{target_size[0]}×{target_size[1]}")
                    else:
                        h, w = img.shape[:2]
                        target_size = (w, h)
                        self.log.emit(f"已設定輸出尺寸（首張可讀影像）：{w}×{h}")

                    writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, target_size, True)
                    if not writer.isOpened():
                        self.aborted.emit("建立 VideoWriter 失敗（編碼器或輸出路徑問題）。")
                        return

                # 產生寫入影格
                if self.custom_size is None:
                    # 尺寸以首張為準：letterbox=True → 等比置中；False → 直接縮放/拉伸
                    frame = resize_to_fit(img, target_size, self.letterbox) if self.letterbox \
                            else (img if (img.shape[1], img.shape[0]) == target_size else cv2.resize(img, target_size))
                else:
                    # 自訂尺寸模式：同上
                    frame = resize_to_fit(img, target_size, self.letterbox) if self.letterbox \
                            else cv2.resize(img, target_size)

                writer.write(frame)
                done += 1
                self.progress.emit(done, nframes)

            if writer is None:
                self.aborted.emit("選擇範圍內沒有任何可讀影像。")
                return

            writer.release()
            self.log.emit("完成。")
            self.finished.emit(str(self.out_path))

        except Exception as e:
            self.aborted.emit(f"發生錯誤：{e}")

# ------------------ 主視窗 ------------------

class MainWindow(QWidget):
    def __init__(self, ui_scale: float = 1.0):
        super().__init__()
        self.ui_scale = max(1.0, ui_scale)
        self.setWindowTitle("影像序列 → 影片（PyQt5）")
        self.resize(int(840 * self.ui_scale), int(640 * self.ui_scale))

        # 控制項
        self.in_edit = QLineEdit()
        self.in_btn  = QPushButton("選擇影像資料夾…")

        self.refresh_btn = QPushButton("重新掃描")
        self.count_label = QLabel("影像數：0")
        self.size_label  = QLabel("輸出尺寸：首張可讀影像（或自訂）")

        # 起訖（1 起算）
        self.start_spin = QSpinBox()
        self.end_spin   = QSpinBox()
        self.start_spin.setMinimum(1)
        self.end_spin.setMinimum(1)

        # FPS
        self.fps_spin   = QDoubleSpinBox()
        self.fps_spin.setDecimals(2)
        self.fps_spin.setRange(0.1, 240.0)
        self.fps_spin.setValue(30.0)

        # 格式
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["MP4", "AVI"])

        # 尺寸控制
        self.custom_size_chk = QCheckBox("自訂輸出尺寸")
        self.custom_size_chk.setChecked(False)  # 預設不用自訂
        self.w_spin = QSpinBox(); self.w_spin.setRange(2, 10000); self.w_spin.setValue(1920); self.w_spin.setEnabled(False)
        self.h_spin = QSpinBox(); self.h_spin.setRange(2, 10000); self.h_spin.setValue(1080); self.h_spin.setEnabled(False)

        # letterbox 預設關閉
        self.letterbox_chk = QCheckBox("等比例加黑邊（letterbox）")
        self.letterbox_chk.setChecked(False)

        self.out_edit = QLineEdit()
        self.out_btn  = QPushButton("選擇輸出檔…")

        self.start_btn = QPushButton("開始輸出")
        self.stop_btn  = QPushButton("停止")
        self.stop_btn.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        # 版面配置
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("影像資料夾："))
        row1.addWidget(self.in_edit, 1)
        row1.addWidget(self.in_btn)
        row1.addWidget(self.refresh_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.count_label)
        row2.addSpacing(20)
        row2.addWidget(self.size_label)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("起始（1 起算）："))
        row3.addWidget(self.start_spin)
        row3.addSpacing(10)
        row3.addWidget(QLabel("終點（1 起算）："))
        row3.addWidget(self.end_spin)
        row3.addSpacing(20)
        row3.addWidget(QLabel("FPS："))
        row3.addWidget(self.fps_spin)
        row3.addSpacing(20)
        row3.addWidget(QLabel("格式："))
        row3.addWidget(self.fmt_combo)
        layout.addLayout(row3)

        row3b = QHBoxLayout()
        row3b.addWidget(self.custom_size_chk)
        row3b.addWidget(QLabel("寬："))
        row3b.addWidget(self.w_spin)
        row3b.addWidget(QLabel("高："))
        row3b.addWidget(self.h_spin)
        row3b.addSpacing(20)
        row3b.addWidget(self.letterbox_chk)
        layout.addLayout(row3b)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("輸出檔："))
        row4.addWidget(self.out_edit, 1)
        row4.addWidget(self.out_btn)
        layout.addLayout(row4)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        row5 = QHBoxLayout()
        row5.addWidget(self.start_btn)
        row5.addWidget(self.stop_btn)
        row5.addWidget(self.progress, 1)
        layout.addLayout(row5)

        layout.addWidget(self.log_view, 1)

        # 元件高度
        min_h = int(28 * self.ui_scale)
        btn_h = int(32 * self.ui_scale)
        for w in [self.in_edit, self.out_edit]:
            w.setMinimumHeight(min_h)
        for b in [self.in_btn, self.refresh_btn, self.out_btn, self.start_btn, self.stop_btn]:
            b.setMinimumHeight(btn_h)
        self.progress.setMinimumHeight(int(18 * self.ui_scale))

        # 事件
        self.in_btn.clicked.connect(self.choose_input)
        self.refresh_btn.clicked.connect(self.scan_folder)
        self.out_btn.clicked.connect(self.choose_output)
        self.start_btn.clicked.connect(self.start_encode)
        self.stop_btn.clicked.connect(self.stop_encode)
        self.fmt_combo.currentTextChanged.connect(self.auto_update_out_ext)
        self.custom_size_chk.toggled.connect(self.on_custom_size_toggled)

        # 狀態
        self.thread = None
        self.worker = None
        self.images = []

    # --------- UI handlers ---------

    def on_custom_size_toggled(self, checked: bool):
        self.w_spin.setEnabled(checked)
        self.h_spin.setEnabled(checked)
        self.size_label.setText("輸出尺寸：自訂" if checked else "輸出尺寸：首張可讀影像（或自訂）")

    def choose_input(self):
        path = QFileDialog.getExistingDirectory(self, "選擇影像資料夾")
        if path:
            self.in_edit.setText(path)
            self.scan_folder()

    def scan_folder(self):
        folder = Path(self.in_edit.text().strip())
        self.images = []
        self.count_label.setText("影像數：0")
        self.start_spin.setValue(1)
        self.end_spin.setValue(1)

        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "提示", "請選擇有效的影像資料夾。")
            return

        imgs = list_images(folder)
        self.images = imgs
        n = len(imgs)
        self.count_label.setText(f"影像數：{n}")
        self.start_spin.setRange(1, max(1, n))
        self.end_spin.setRange(1, max(1, n))
        self.end_spin.setValue(max(1, n))

        if n > 0 and not self.out_edit.text().strip():
            fmt = self.fmt_combo.currentText()
            _, ext = pick_fourcc(fmt)
            self.out_edit.setText(str(folder / f"output{ext}"))

    def auto_update_out_ext(self):
        path = self.out_edit.text().strip()
        if not path:
            return
        p = Path(path)
        fmt = self.fmt_combo.currentText()
        _, ext = pick_fourcc(fmt)
        self.out_edit.setText(str(p.with_suffix(ext)))

    def choose_output(self):
        fmt = self.fmt_combo.currentText()
        _, ext = pick_fourcc(fmt)
        path, _ = QFileDialog.getSaveFileName(self, "選擇輸出檔", filter=f"{fmt} (*{ext})")
        if path:
            p = Path(path)
            self.out_edit.setText(str(p.with_suffix(ext)))

    def start_encode(self):
        folder = Path(self.in_edit.text().strip())
        out_path = Path(self.out_edit.text().strip())
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "提示", "請選擇有效的影像資料夾。")
            return
        if not out_path.parent.exists():
            QMessageBox.warning(self, "提示", "輸出資料夾不存在。請先建立或選擇其他位置。")
            return
        if len(self.images) == 0:
            QMessageBox.warning(self, "提示", "資料夾內沒有可用影像。")
            return

        s1 = self.start_spin.value()
        e1 = self.end_spin.value()
        if e1 < s1:
            QMessageBox.warning(self, "提示", "終點不可小於起點（1 起算）。")
            return

        fps = self.fps_spin.value()
        if fps <= 0:
            QMessageBox.warning(self, "提示", "FPS 必須大於 0。")
            return

        fmt = self.fmt_combo.currentText()
        _, ext = pick_fourcc(fmt)
        out_path = out_path.with_suffix(ext)
        self.out_edit.setText(str(out_path))

        # 自訂尺寸（若有勾選）
        custom_size = None
        if self.custom_size_chk.isChecked():
            custom_size = (self.w_spin.value(), self.h_spin.value())

        # 啟動背景執行
        self.progress.setValue(0)
        self.log_view.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = EncodeWorker(
            folder=folder,
            out_path=out_path,
            fps=fps,
            start_1based=s1,
            end_1based=e1,
            fmt=fmt,
            letterbox=self.letterbox_chk.isChecked(),
            custom_size=custom_size
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.aborted.connect(self.on_aborted)
        self.worker.finished.connect(self.thread.quit)
        self.worker.aborted.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)

        self.thread.start()
        self.append_log(f"=== 來源：{folder} ===")
        self.append_log(f"=== 範圍（1 起算）：{s1} ~ {e1}，FPS={fps}，格式={fmt} ===")
        self.append_log(f"=== 尺寸模式：{'自訂 {}x{}'.format(*custom_size) if custom_size else '首張可讀影像'}；letterbox：{'是' if self.letterbox_chk.isChecked() else '否'} ===")
        self.append_log(f"=== 輸出：{out_path} ===")

    def stop_encode(self):
        if self.worker:
            self.worker.stop()
            self.append_log("嘗試中止編碼…")

    def on_progress(self, done, total):
        if total > 0:
            self.progress.setValue(int(done * 100 / total))

    def append_log(self, text: str):
        self.log_view.append(text)

    def on_finished(self, out_path: str):
        self.append_log(f"✅ 完成：{out_path}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_aborted(self, reason: str):
        self.append_log(f"⛔ {reason}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def cleanup_thread(self):
        self.worker = None
        self.thread = None


# -------- 高 DPI + UI 放大（避免累乘） --------
_APPLIED_ONCE = False

def main():
    global _APPLIED_ONCE
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    UI_SCALE = 1.30
    BASE_PT  = 10.0

    if not _APPLIED_ONCE:
        f = QFont(); f.setPointSizeF(BASE_PT * UI_SCALE); app.setFont(f)
        pad = int(6 * UI_SCALE); padx = int(12 * UI_SCALE); pbar_h = int(18 * UI_SCALE)
        app.setStyleSheet(f"""
            QLineEdit, QTextEdit, QPlainTextEdit {{ padding:{pad}px; }}
            QPushButton {{ padding:{pad}px {padx}px; }}
            QProgressBar {{ min-height:{pbar_h}px; }}
        """)
        _APPLIED_ONCE = True

    w = MainWindow(ui_scale=UI_SCALE)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
