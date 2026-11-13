# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:50:02 2026

@author: chen8

PyQt5 GUI：將資料夾中的照片做成序列影片（MP4/AVI）
整合版功能：
- 開始輸出時即時掃描資料夾（不依賴舊的 self.images）
- MP4 writer 失敗自動 fallback -> AVI(MJPG)
- thread/worker deleteLater 清理更完整
- 起始/終點為「1 起算」
- 尺寸可選：自動以首張可讀影像、或自訂寬高
- letterbox 預設關閉
- 新增：平滑形心自動裁切（依最大輪廓）
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
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def load_image(path: Path):
    # 跨平台穩定讀圖（支援中文路徑）
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
        resized = cv2.resize(
            img, (nw, nh),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        )
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        x = (tw - nw) // 2
        y = (th - nh) // 2
        canvas[y:y + nh, x:x + nw] = resized
        return canvas
    else:
        return cv2.resize(
            img, (tw, th),
            interpolation=cv2.INTER_AREA if (tw < iw or th < ih) else cv2.INTER_CUBIC
        )


def crop_by_smoothed_centroid(
    img,
    history_cx,
    history_cy,
    crop_width: int,
    crop_height: int,
    window_size: int = 10
):
    """
    依最大輪廓形心做平滑裁切
    回傳:
        cropped_img, history_cx, history_cy, debug_text
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu 二值化
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 找輪廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = img.shape[1] // 2
            cy = img.shape[0] // 2
    else:
        cx = img.shape[1] // 2
        cy = img.shape[0] // 2

    # 更新歷史
    history_cx.append(cx)
    history_cy.append(cy)

    if len(history_cx) > window_size:
        history_cx.pop(0)
        history_cy.pop(0)

    # 平滑後中心
    avg_cx = int(np.mean(history_cx))
    avg_cy = int(np.mean(history_cy))

    # 以平滑中心裁切
    x1 = max(avg_cx - crop_width // 2, 0)
    y1 = max(avg_cy - crop_height // 2, 0)
    x2 = x1 + crop_width
    y2 = y1 + crop_height

    # 超出右側/下側時往回修正
    if x2 > img.shape[1]:
        x2 = img.shape[1]
        x1 = max(x2 - crop_width, 0)

    if y2 > img.shape[0]:
        y2 = img.shape[0]
        y1 = max(y2 - crop_height, 0)

    cropped = img[y1:y2, x1:x2]

    debug_text = f"centroid=({cx},{cy}), smooth=({avg_cx},{avg_cy}), crop=({x1},{y1})-({x2},{y2})"
    return cropped, history_cx, history_cy, debug_text


def pick_fourcc(fmt: str):
    fmt = fmt.upper()
    if fmt == "MP4":
        return cv2.VideoWriter_fourcc(*'mp4v'), ".mp4"
    else:
        return cv2.VideoWriter_fourcc(*'MJPG'), ".avi"


def try_open_writer(out_path: Path, fmt: str, fps: float, target_size: Tuple[int, int]):
    """
    嘗試建立 VideoWriter。
    - fmt=MP4 若失敗：自動 fallback -> AVI(MJPG)
    回傳：(writer, actual_fmt, actual_out_path, message)
    """
    fourcc, ext = pick_fourcc(fmt)
    out_path = out_path.with_suffix(ext)

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, target_size, True)
    if writer.isOpened():
        return writer, fmt.upper(), out_path, ""

    if fmt.upper() == "MP4":
        try:
            writer.release()
        except Exception:
            pass

        fb_fmt = "AVI"
        fb_fourcc, fb_ext = pick_fourcc(fb_fmt)
        fb_out = out_path.with_suffix(fb_ext)

        fb_writer = cv2.VideoWriter(str(fb_out), fb_fourcc, fps, target_size, True)
        if fb_writer.isOpened():
            msg = "⚠️ MP4 編碼器不可用（mp4v 建立失敗），已自動改用 AVI(MJPG) 輸出（播放最穩，但檔案較大）。"
            return fb_writer, fb_fmt, fb_out, msg

    return None, fmt.upper(), out_path, "建立 VideoWriter 失敗（編碼器或輸出路徑問題）。"


# ------------------ 背景工作執行緒 ------------------

class EncodeWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    aborted = pyqtSignal(str)

    def __init__(self, folder: Path, out_path: Path, fps: float,
                 start_1based: int, end_1based: int, fmt: str,
                 letterbox: bool, custom_size: Optional[Tuple[int, int]],
                 enable_smart_crop: bool = False,
                 crop_width: int = 600,
                 crop_height: int = 800,
                 crop_window_size: int = 10):
        super().__init__()
        self.folder = folder
        self.out_path = out_path
        self.fps = fps
        self.start_1 = start_1based
        self.end_1 = end_1based
        self.fmt = fmt
        self.letterbox = letterbox
        self.custom_size = custom_size
        self.enable_smart_crop = enable_smart_crop
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_window_size = crop_window_size
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        writer = None
        try:
            imgs = list_images(self.folder)
            if not imgs:
                self.aborted.emit("資料夾中沒有找到影像檔。")
                return

            total = len(imgs)

            # 1 起算 → 0 起算（並 clamp）
            s = max(1, min(self.start_1, total)) - 1
            e = max(1, min(self.end_1, total)) - 1
            if e < s:
                self.aborted.emit("終點不可小於起點（1 起算）。")
                return

            sel = imgs[s:e + 1]
            nframes = len(sel)
            self.log.emit(
                f"共找到 {total} 張影像，輸出範圍（1 起算）：{self.start_1} ~ {self.end_1}（實際 {nframes} 張），FPS={self.fps}"
            )

            if self.custom_size is not None:
                cw, ch = self.custom_size
                if cw < 2 or ch < 2:
                    self.aborted.emit("自訂尺寸無效（寬高需 ≥ 2）。")
                    return

            target_size = None
            done = 0
            actual_fmt = self.fmt.upper()
            actual_out = self.out_path

            # 平滑形心歷史
            history_cx = []
            history_cy = []

            for p in sel:
                if self._stop:
                    if writer:
                        writer.release()
                    self.aborted.emit("已中止。")
                    return

                img = load_image(p)
                if img is None:
                    try:
                        sz = p.stat().st_size if p.exists() else -1
                    except Exception:
                        sz = -1
                    self.log.emit(f"[跳過] {p.name}：無法讀取（exists={p.exists()}, size={sz}）")
                    done += 1
                    self.progress.emit(done, nframes)
                    continue

                # 啟用智慧裁切
                if self.enable_smart_crop:
                    actual_crop_w = min(self.crop_width, img.shape[1])
                    actual_crop_h = min(self.crop_height, img.shape[0])

                    if actual_crop_w != self.crop_width or actual_crop_h != self.crop_height:
                        self.log.emit(
                            f"[警告] {p.name}：裁切尺寸 {self.crop_width}x{self.crop_height} "
                            f"大於原圖 {img.shape[1]}x{img.shape[0]}，已自動改為 {actual_crop_w}x{actual_crop_h}"
                        )

                    img, history_cx, history_cy, debug_text = crop_by_smoothed_centroid(
                        img,
                        history_cx,
                        history_cy,
                        crop_width=actual_crop_w,
                        crop_height=actual_crop_h,
                        window_size=self.crop_window_size
                    )
                    self.log.emit(f"[裁切] {p.name}: {debug_text}")

                # 初始化 writer（以首張可讀影像決定輸出尺寸）
                if writer is None:
                    if self.custom_size is not None:
                        target_size = self.custom_size
                        self.log.emit(f"已設定輸出尺寸（自訂）：{target_size[0]}×{target_size[1]}")
                    else:
                        h, w = img.shape[:2]
                        target_size = (w, h)
                        self.log.emit(f"已設定輸出尺寸（首張可讀影像）：{w}×{h}")

                    writer, actual_fmt, actual_out, msg = try_open_writer(
                        self.out_path, self.fmt, self.fps, target_size
                    )
                    if msg:
                        self.log.emit(msg)
                    if writer is None:
                        self.aborted.emit("建立 VideoWriter 失敗（MP4/AVI 皆不可用或輸出路徑問題）。")
                        return

                    self.log.emit(f"實際輸出格式：{actual_fmt}；輸出檔：{actual_out}")

                # 寫入影格
                if self.custom_size is None:
                    if self.letterbox:
                        frame = resize_to_fit(img, target_size, True)
                    else:
                        if (img.shape[1], img.shape[0]) == target_size:
                            frame = img
                        else:
                            frame = cv2.resize(img, target_size)
                else:
                    frame = resize_to_fit(img, target_size, True) if self.letterbox else cv2.resize(img, target_size)

                writer.write(frame)
                done += 1
                self.progress.emit(done, nframes)

            if writer is None:
                self.aborted.emit("選擇範圍內沒有任何可讀影像。")
                return

            writer.release()
            writer = None
            self.log.emit("完成。")
            self.finished.emit(str(actual_out))

        except Exception as e:
            try:
                if writer:
                    writer.release()
            except Exception:
                pass
            self.aborted.emit(f"發生錯誤：{e}")

        finally:
            try:
                if writer:
                    writer.release()
            except Exception:
                pass


# ------------------ 主視窗 ------------------

class MainWindow(QWidget):
    def __init__(self, ui_scale: float = 1.0):
        super().__init__()
        self.ui_scale = max(1.0, ui_scale)
        self.setWindowTitle("影像序列 → 影片（PyQt5）")
        self.resize(int(920 * self.ui_scale), int(700 * self.ui_scale))

        # 控制項
        self.in_edit = QLineEdit()
        self.in_btn = QPushButton("選擇影像資料夾…")

        self.refresh_btn = QPushButton("重新掃描")
        self.count_label = QLabel("影像數：0")
        self.size_label = QLabel("輸出尺寸：首張可讀影像（或自訂）")

        # 起訖（1 起算）
        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        self.start_spin.setMinimum(1)
        self.end_spin.setMinimum(1)

        # FPS
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setDecimals(2)
        self.fps_spin.setRange(0.1, 240.0)
        self.fps_spin.setValue(30.0)

        # 格式
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["MP4", "AVI"])

        # 尺寸控制
        self.custom_size_chk = QCheckBox("自訂輸出尺寸")
        self.custom_size_chk.setChecked(False)

        self.w_spin = QSpinBox()
        self.w_spin.setRange(2, 10000)
        self.w_spin.setValue(1920)
        self.w_spin.setEnabled(False)

        self.h_spin = QSpinBox()
        self.h_spin.setRange(2, 10000)
        self.h_spin.setValue(1080)
        self.h_spin.setEnabled(False)

        # letterbox
        self.letterbox_chk = QCheckBox("等比例加黑邊（letterbox）")
        self.letterbox_chk.setChecked(False)

        # 新增：智慧裁切
        self.smart_crop_chk = QCheckBox("啟用平滑形心裁切")
        self.smart_crop_chk.setChecked(False)

        self.crop_w_spin = QSpinBox()
        self.crop_w_spin.setRange(10, 10000)
        self.crop_w_spin.setValue(600)

        self.crop_h_spin = QSpinBox()
        self.crop_h_spin.setRange(10, 10000)
        self.crop_h_spin.setValue(800)

        self.crop_window_spin = QSpinBox()
        self.crop_window_spin.setRange(1, 100)
        self.crop_window_spin.setValue(10)

        self.out_edit = QLineEdit()
        self.out_btn = QPushButton("選擇輸出檔…")

        self.start_btn = QPushButton("開始輸出")
        self.stop_btn = QPushButton("停止")
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

        row3c = QHBoxLayout()
        row3c.addWidget(self.smart_crop_chk)
        row3c.addWidget(QLabel("裁切寬："))
        row3c.addWidget(self.crop_w_spin)
        row3c.addWidget(QLabel("裁切高："))
        row3c.addWidget(self.crop_h_spin)
        row3c.addWidget(QLabel("平滑窗口："))
        row3c.addWidget(self.crop_window_spin)
        layout.addLayout(row3c)

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
        self.thread: Optional[QThread] = None
        self.worker: Optional[EncodeWorker] = None

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
        self.count_label.setText("影像數：0")
        self.start_spin.setValue(1)
        self.end_spin.setValue(1)

        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "提示", "請選擇有效的影像資料夾。")
            return

        imgs = list_images(folder)
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

        imgs = list_images(folder)
        if len(imgs) == 0:
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

        custom_size = None
        if self.custom_size_chk.isChecked():
            custom_size = (self.w_spin.value(), self.h_spin.value())

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
            custom_size=custom_size,
            enable_smart_crop=self.smart_crop_chk.isChecked(),
            crop_width=self.crop_w_spin.value(),
            crop_height=self.crop_h_spin.value(),
            crop_window_size=self.crop_window_spin.value()
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
        self.append_log(
            f"=== 尺寸模式：{'自訂 {}x{}'.format(*custom_size) if custom_size else '首張可讀影像'}；letterbox：{'是' if self.letterbox_chk.isChecked() else '否'} ==="
        )
        self.append_log(
            f"=== 智慧裁切：{'開啟' if self.smart_crop_chk.isChecked() else '關閉'}"
            + (
                f"；裁切={self.crop_w_spin.value()}x{self.crop_h_spin.value()}；窗口={self.crop_window_spin.value()}"
                if self.smart_crop_chk.isChecked() else ""
            )
            + " ==="
        )
        self.append_log(f"=== 輸出（初始）：{out_path} ===")

    def stop_encode(self):
        if self.worker:
            self.worker.stop()
            self.append_log("嘗試中止編碼…（會在下一張影格切換時停止）")

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
        if self.worker is not None:
            try:
                self.worker.deleteLater()
            except Exception:
                pass
        if self.thread is not None:
            try:
                self.thread.deleteLater()
            except Exception:
                pass
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
    BASE_PT = 10.0

    if not _APPLIED_ONCE:
        f = QFont()
        f.setPointSizeF(BASE_PT * UI_SCALE)
        app.setFont(f)
        pad = int(6 * UI_SCALE)
        padx = int(12 * UI_SCALE)
        pbar_h = int(18 * UI_SCALE)
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