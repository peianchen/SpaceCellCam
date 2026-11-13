# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:14:46 2025

@author: chen8
"""

# -*- coding: utf-8 -*-
"""
Video TimeScaler GUI (PyQt5 + OpenCV)
- 輸入影片，依速度表分段加速/減速後輸出為 MP4/AVI（無音訊）
- 速度表語法: 'start-end@speed, ...'，start/end 為「1 起算」的來源幀序號
- 例: '1-300@0.5, 301-900@2, 901-1200@1'
"""

import sys
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QCheckBox, QProgressBar, QTextEdit, QMessageBox,
    QFrame, QDoubleSpinBox, QComboBox, QSpinBox
)

# ------------------ 工具函式 ------------------

def pick_fourcc(fmt: str):
    if fmt.upper() == "MP4":
        return cv2.VideoWriter_fourcc(*'mp4v'), ".mp4"
    else:
        return cv2.VideoWriter_fourcc(*'XVID'), ".avi"

def parse_speed_spec(spec_text: str, total_frames: Optional[int]) -> List[Tuple[int, int, float]]:
    """
    解析速度表:
      '1-300@0.5, 301-900@2, 901-1200@1'
    回傳 [(start0,end0,speed0), ...]，皆為 0 起算索引（閉區間）
    若 spec 為空，回傳整段 1.0x。若 total_frames 已知，用它做上界裁切。
    """
    spec_text = (spec_text or "").strip()
    segs: List[Tuple[int,int,float]] = []
    if not spec_text:
        if total_frames is None:
            raise ValueError("無速度表且未知總幀數，請先填速度表或載入影片。")
        return [(0, total_frames - 1, 1.0)]

    for part in spec_text.split(","):
        part = part.strip()
        m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*@\s*([0-9]*\.?[0-9]+)\s*$', part)
        if not m:
            raise ValueError(f"速度表格式錯誤：{part}")
        a1 = int(m.group(1))
        b1 = int(m.group(2))
        sp = float(m.group(3))
        if b1 < a1:
            raise ValueError(f"區間終點小於起點：{part}")
        if sp <= 0:
            raise ValueError(f"速度必須 > 0：{part}")
        a0 = a1 - 1
        b0 = b1 - 1
        if total_frames is not None:
            a0 = max(0, min(a0, total_frames - 1))
            b0 = max(0, min(b0, total_frames - 1))
        if b0 >= a0:
            segs.append((a0, b0, sp))

    if not segs:
        raise ValueError("速度表解析結果為空，請檢查範圍是否合理。")

    segs.sort(key=lambda x: x[0])
    # 檢查重疊
    for i in range(1, len(segs)):
        if segs[i][0] <= segs[i-1][1]:
            raise ValueError("速度區間彼此重疊，請調整。")
    return segs

def speed_for_index(idx0: int, segs: List[Tuple[int,int,float]]) -> float:
    # 假設 segs 已依起點排序且無重疊
    # 二分搜尋也可，這裡線性足夠（簡潔）
    for a,b,s in segs:
        if a <= idx0 <= b:
            return s
        if idx0 < a:
            break
    return 1.0

# ------------------ 背景工作執行緒 ------------------

class ScaleWorker(QObject):
    progress = pyqtSignal(int, int)   # done, total_in_frames (若未知則 total可能<=0)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    aborted = pyqtSignal(str)

    def __init__(self, in_path: Path, out_path: Path, out_fps: float,
                 fmt: str, speed_spec: str, keep_src_fps: bool):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.out_fps = out_fps
        self.fmt = fmt
        self.speed_spec = speed_spec
        self.keep_src_fps = keep_src_fps
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cap = cv2.VideoCapture(str(self.in_path))
            if not cap.isOpened():
                self.aborted.emit("無法開啟輸入影片。")
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fin = cap.get(cv2.CAP_PROP_FPS)
            if not fin or fin <= 0:
                # 某些檔案取不到 fps，給個預設
                fin = 30.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log.emit(f"來源資訊：{width}x{height}, FPS={fin:.3f}, 總幀={total if total>0 else '未知'}")

            # 解析速度表
            try:
                segs = parse_speed_spec(self.speed_spec, total if total>0 else None)
            except Exception as ex:
                self.aborted.emit(f"速度表錯誤：{ex}")
                cap.release()
                return
            self.log.emit(f"速度段數：{len(segs)} ；{self.speed_spec or '（全程 1.0x）'}")

            # 決定輸出 FPS
            fps_out = fin if self.keep_src_fps else self.out_fps
            if fps_out <= 0:
                fps_out = fin
            self.log.emit(f"輸出 FPS：{fps_out:.3f}")

            fourcc, _ = pick_fourcc(self.fmt)
            writer = cv2.VideoWriter(str(self.out_path), fourcc, fps_out, (width, height), True)
            if not writer.isOpened():
                cap.release()
                self.aborted.emit("建立 VideoWriter 失敗（編碼器或輸出路徑問題）。")
                return

            # 以時間軸法重取樣：
            # 每個來源幀的「等效輸出時間寬度」： dt' = (1/fin) / speed
            # 當累積的 T' 超過 next_emit_time (= k*(1/fps_out)) 就 emit 目前幀（可能重複多次或一次都不 emit）
            dt_out = 1.0 / fps_out
            Tprime = 0.0
            next_emit_time = 0.0

            src_idx = 0
            done_in = 0
            ok, frame = cap.read()
            last_good = None

            while ok:
                if self._stop:
                    writer.release(); cap.release()
                    self.aborted.emit("已中止。")
                    return

                last_good = frame
                sp = speed_for_index(src_idx, segs)
                dt_prime = (1.0 / fin) / sp

                # emit 0~n 次當前幀（慢動作會重複；加速可能一次都不 emit）
                # 注意：這樣會讓輸出時間對齊常數 FPS 的節拍點
                while Tprime + dt_prime >= next_emit_time:
                    writer.write(frame)
                    next_emit_time += dt_out

                Tprime += dt_prime
                src_idx += 1
                done_in += 1
                if total > 0:
                    self.progress.emit(done_in, total)
                else:
                    self.progress.emit(done_in, 0)

                ok, frame = cap.read()

            # 來源讀完後，如果還差一點點到下個節拍，不再補幀（保持收尾乾淨）
            writer.release()
            cap.release()

            if last_good is None:
                self.aborted.emit("來源幀讀取皆失敗。")
                return

            self.log.emit("完成。")
            self.finished.emit(str(self.out_path))

        except Exception as e:
            self.aborted.emit(f"發生錯誤：{e}")

# ------------------ 主視窗 ------------------

class MainWindow(QWidget):
    def __init__(self, ui_scale: float = 1.0):
        super().__init__()
        self.ui_scale = max(1.0, ui_scale)
        self.setWindowTitle("影片變速輸出（PyQt5）")
        self.resize(int(880 * self.ui_scale), int(620 * self.ui_scale))

        # 路徑
        self.in_edit = QLineEdit()
        self.in_btn  = QPushButton("選擇輸入影片…")
        self.out_edit = QLineEdit()
        self.out_btn  = QPushButton("選擇輸出檔…")

        # FPS 與格式
        self.keep_src_fps_chk = QCheckBox("沿用來源 FPS")
        self.keep_src_fps_chk.setChecked(True)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setDecimals(3)
        self.fps_spin.setRange(0.1, 240.0)
        self.fps_spin.setValue(59.94)
        self.fps_spin.setEnabled(False)

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["MP4", "AVI"])

        # 速度表
        self.speed_edit = QLineEdit()
        self.speed_edit.setPlaceholderText("速度表：start-end@倍速，用逗號分隔。例：1-300@0.5, 301-900@2")

        # 控制
        self.start_btn = QPushButton("開始輸出")
        self.stop_btn  = QPushButton("停止")
        self.stop_btn.setEnabled(False)

        # 進度與日誌
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        # 版面配置
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("輸入影片："))
        row1.addWidget(self.in_edit, 1)
        row1.addWidget(self.in_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("輸出檔："))
        row2.addWidget(self.out_edit, 1)
        row2.addWidget(self.out_btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(self.keep_src_fps_chk)
        row3.addSpacing(10)
        row3.addWidget(QLabel("輸出 FPS："))
        row3.addWidget(self.fps_spin)
        row3.addSpacing(20)
        row3.addWidget(QLabel("格式："))
        row3.addWidget(self.fmt_combo)
        layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("速度表："))
        row4.addWidget(self.speed_edit, 1)
        layout.addLayout(row4)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        row5 = QHBoxLayout()
        row5.addWidget(self.start_btn)
        row5.addWidget(self.stop_btn)
        row5.addWidget(self.progress, 1)
        layout.addLayout(row5)

        layout.addWidget(self.log_view, 1)

        # UI 細節
        min_h = int(28 * self.ui_scale)
        btn_h = int(32 * self.ui_scale)
        for w in [self.in_edit, self.out_edit]:
            w.setMinimumHeight(min_h)
        for b in [self.in_btn, self.out_btn, self.start_btn, self.stop_btn]:
            b.setMinimumHeight(btn_h)
        self.progress.setMinimumHeight(int(18 * self.ui_scale))

        # 事件
        self.in_btn.clicked.connect(self.choose_input)
        self.out_btn.clicked.connect(self.choose_output)
        self.start_btn.clicked.connect(self.start_encode)
        self.stop_btn.clicked.connect(self.stop_encode)
        self.fmt_combo.currentTextChanged.connect(self.auto_update_out_ext)
        self.keep_src_fps_chk.toggled.connect(self.on_keep_src_fps_toggled)

        # 狀態
        self.thread: Optional[QThread] = None
        self.worker: Optional[ScaleWorker] = None

    # --------- UI handlers ---------

    def on_keep_src_fps_toggled(self, checked: bool):
        self.fps_spin.setEnabled(not checked)

    def choose_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇輸入影片", filter="Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All Files (*.*)")
        if path:
            self.in_edit.setText(path)
            # 若尚未指定輸出，預設用同資料夾同名
            if not self.out_edit.text().strip():
                fmt = self.fmt_combo.currentText()
                _, ext = pick_fourcc(fmt)
                p = Path(path)
                self.out_edit.setText(str(p.with_name(p.stem + "_scaled").with_suffix(ext)))

    def choose_output(self):
        fmt = self.fmt_combo.currentText()
        _, ext = pick_fourcc(fmt)
        path, _ = QFileDialog.getSaveFileName(self, "選擇輸出檔", filter=f"{fmt} (*{ext})")
        if path:
            p = Path(path)
            self.out_edit.setText(str(p.with_suffix(ext)))

    def auto_update_out_ext(self):
        path = self.out_edit.text().strip()
        if not path:
            return
        p = Path(path)
        fmt = self.fmt_combo.currentText()
        _, ext = pick_fourcc(fmt)
        self.out_edit.setText(str(p.with_suffix(ext)))

    def start_encode(self):
        in_path = Path(self.in_edit.text().strip())
        out_path = Path(self.out_edit.text().strip())
        if not in_path.exists():
            QMessageBox.warning(self, "提示", "請選擇有效的輸入影片。")
            return
        if not out_path.parent.exists():
            QMessageBox.warning(self, "提示", "輸出資料夾不存在。請先建立或選擇其他位置。")
            return

        fmt = self.fmt_combo.currentText()
        _, ext = pick_fourcc(fmt)
        out_path = out_path.with_suffix(ext)
        self.out_edit.setText(str(out_path))

        fps_out = self.fps_spin.value()
        keep_src = self.keep_src_fps_chk.isChecked()

        # 啟動背景執行
        self.progress.setValue(0)
        self.log_view.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = ScaleWorker(
            in_path=in_path,
            out_path=out_path,
            out_fps=fps_out,
            fmt=fmt,
            speed_spec=self.speed_edit.text(),
            keep_src_fps=keep_src
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
        self.append_log(f"=== 輸入：{in_path} ===")
        self.append_log(f"=== 輸出：{out_path}；格式={fmt}；FPS={'沿用來源' if keep_src else self.fps_spin.value()} ===")
        self.append_log(f"=== 速度表：{self.speed_edit.text().strip() or '（全程 1.0x）'} ===")

    def stop_encode(self):
        if self.worker:
            self.worker.stop()
            self.append_log("嘗試中止…")

    def on_progress(self, done, total):
        if total and total > 0:
            self.progress.setValue(int(done * 100 / total))
        else:
            # 總幀未知時，就讓進度條循序滾動
            self.progress.setMaximum(0)

    def append_log(self, text: str):
        self.log_view.append(text)

    def on_finished(self, out_path: str):
        self.progress.setMaximum(100)
        self.progress.setValue(100)
        self.append_log(f"✅ 完成：{out_path}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_aborted(self, reason: str):
        self.progress.setMaximum(100)
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

    UI_SCALE = 1.20
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
