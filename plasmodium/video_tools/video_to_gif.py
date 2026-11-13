# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:00:45 2025

@author: chen8
"""

import os
import sys
import subprocess
from pathlib import Path

# ================== 可調參數 ==================
FPS = 30               # 每秒影格數 (建議 10~20)
SCALE = None            # GIF寬度像素（維持比例），例如 480，若不縮放設 None
START_TIME = None      # 例如 "00:00:05" 起始時間；不剪輯設 None
DURATION = None        # 例如 3 → 只取 3 秒長度；不剪輯設 None
PALETTE_SIZE = 256     # 顏色數 (max 256)
SUFFIX = "_gif"        # 輸出檔名後綴
OVERWRITE = True        # True 覆蓋既有檔案
VERBOSE = False         # 顯示 ffmpeg log
# ==============================================

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise SystemExit("找不到 ffmpeg，請先安裝並加入 PATH。")

def build_filters():
    parts = []
    if SCALE:
        parts.append(f"scale={SCALE}:-1:flags=lanczos")
    parts.append(f"fps={FPS}")
    return ",".join(parts)

def video_to_gif(input_path: Path):
    input_path = input_path.resolve()
    out_path = input_path.with_name(f"{input_path.stem}{SUFFIX}.gif")

    if out_path.exists() and not OVERWRITE:
        print(f"[跳過] 已存在：{out_path}")
        return True

    filter_complex = build_filters()

    # palette 暫存檔
    palette_path = input_path.with_name(f"{input_path.stem}_palette.png")

    # Step 1: 產生調色盤
    cmd_palette = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-hide_banner",
        "-i", str(input_path),
    ]
    if START_TIME:
        cmd_palette += ["-ss", START_TIME]
    if DURATION:
        cmd_palette += ["-t", str(DURATION)]
    cmd_palette += [
        "-vf", f"{filter_complex},palettegen=max_colors={PALETTE_SIZE}",
        str(palette_path)
    ]

    # Step 2: 使用調色盤生成 GIF
    cmd_gif = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-hide_banner",
        "-i", str(input_path),
        "-i", str(palette_path),
    ]
    if START_TIME:
        cmd_gif += ["-ss", START_TIME]
    if DURATION:
        cmd_gif += ["-t", str(DURATION)]
    cmd_gif += [
        "-filter_complex", f"{filter_complex}[x];[x][1:v]paletteuse=dither=sierra2_4a",
        str(out_path)
    ]

    log = None if VERBOSE else subprocess.DEVNULL
    print(f"🎞️ 生成調色盤: {' '.join(cmd_palette)}")
    subprocess.run(cmd_palette, stdout=log, stderr=log)

    print(f"🌀 轉換為GIF: {' '.join(cmd_gif)}")
    result = subprocess.run(cmd_gif, stdout=log, stderr=log)

    if result.returncode == 0:
        print(f"[完成] {out_path}")
        if palette_path.exists():
            palette_path.unlink()  # 刪暫存
        return True
    else:
        print(f"[失敗] {input_path}")
        return False

# ---------- 檔案選擇：優先用 Qt（Spyder 最穩） ----------
def pick_files():
    # PyQt6
    try:
        from PyQt6.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication(sys.argv)
        files, _ = QFileDialog.getOpenFileNames(
            None, "選擇要轉成GIF的影片",
            "", "影片檔案 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;所有檔案 (*)"
        )
        if not QApplication.instance():
            app.quit()
        return [Path(f) for f in files]
    except Exception:
        pass
    # PyQt5
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication(sys.argv)
        files, _ = QFileDialog.getOpenFileNames(
            None, "選擇要轉成GIF的影片",
            "", "影片檔案 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;所有檔案 (*)"
        )
        if not QApplication.instance():
            app.quit()
        return [Path(f) for f in files]
    except Exception:
        pass
    # fallback CLI
    print("（未找到 Qt 檔案選擇器；改用命令列）")
    line = input("請輸入影片路徑（可多個，以 | 分隔）：\n> ").strip()
    if not line:
        return []
    return [Path(p.strip()) for p in line.split("|") if p.strip()]

def main():
    ensure_ffmpeg()
    paths = pick_files()
    if not paths:
        print("未選擇任何檔案。")
        return
    for p in paths:
        video_to_gif(p)
    print("\n== GIF 轉換完成 ==")

if __name__ == "__main__":
    main()
