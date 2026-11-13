import os
import sys
import subprocess
from pathlib import Path

# ================== 可調參數 ==================
CRF = 23               # 品質(18~28 常用；越小越好、檔案越大)
PRESET = "medium"      # ultrafast, veryfast, medium, slow, slower
AUDIO = "128k"         # 'copy' 或 96k/128k/192k ...
CODEC = "libx264"      # libx264 / libx265 / h264_nvenc / hevc_nvenc / h264_videotoolbox / hevc_videotoolbox
RESIZE_LONGER = None   # 例如 1280；限制最長邊，維持比例；不縮放設 None
OVERWRITE = True       # True 覆蓋既有輸出
VERBOSE = False        # 顯示 ffmpeg log
SUFFIX = "_compressed" # 輸出檔名後綴
# ==============================================

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise SystemExit("找不到 ffmpeg/ffprobe，請先安裝並加入 PATH。")

def build_scale_filter():
    if RESIZE_LONGER:
        return ["-vf", f"scale='if(gt(iw,ih),{RESIZE_LONGER},-2)':'if(gt(iw,ih),-2,{RESIZE_LONGER})'"]
    return []

def compress_one(input_path: Path):
    input_path = input_path.resolve()
    out_path = input_path.with_name(f"{input_path.stem}{SUFFIX}.mp4")

    if out_path.exists() and not OVERWRITE:
        print(f"[跳過] 已存在：{out_path}")
        return True

    cmd = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-hide_banner",
        "-i", str(input_path),
        *build_scale_filter(),
        "-c:v", CODEC,
        "-preset", PRESET,
        "-crf", str(CRF),
        "-c:a", "aac" if AUDIO != "copy" else "copy",
    ]
    if AUDIO != "copy":
        cmd += ["-b:a", AUDIO]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]

    print(">>", " ".join(cmd))
    log = None if VERBOSE else subprocess.DEVNULL
    proc = subprocess.run(cmd, stdout=log, stderr=log)
    if proc.returncode == 0:
        print(f"[完成] {out_path}")
        return True
    else:
        print(f"[失敗] {input_path}")
        return False

# ---------- 檔案選擇：優先用 Qt（Spyder 最穩），再退回 CLI ----------
def pick_files():
    # 1) 試 PyQt6
    try:
        from PyQt6.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication(sys.argv)
        files, _ = QFileDialog.getOpenFileNames(
            None, "選擇要壓縮的影片（可多選）",
            "", "影片檔案 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;所有檔案 (*)"
        )
        if not QApplication.instance():
            app.quit()
        return [Path(f) for f in files]
    except Exception:
        pass

    # 2) 試 PyQt5
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication(sys.argv)
        files, _ = QFileDialog.getOpenFileNames(
            None, "選擇要壓縮的影片（可多選）",
            "", "影片檔案 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;所有檔案 (*)"
        )
        if not QApplication.instance():
            app.quit()
        return [Path(f) for f in files]
    except Exception:
        pass

    # 3) 試 PySide6
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication(sys.argv)
        files, _ = QFileDialog.getOpenFileNames(
            None, "選擇要壓縮的影片（可多選）",
            "", "影片檔案 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;所有檔案 (*)"
        )
        if not QApplication.instance():
            app.quit()
        return [Path(f) for f in files]
    except Exception:
        pass

    # 4) 全部失敗：退回命令列輸入
    print("（未找到 Qt 檔案選擇器；改用命令列）")
    line = input("請輸入影片路徑（可多個，以 | 分隔）：\n> ").strip()
    if not line:
        return []
    return [Path(p.strip()) for p in line.split("|") if p.strip()]

def is_video(p: Path):
    return p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

def main():
    ensure_ffmpeg()
    paths = pick_files()
    if not paths:
        print("未選擇任何檔案。")
        return

    ok = 0
    total = 0
    for p in paths:
        if p.is_file() and is_video(p):
            total += 1
            ok += 1 if compress_one(p) else 0
        else:
            print(f"[略過] 非影片或找不到檔案：{p}")

    print(f"\n== 結果：成功 {ok} / {total} ==")

if __name__ == "__main__":
    main()
