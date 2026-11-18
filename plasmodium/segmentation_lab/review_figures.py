"""Readable local review figures built only from saved experimental outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .contracts import RoiSpec
from .visualize import _font, _roi_crop, _square_rgb


def _read(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def build(run: Path, destination: Path) -> None:
    meta = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if meta["status"] != "completed":
        raise ValueError("Only completed runs can be presented")
    samples = json.loads((run / "samples.json").read_text(encoding="utf-8"))["samples"]
    destination.mkdir(parents=True, exist_ok=False)
    # Common full-height crop shows the dish despite different video positions.
    roi = RoiSpec((1280, 960), 960, 0)
    columns = [("original.png", "原始影像"), ("v2_spatial/overlay.png", "目前 V2：單張分割"),
               ("adaptive_chroma/overlay.png", "本次候選方法：排除食物、保留細網")]
    groups = {}
    for row in samples:
        groups.setdefault(row["video_id"], []).append(row)
    for video_id, rows in groups.items():
        width, top, rowheight = 540, 126, 604
        canvas = Image.new("RGB", (width * 3, top + rowheight * len(rows)), "#f4f7fb")
        draw = ImageDraw.Draw(canvas)
        source = Path(rows[0]["source_video"]).name
        draw.text((16, 12), f"{source}｜同一影格的原圖與兩種分割結果", font=_font(28, True), fill="#20364d")
        draw.text((16, 54), "綠色＝辨識邊界；紫色＝食物排除邊界。此為開發觀察，尚無人工準確率。", font=_font(22), fill="#435466")
        for col, (_, label) in enumerate(columns):
            draw.text((col * width + 12, 91), label, font=_font(23, True), fill="#20364d")
        for index, row in enumerate(rows):
            folder = run / "samples" / row["sample_id"]
            y = top + index * rowheight
            period = {"early": "早期", "middle": "中期", "late": "晚期"}[row["section"]]
            for col, (relative, _) in enumerate(columns):
                image = _read(folder / relative)
                panel = _square_rgb(_roi_crop(image, roi), width)
                canvas.paste(panel, (col * width, y + 48))
                label = f"{period}｜影格 {row['frame_index']}"
                if col:
                    method = "v2_spatial" if col == 1 else "adaptive_chroma"
                    count = int((_read(folder / method / "mask.png")[..., 0] > 0).sum())
                    label += f"｜{count:,} 像素"
                draw.text((col * width + 12, y + 10), label, font=_font(22), fill="#20364d")
        canvas.save(destination / f"{video_id}.jpg", quality=94)
    (destination / "README.txt").write_text("同一原始影格、共同裁切範圍。綠色是自動分割邊界，紫色是待覆核的食物邊界。像素數不是準確率。每張完整解析度原圖/mask/overlay及QC見上一層samples。\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.run, args.destination)
