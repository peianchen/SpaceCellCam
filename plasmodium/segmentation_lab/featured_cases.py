"""Two source-traceable examples: visible improvement and remaining failure."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from .contracts import RoiSpec
from .review_figures import _read
from .visualize import _font, _roi_crop, _square_rgb


def build(run, output):
    rows = json.loads((run / "samples.json").read_text(encoding="utf-8"))["samples"]
    indexed = {row["sample_id"]: row for row in rows}
    cases = [
        ("still_00_31bd0a24_late_f006627", "案例 A：淡色細網補回較多，中央食物另外排除", (1307, 984), 660),
        ("rotating_02_b3ec4905_late_f006390", "案例 B：暗光下仍漏掉部分左下方細網，尚不能當正式面積", (1315, 989), 790),
    ]
    width, rowheight = 600, 716
    canvas = Image.new("RGB", (width*3, rowheight*2+74), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), "第一輪實際結果｜綠色：預測黏菌邊界；紫色：食物排除邊界", font=_font(29, True), fill="#20364d")
    draw.text((18, 47), "傳統多特徵影像處理；沒有訓練 AI。以下為目視觀察，尚無人工真值準確率。", font=_font(22), fill="#435466")
    sources = []
    for index, (sid, title, center, radius) in enumerate(cases):
        row = indexed[sid]
        folder = run / "samples" / sid
        y = 74+index*rowheight
        draw.text((18, y+8), title, font=_font(28, True), fill="#20364d" if index == 0 else "#9b4c17")
        draw.text((18, y+45), f"{Path(row['source_video']).name}｜影格 {row['frame_index']}", font=_font(22), fill="#435466")
        for col, (rel, label) in enumerate([
            ("original.png", "原始影像"), ("v2_spatial/overlay.png", "目前 V2：單張結果"),
            ("adaptive_chroma/overlay.png", "本次候選方法：仍需人工覆核"),
        ]):
            draw.text((col*width+12, y+77), label, font=_font(25, True), fill="#20364d")
            panel = _square_rgb(_roi_crop(_read(folder/rel), RoiSpec(center, radius, 0)), width)
            canvas.paste(panel, (col*width, y+112))
        sources.append({"sample_id": sid, "shared_crop_center": center, "shared_crop_radius": radius,
                        "interpretation": title, "source_pixel_sha256": row["source_pixel_sha256"]})
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)
    output.with_suffix(".json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.run, args.output)
