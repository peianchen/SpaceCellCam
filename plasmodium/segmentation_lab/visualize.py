"""Local, coordinate-preserving diagnostic figures; no generated image content."""
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import RoiSpec, SegmentationResult


def write_image(path: Path, array: np.ndarray) -> None:
    """OpenCV encoding plus native file IO supports non-ASCII Windows paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, array, [cv2.IMWRITE_PNG_COMPRESSION, 3]
                             if path.suffix.lower() == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def _display_u8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    values = np.nan_to_num(array.astype(np.float32))
    if values.size and values.max() <= 1.0 and values.min() >= 0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.12,
                 excluded_food: np.ndarray | None = None) -> np.ndarray:
    """Light fill and a one-pixel internal boundary preserve real network holes."""
    result = image.copy()
    color = np.asarray((45, 255, 60), dtype=np.float32)
    result[mask] = np.rint(result[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    result[mask & ~eroded] = (45, 255, 60)
    if excluded_food is not None:
        food_boundary = excluded_food & ~cv2.erode(excluded_food.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        result[food_boundary] = (235, 50, 235)
    return result


def save_method_images(folder: Path, image: np.ndarray,
                       result: SegmentationResult) -> dict[str, str]:
    result.validate(image.shape[:2])
    folder.mkdir(parents=True, exist_ok=False)
    score = cv2.applyColorMap(np.rint(np.clip(result.score, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)
    score[~result.roi_mask] = 0
    food = image.copy()
    excluded = result.excluded_food
    food[excluded] = np.rint(0.70 * food[excluded] + 0.30 * np.asarray((235, 50, 235))).astype(np.uint8)
    boundary = excluded & ~cv2.erode(excluded.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    food[boundary] = (235, 50, 235)
    products = {
        "preprocessed_path": ("preprocessed.png", _display_u8(result.preprocessed)),
        "mask_path": ("mask.png", result.mask.astype(np.uint8) * 255),
        "score_path": ("score.png", score),
        "food_exclusion_path": ("food_exclusion.png", food),
        "food_mask_path": ("food_mask.png", excluded.astype(np.uint8) * 255),
        "overlay_path": ("overlay.png", mask_overlay(image, result.mask, excluded_food=excluded)),
    }
    for _, (filename, array) in products.items():
        write_image(folder / filename, array)
    return {key: filename for key, (filename, _) in products.items()}


def _font(size: int, bold: bool = False):
    candidates = [Path("C:/Windows/Fonts/msjhbd.ttc" if bold else "C:/Windows/Fonts/msjh.ttc"),
                  Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, current = [], ""
    for char in text:
        if char == "\n" or (current and draw.textlength(current + char, font=font) > width):
            lines.append(current)
            current = "" if char == "\n" else char
        else:
            current += char
    lines.append(current)
    return lines


def _roi_crop(image: np.ndarray, roi: RoiSpec) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy = roi.center
    # Use the full nominal dish circle for every method, including legacy rim differences.
    x0, y0 = max(0, int(math.floor(cx - roi.radius))), max(0, int(math.floor(cy - roi.radius)))
    x1, y1 = min(width, int(math.ceil(cx + roi.radius + 1))), min(height, int(math.ceil(cy + roi.radius + 1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI does not overlap the image")
    return image[y0:y1, x0:x1]


def _square_rgb(image: np.ndarray, width: int) -> Image.Image:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = Image.fromarray(cv2.cvtColor(_display_u8(image), cv2.COLOR_BGR2RGB))
    panel.thumbnail((width, width), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, width), "#101820")
    canvas.paste(panel, ((width - panel.width) // 2, (width - panel.height) // 2))
    return canvas


def save_comparison(path: Path, image: np.ndarray, roi: RoiSpec,
                    sample: dict[str, Any], method_records: list[dict[str, Any]],
                    output_root: Path, panel_width: int = 500) -> None:
    """Compose from saved products to keep only one full-size method result in RAM."""
    font, small, title_font = _font(18), _font(15), _font(23, True)
    card_texts = [("Original", "Same source frame; green = predicted boundary; magenta = food exclusion candidate.")]
    for record in method_records:
        if record["status"] == "ok":
            flags = ", ".join(record.get("qc_flags", [])) or "none (not proof of accuracy)"
            info = f"area {record['area_px']} px; food excluded {record['food_excluded_px']} px\nQC: {flags}"
            if record["method"] in {"legacy_original", "v2_spatial"}:
                info += "\nOriginal spatial algorithm at full resolution; no added food exclusion."
        else:
            info = "ERROR: " + str(record.get("error", "unknown"))
        card_texts.append((record["method"], info))
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    card_lines = [_wrap(temp_draw, info, small, panel_width - 24) for _, info in card_texts]
    info_height = 46 + max(len(lines) for lines in card_lines) * 21
    header = (f"{sample['sample_id']} | {sample.get('group', '')} / {sample.get('section', '')}\n"
              f"frame {sample['frame_index']} | source={sample['source_mode']} | DEVELOPMENT SEEN; no accuracy claim")
    source_name = Path(sample.get("source_video", "")).name
    header += f"\nSource: {source_name}; video fps is not biological sampling interval."
    header_lines = _wrap(temp_draw, header, font, panel_width * len(card_texts) - 28)
    top = 20 + len(header_lines) * 25
    canvas = Image.new("RGB", (panel_width * len(card_texts), top + info_height + panel_width + 12), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    for row, line in enumerate(header_lines):
        draw.text((14, 10 + row * 25), line, font=font, fill="#25364a")
    for index, ((title, _), lines) in enumerate(zip(card_texts, card_lines)):
        left = index * panel_width
        draw.text((left + 12, top + 6), title, font=title_font, fill="#183a56")
        for row, line in enumerate(lines):
            draw.text((left + 12, top + 40 + row * 21), line, font=small, fill="#41556a")
        if index == 0:
            panel = _square_rgb(_roi_crop(image, roi), panel_width)
        else:
            record = method_records[index - 1]
            if record["status"] == "ok":
                file = output_root / record["overlay_path"]
                overlay = cv2.imdecode(np.fromfile(str(file), np.uint8), cv2.IMREAD_COLOR)
                if overlay is None:
                    raise OSError(f"Could not read saved overlay: {file}")
                panel = _square_rgb(_roi_crop(overlay, roi), panel_width)
            else:
                panel = Image.new("RGB", (panel_width, panel_width), "#ffe8e8")
                ImageDraw.Draw(panel).text((24, 24), "METHOD FAILED", font=title_font, fill="#a20d22")
        canvas.paste(panel, (left, top + info_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=93)


def save_contact_sheets(root: Path, samples: list[dict[str, Any]], per_page: int = 4) -> list[str]:
    """At most four comparison strips/page avoids an unreadably tall image."""
    comparable = [sample for sample in samples if sample.get("comparison_path")]
    paths = []
    for start in range(0, len(comparable), per_page):
        thumbs = []
        for sample in comparable[start:start + per_page]:
            with Image.open(root / sample["comparison_path"]) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((2000, 800), Image.Resampling.LANCZOS)
            thumbs.append(thumb)
        sheet = Image.new("RGB", (max(x.width for x in thumbs), sum(x.height for x in thumbs) + 10 * (len(thumbs) - 1)), "white")
        top = 0
        for thumb in thumbs:
            sheet.paste(thumb, (0, top))
            top += thumb.height + 10
        relative = f"contact_sheets/page_{start // per_page + 1:02d}.jpg"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(root / relative, quality=92)
        paths.append(relative)
    return paths


def save_gallery(root: Path, samples: list[dict[str, Any]], run: dict[str, Any]) -> None:
    escape = lambda value: html.escape(str(value), quote=True)
    cards = []
    for sample in samples:
        methods = []
        for method, record in sample.get("methods", {}).items():
            if record["status"] == "ok":
                links = " · ".join(f'<a href="{escape(record[key])}">{label}</a>' for key, label in (
                    ("overlay_path", "全尺寸疊圖"), ("mask_path", "二值 mask"),
                    ("preprocessed_path", "預處理"), ("score_path", "特徵分數"),
                    ("food_exclusion_path", "食物排除候選"), ("qc_path", "QC 詳細記錄")))
                flags = ", ".join(record.get("qc_flags", [])) or "無旗標（不代表準確）"
                methods.append(f'<li><strong>{escape(method)}</strong>：{record["area_px"]:,} px；'
                               f'食物排除 {record["food_excluded_px"]:,} px；{record["elapsed_seconds"]:.3f} 秒。'
                               f'<br>QC：{escape(flags)}<br>{links}</li>')
            else:
                methods.append(f'<li class="error"><strong>{escape(method)}</strong>：失敗 — {escape(record.get("error", ""))}</li>')
        comparison = sample.get("comparison_path")
        figure = (f'<a href="{escape(comparison)}"><img loading="lazy" src="{escape(comparison)}" '
                  f'alt="{escape(sample["sample_id"])} 各方法比較"></a>' if comparison else "")
        original = sample.get("image_path")
        original_link = f'<a href="{escape(original)}">全尺寸原圖</a>' if original else "原圖讀取失敗"
        sample_error = f'<p class="error">{escape(sample["error"])}</p>' if sample.get("error") else ""
        cards.append(f'<article data-group="{escape(sample.get("group", ""))}" data-section="{escape(sample.get("section", ""))}">'
                     f'<h2>{escape(sample["sample_id"])}</h2><p>{escape(sample.get("group", ""))} / '
                     f'{escape(sample.get("section", ""))}；frame {escape(sample["frame_index"])}；'
                     f'{escape(sample["source_mode"])}；{original_link}</p>{sample_error}{figure}<ul>{"".join(methods)}</ul></article>')
    groups = sorted({str(sample.get("group", "")) for sample in samples})
    sections = sorted({str(sample.get("section", "")) for sample in samples})
    options = lambda values: '<option value="">全部</option>' + ''.join(f'<option value="{escape(v)}">{escape(v)}</option>' for v in values)
    pages = " · ".join(f'<a href="{escape(path)}">比較總覽 {i + 1}</a>' for i, path in enumerate(run.get("contact_sheets", [])))
    document = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>黏菌分割比較 — {escape(run['run_id'])}</title><style>
body{{font:16px/1.6 system-ui,"Microsoft JhengHei",sans-serif;margin:0;background:#eef3f7;color:#182c3d}}main{{max-width:1500px;margin:auto;padding:24px}}
h1{{font-size:28px}}h2{{font-size:20px;overflow-wrap:anywhere}}article{{background:white;border:1px solid #cbd6e1;border-radius:10px;padding:20px;margin:22px 0}}
img{{width:100%;height:auto}}a{{color:#095da4}}li{{margin:10px 0}}.notice{{padding:16px;background:#fff5d8;border-left:5px solid #c18b08}}
.error{{color:#a01528}}select{{font:inherit;padding:6px;margin-right:20px}}[hidden]{{display:none}}code{{overflow-wrap:anywhere}}</style><main>
<h1>同一影格的 mask 方法比較</h1><p class="notice">全部樣本標記為 <strong>development_seen</strong>，已用於開發觀察，不能當成未看過的測試集。
面積、時間與 QC 旗標是診斷資訊，沒有人工真值時不提供準確率。影片 fps 是播放幀率，不能直接當成生物拍攝間隔。
特徵分數圖是顏色／形態證據，不是經校準的機率。綠色是預測邊界與輕填色，網孔保留；洋紅色食物排除邊界只是演算法候選，需人工確認。
legacy_original / v2_spatial 是原解析度 spatial 比較，未加時間補償或新食物排除；舊方法保留原皿圈範圍，新方法可能使用邊緣內縮，面積差不能解讀成準確度。</p>
<p>執行狀態：<strong>{escape(run.get('status'))}</strong>；樣本 {len(samples)} 張；錯誤 {run.get('error_count', 0)} 筆。</p>
<p><a href="run_manifest.json">執行來源與參數</a> · <a href="metrics.csv">逐圖診斷 CSV</a> · <a href="samples.json">樣本紀錄</a> · {pages}</p>
<label>組別 <select id="group">{options(groups)}</select></label><label>階段 <select id="section">{options(sections)}</select></label>
{''.join(cards)}</main><script>const group=document.getElementById('group'),section=document.getElementById('section');
function filter(){{document.querySelectorAll('article').forEach(a=>a.hidden=(group.value&&a.dataset.group!==group.value)||(section.value&&a.dataset.section!==section.value));}}
group.addEventListener('change',filter);section.addEventListener('change',filter);</script></html>'''
    (root / "gallery.html").write_text(document, encoding="utf-8")
