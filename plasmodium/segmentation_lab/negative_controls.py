"""Visually selected real-background controls; these are not verified ground truth.

The four fixed crops were inspected in the raw source frames from raw24.
Analysis uses original pixels without resizing. Crop-only ROI settings differ
from full-frame analysis, so results describe these controls, not whole-frame
accuracy or a validated false-positive rate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .benchmark import _environment, _json, file_sha256, pixel_sha256
from .contracts import RoiSpec
from .visualize import _font, save_method_images, write_image


# xyxy: source coordinates, right/bottom exclusive. Never scale these coordinates.
CONTROL_CROPS = (
    {"sample_id": "still_00_31bd0a24_early_f001406", "xyxy": [1180, 1180, 1404, 1404],
     "visual_note": "Interior gray floor below the food/tissue; no visible yellow network or dish rim in crop."},
    {"sample_id": "still_01_bbb345f7_early_f001478", "xyxy": [850, 850, 1074, 1074],
     "visual_note": "Interior gray floor left of the central food/tissue; outside bright dish rim."},
    {"sample_id": "rotating_00_98e8006a_early_f001339", "xyxy": [1110, 1220, 1334, 1412],
     "visual_note": "Interior floor below the lowest yellow tissue, above the inner circular floor mark."},
    {"sample_id": "rotating_01_799f65b5_early_f001411", "xyxy": [1150, 1200, 1374, 1424],
     "visual_note": "Interior gray floor below the central food/tissue; away from the lower bright rim."},
)
METHODS = ("adaptive_chroma", "local_lab")
LIMITATION = (
    "visually_selected_background_control: patches were selected by visual inspection, "
    "not independently annotated pixelwise ground truth. Foreground counts are apparent "
    "background detections, not verified biological false-positive rates or overall accuracy. "
    "No tissue/food/rim is visually apparent in selected patches, but faint biological "
    "material cannot be excluded by this inspection. Crop-only background estimation "
    "and ROI differ from a full-frame analysis; these results do not replace full-frame validation."
)


def _source_image(run: Path, row: dict) -> tuple[np.ndarray, Path]:
    if row.get("source_mode") != "raw":
        raise ValueError(f"Control must use raw source pixels: {row['sample_id']}")
    path = (run / row["image_path"]).resolve()
    if not path.is_relative_to(run):
        raise ValueError("Original image must be inside the supplied benchmark run")
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot decode source image: {path}")
    expected = row.get("source_pixel_sha256")
    if not expected or pixel_sha256(image) != expected:
        raise ValueError(f"Source pixel hash missing or mismatched: {row['sample_id']}")
    return image, path


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: file_sha256(root / name) for name in
            ("negative_controls.py", "segmenter.py", "contracts.py", "benchmark.py", "visualize.py")}


def _context_image(image: np.ndarray, xyxy: list[int]) -> np.ndarray:
    context = image.copy()
    x0, y0, x1, y1 = xyxy
    cv2.rectangle(context, (x0, y0), (x1 - 1, y1 - 1), (30, 210, 255), 5)
    cv2.putText(context, "BACKGROUND CONTROL CROP", (max(5, x0 - 15), max(30, y0 - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 210, 255), 2, cv2.LINE_AA)
    # Only this location overview is resized, never the crop passed to segmentation.
    return cv2.resize(context, (960, round(image.shape[0] * 960 / image.shape[1])), interpolation=cv2.INTER_AREA)


def _comparison(out: Path, record: dict, crop: np.ndarray) -> None:
    width, height = 430, max(224, crop.shape[0]) + 170
    methods = record.get("methods", {})
    cards = [("Original crop", "native pixels; visually selected", crop)]
    for method in METHODS:
        info = methods.get(method)
        if not info:
            continue
        if info["status"] == "ok":
            overlay = cv2.imdecode(np.fromfile(str(out / info["overlay_path"]), np.uint8), cv2.IMREAD_COLOR)
            message = f"foreground {info['false_positive_pixels']} px / ROI {info['roi_area_px']} px"
            message += f"\nfraction {info['false_positive_roi_fraction']:.6f} (not GT accuracy)"
        else:
            overlay = np.full_like(crop, (205, 205, 255))
            message = "METHOD ERROR; inspect JSON"
        cards.append((method, message, overlay))
    canvas = Image.new("RGB", (width * len(cards), height), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    title, body = _font(19, True), _font(15)
    for index, (name, message, panel) in enumerate(cards):
        left = index * width
        draw.text((left + 14, 12), name, font=title, fill="#173952")
        draw.multiline_text((left + 14, 43), message, font=body, fill="#344c61", spacing=6)
        draw.text((left + 14, 98), f"crop xyxy={record['crop_xyxy']}", font=body, fill="#344c61")
        rgb = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        canvas.paste(rgb, (left + (width - rgb.width) // 2, 140))
    canvas.save(out / record["comparison_path"], quality=94)


def run_controls(args: argparse.Namespace) -> int:
    run, out = args.run.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Control output must be a new directory: {out}")
    samples_path = run / "samples.json"
    data = json.loads(samples_path.read_text(encoding="utf-8"))
    if data.get("source_mode") != "raw":
        raise ValueError("--run must refer to a raw-frame benchmark run")
    rows = {row["sample_id"]: row for row in data["samples"]}
    missing = [spec["sample_id"] for spec in CONTROL_CROPS if spec["sample_id"] not in rows]
    if missing:
        raise ValueError(f"The reviewed source sample IDs are missing: {missing}")
    if len({rows[spec["sample_id"]]["video_id"] for spec in CONTROL_CROPS}) != 4:
        raise ValueError("The four controls must come from four distinct source videos")
    supplied = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    if not isinstance(supplied, dict):
        raise ValueError("--config must contain a JSON object")
    out.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "control_type": "visually_selected_background_control",
               "source_run": str(run), "source_samples_json_sha256": file_sha256(samples_path),
               "created_utc": datetime.now(timezone.utc).isoformat(), "preview_only": args.preview_only,
               "ground_truth_available": False, "evaluation_role": "development_seen",
               "limitation": LIMITATION, "analysis_resizing": False,
               "crop_coordinate_convention": "xyxy source-pixel coordinates; right and bottom exclusive",
               "roi_policy": "Circle at crop center, radius=0.45*min(width,height), margin=0; no ROI refinement.",
               "config_requested": supplied, "code_sha256": _code_hashes(), "environment": _environment(),
               "controls": [], "error_count": 0, "status": "running"}
    _json(out / "negative_controls.json", summary)
    if not args.preview_only:
        from .segmenter import segment
    csv_rows = []
    for index, spec in enumerate(CONTROL_CROPS, 1):
        row = rows[spec["sample_id"]]
        folder = out / f"control_{index:02d}"
        folder.mkdir()
        record = {"control_type": summary["control_type"], "source_sample_id": row["sample_id"],
                  "source_video_id": row["video_id"], "source_video": row.get("source_video"),
                  "source_frame_index": row["frame_index"], "source_group": row.get("group"),
                  "source_mode": "raw", "crop_xyxy": spec["xyxy"], "visual_selection_note": spec["visual_note"],
                  "ground_truth_available": False, "analysis_resizing": False,
                  "methods": {}, "status": "running"}
        summary["controls"].append(record)
        try:
            image, path = _source_image(run, row)
            x0, y0, x1, y1 = spec["xyxy"]
            if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                raise ValueError("Fixed crop is outside the source image")
            crop = image[y0:y1, x0:x1].copy()
            record.update(source_image_path=str(path), source_file_sha256=file_sha256(path),
                          source_pixel_sha256=pixel_sha256(image), crop_pixel_sha256=pixel_sha256(crop),
                          crop_width=crop.shape[1], crop_height=crop.shape[0],
                          crop_path=f"control_{index:02d}/crop.png", context_path=f"control_{index:02d}/context.jpg",
                          comparison_path=f"control_{index:02d}/comparison.jpg")
            write_image(out / record["crop_path"], crop)
            write_image(out / record["context_path"], _context_image(image, spec["xyxy"]))
            del image
            roi = RoiSpec(((crop.shape[1] - 1) / 2, (crop.shape[0] - 1) / 2), min(crop.shape[:2]) * .45, 0.)
            record["roi"] = {"center": roi.center, "radius": roi.radius, "margin_px": roi.margin_px}
            if args.preview_only:
                record["status"] = "preview_only_no_segmentation"
            else:
                for method in METHODS:
                    started = time.perf_counter()
                    params = dict(supplied.get("methods", {}).get(method, {})) if "methods" in supplied else dict(supplied)
                    params.update(refine_roi=False, food_exclusion=False)
                    info = {"method": method, "parameters_overrides": params}
                    record["methods"][method] = info
                    try:
                        result = segment(crop.copy(), roi, method=method, config=params)
                        info["elapsed_seconds"] = time.perf_counter() - started
                        result.validate(crop.shape[:2])
                        saved = save_method_images(folder / method, crop, result)
                        area = int(result.mask.sum())
                        roi_area = int(result.roi_mask.sum())
                        info.update(status="ok", false_positive_pixels=area, predicted_foreground_px=area,
                                    roi_area_px=roi_area, false_positive_roi_fraction=area / max(1, roi_area),
                                    metric_interpretation="Apparent foreground within visually selected background; not verified GT FP rate.",
                                    qc_flags=list(result.qc_flags), parameters=result.diagnostics.get("parameters", params),
                                    diagnostics=result.diagnostics,
                                    control_flag="background_foreground_detected" if area else "no_background_foreground_detected")
                        info.update({key: str((folder / method / value).relative_to(out)).replace("\\", "/") for key, value in saved.items()})
                        del result
                    except Exception as exc:
                        info.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                        summary["error_count"] += 1
                    _json(folder / f"{method}.json", info)
                    csv_rows.append({"control_type": summary["control_type"], "source_sample_id": row["sample_id"],
                                     "source_video_id": row["video_id"], "crop_xyxy": json.dumps(spec["xyxy"]),
                                     "source_pixel_sha256": record["source_pixel_sha256"],
                                     **{key: info.get(key) for key in ("method", "status", "false_positive_pixels", "roi_area_px", "false_positive_roi_fraction", "control_flag", "error")}})
                record["status"] = "ok" if all(info["status"] == "ok" for info in record["methods"].values()) else "error"
            _comparison(out, record, crop)
            del crop
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            summary["error_count"] += 1
        _json(folder / "control.json", record)
        _json(out / "negative_controls.json", summary)
    if csv_rows:
        with (out / "negative_controls.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    summary["code_changed_during_run"] = _code_hashes() != summary["code_sha256"]
    if summary["code_changed_during_run"]:
        summary["error_count"] += 1
    summary["status"] = "completed_with_errors" if summary["error_count"] else ("preview_only" if args.preview_only else "completed")
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _json(out / "negative_controls.json", summary)
    print(f"{summary['status']}: {out}", flush=True)
    for record in summary["controls"]:
        for method, info in record["methods"].items():
            print(record["source_sample_id"], method, info.get("status"),
                  f"foreground={info.get('false_positive_pixels')} fraction={info.get('false_positive_roi_fraction')}", flush=True)
    return 1 if summary["error_count"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing raw24 benchmark directory")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--config", type=Path, help="Optional method overrides; ROI refinement and food exclusion are always disabled")
    parser.add_argument("--preview-only", action="store_true", help="Export native crop/location evidence without invoking segmentation")
    args = parser.parse_args(argv)
    try:
        return run_controls(args)
    except Exception as exc:
        print(f"Background controls failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
