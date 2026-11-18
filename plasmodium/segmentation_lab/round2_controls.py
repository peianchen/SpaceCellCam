"""Compare round 1 and round 2 on four visually selected real-background crops.

No independently confirmed pixel labels exist for these controls. Report
foreground detections and additive changes, never ground-truth accuracy.
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
from .negative_controls import CONTROL_CROPS, LIMITATION, _source_image, _context_image
from .visualize import _font, save_method_images, write_image


METHODS = ("round1_adaptive_chroma", "round2_local_ridge")


def _read_config(path: Path | None, method: str) -> dict:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Config must contain a JSON object")
    if "methods" in value:
        value = value["methods"].get(method, {})
    if not isinstance(value, dict):
        raise ValueError("Method config must contain a JSON object")
    return dict(value)


def _code_hashes() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    names = ("round2_controls.py", "segmenter_round2.py", "segmenter.py", "contracts.py",
             "negative_controls.py", "benchmark.py", "visualize.py")
    return {name: file_sha256(folder / name) for name in names}


def _figure(out: Path, item: dict, crop: np.ndarray) -> None:
    cards = [("Original crop", "native pixels / visually selected background", crop)]
    for method in METHODS:
        record = item["methods"][method]
        path = out / record["overlay_path"]
        panel = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if panel is None:
            raise OSError(f"Cannot read overlay: {path}")
        message = f"foreground {record['foreground_pixels']} / {record['roi_area_px']} px"
        message += f"\nROI fraction {record['roi_fraction']:.6f}"
        if method == METHODS[1]:
            message += f"; added {record['added_vs_round1_px']} px"
        cards.append((method, message, panel))
    width, height = 460, max(224, crop.shape[0]) + 160
    canvas = Image.new("RGB", (width * len(cards), height), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    title, text = _font(19, True), _font(15)
    for i, (name, message, panel) in enumerate(cards):
        left = i * width
        draw.text((left + 12, 12), name, font=title, fill="#173952")
        draw.multiline_text((left + 12, 43), message, font=text, fill="#344c61", spacing=5)
        draw.text((left + 12, 97), "Development control; no independent pixel labels", font=text, fill="#344c61")
        rgb = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        canvas.paste(rgb, (left + (width - rgb.width) // 2, 135))
    canvas.save(out / item["comparison_path"], quality=94)


def run_controls(args: argparse.Namespace) -> int:
    from .segmenter_round2 import DEFAULTS, prepare_baseline, segment as segment_round2
    run, out = args.run.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Output must not exist: {out}")
    manifest_path = run / "samples.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("source_mode") != "raw":
        raise ValueError("--run must be a raw-frame benchmark output")
    samples = {row["sample_id"]: row for row in data["samples"]}
    missing = [spec["sample_id"] for spec in CONTROL_CROPS if spec["sample_id"] not in samples]
    if missing:
        raise ValueError(f"Required previously reviewed control sources are missing: {missing}")
    if len({samples[spec["sample_id"]]["video_id"] for spec in CONTROL_CROPS}) != 4:
        raise ValueError("Controls must cover four distinct source videos")
    config = _read_config(args.config, "local_ridge_round2")
    base_config = _read_config(args.base_config, "adaptive_chroma")
    base_config.update(refine_roi=False, food_exclusion=False)
    out.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "control_type": "visually_selected_background_control",
               "evaluation_role": "development_seen", "ground_truth_available": False,
               "metric_interpretation": "Foreground detections in visually selected background, not independently verified error rates.",
               "limitation": LIMITATION, "source_run": str(run), "source_samples_sha256": file_sha256(manifest_path),
               "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "error_count": 0,
               "code_sha256": _code_hashes(), "environment": _environment(),
               "round2_parameters": {**DEFAULTS, **config}, "round1_overrides": base_config,
               "timing_note": "Round-2 elapsed_seconds measures the additive path with cached baseline; elapsed_total_including_baseline_seconds includes the separately timed round-1 call.",
               "analysis_resizing": False, "roi_policy": "crop center, radius 0.45*short side, margin 0; ROI refinement disabled",
               "controls": []}
    for key, path in (("config_file", args.config), ("base_config_file", args.base_config)):
        if path:
            summary[key] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    _json(out / "round2_controls.json", summary)
    metrics = []
    for i, spec in enumerate(CONTROL_CROPS, 1):
        row = samples[spec["sample_id"]]
        folder = out / f"control_{i:02d}"
        folder.mkdir()
        item = {"source_sample_id": row["sample_id"], "source_video_id": row["video_id"],
                "source_video": row.get("source_video"), "source_frame_index": row["frame_index"],
                "source_group": row.get("group"), "crop_xyxy": spec["xyxy"],
                "visual_selection_note": spec["visual_note"], "methods": {}, "status": "running"}
        summary["controls"].append(item)
        try:
            image, source_path = _source_image(run, row)
            x0, y0, x1, y1 = spec["xyxy"]
            if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                raise ValueError("The fixed crop lies outside source pixels")
            crop = image[y0:y1, x0:x1].copy()
            item.update(source_image_path=str(source_path), source_file_sha256=file_sha256(source_path),
                        source_pixel_sha256=pixel_sha256(image), crop_pixel_sha256=pixel_sha256(crop),
                        crop_shape=list(crop.shape), crop_path=f"control_{i:02d}/raw_crop.png",
                        context_path=f"control_{i:02d}/context.jpg", comparison_path=f"control_{i:02d}/comparison.jpg")
            write_image(out / item["crop_path"], crop)
            write_image(out / item["context_path"], _context_image(image, spec["xyxy"]))
            del image
            roi = RoiSpec(((crop.shape[1] - 1) / 2, (crop.shape[0] - 1) / 2), min(crop.shape[:2]) * .45, 0.)
            item["roi"] = {"center": roi.center, "radius": roi.radius, "margin_px": roi.margin_px}
            started = time.perf_counter()
            baseline = prepare_baseline(crop.copy(), roi, base_config=base_config)
            baseline_seconds = time.perf_counter() - started
            baseline_mask = baseline.mask.copy()
            for method in METHODS:
                if method == METHODS[0]:
                    result, elapsed = baseline, baseline_seconds
                else:
                    started = time.perf_counter()
                    result = segment_round2(crop.copy(), roi, config=config, base_config=base_config, baseline=baseline)
                    elapsed = time.perf_counter() - started
                    if not np.array_equal(baseline.mask, baseline_mask):
                        raise ValueError("Round 2 mutated its supplied baseline")
                    if np.any(baseline_mask & ~result.mask):
                        raise ValueError("Round 2 removed baseline pixels despite additive contract")
                    if not np.array_equal(result.roi_mask, baseline.roi_mask) or not np.array_equal(result.excluded_food, baseline.excluded_food):
                        raise ValueError("Round 2 changed frozen baseline ROI or food exclusion")
                result.validate(crop.shape[:2])
                products = save_method_images(folder / method, crop, result)
                record = {"status": "ok", "foreground_pixels": int(result.mask.sum()),
                          "roi_area_px": int(result.roi_mask.sum()),
                          "roi_fraction": float(result.mask.sum() / max(1, result.roi_mask.sum())),
                          "elapsed_seconds": elapsed, "qc_flags": list(result.qc_flags),
                          "elapsed_total_including_baseline_seconds": elapsed + (baseline_seconds if method == METHODS[1] else 0),
                          "timing_scope": "cached_additive_path_only" if method == METHODS[1] else "round1_full_segmentation",
                          "diagnostics": result.diagnostics,
                          "added_vs_round1_px": int((result.mask & ~baseline_mask).sum()),
                          "removed_vs_round1_px": int((baseline_mask & ~result.mask).sum())}
                record.update({key: str((folder / method / value).relative_to(out)).replace("\\", "/") for key, value in products.items()})
                if method == METHODS[1]:
                    additions = result.mask & ~baseline_mask
                    write_image(folder / "added_vs_round1.png", additions.astype(np.uint8) * 255)
                    record["additions_path"] = f"control_{i:02d}/added_vs_round1.png"
                item["methods"][method] = record
                _json(folder / f"{method}.json", record)
                metrics.append({"control_type": summary["control_type"], "source_sample_id": row["sample_id"],
                                "source_video_id": row["video_id"], "crop_xyxy": json.dumps(spec["xyxy"]),
                                "method": method, "status": "ok", "source_pixel_sha256": item["source_pixel_sha256"],
                                **{key: record[key] for key in ("foreground_pixels", "roi_area_px", "roi_fraction", "added_vs_round1_px", "removed_vs_round1_px", "elapsed_seconds", "elapsed_total_including_baseline_seconds", "timing_scope")}})
            _figure(out, item, crop)
            item["status"] = "ok"
            del crop, baseline, result
        except Exception as exc:
            item.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            summary["error_count"] += 1
        _json(folder / "control.json", item)
        _json(out / "round2_controls.json", summary)
    if metrics:
        with (out / "round2_controls.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    final_hashes = _code_hashes()
    summary["code_changed_during_run"] = final_hashes != summary["code_sha256"]
    if summary["code_changed_during_run"]:
        summary["code_sha256_after"] = final_hashes
        summary["error_count"] += 1
    summary["status"] = "completed_with_errors" if summary["error_count"] else "completed"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _json(out / "round2_controls.json", summary)
    print(f"{summary['status']}: {out}", flush=True)
    for item in summary["controls"]:
        for method, result in item["methods"].items():
            print(item["source_sample_id"], method, "foreground", result["foreground_pixels"],
                  "roi_fraction", result["roi_fraction"], "added", result["added_vs_round1_px"], flush=True)
    return 1 if summary["error_count"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing raw24 benchmark output")
    parser.add_argument("--out", type=Path, required=True, help="New, non-existing output directory")
    parser.add_argument("--config", type=Path, help="Round 2 parameter overrides")
    parser.add_argument("--base-config", type=Path, help="Round 1 overrides; food exclusion and ROI refinement are forced off for patches")
    args = parser.parse_args(argv)
    try:
        return run_controls(args)
    except Exception as exc:
        print(f"Round 2 controls failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
