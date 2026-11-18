"""Sequential development comparison with source provenance and explicit failures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import RoiSpec
from .visualize import save_comparison, save_contact_sheets, save_gallery, save_method_images, write_image

METHODS = ("legacy_original", "v2_spatial", "local_lab", "adaptive_chroma")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default,
                               allow_nan=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"BGR|{image.shape}|{image.dtype}".encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(image)))
    return digest.hexdigest()


def select_samples(rows: list[dict[str, str]], per_section: int, limit: int = 0) -> list[dict[str, str]]:
    if per_section < 0 or limit < 0:
        raise ValueError("per-section and limit must be nonnegative")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["video_id"], row["section"]), []).append(row)
    selected = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: (int(row["frame_index"]), row["sample_id"]))
        count = len(ordered) if per_section == 0 else min(per_section, len(ordered))
        # Quantile-bin centers: one sample selects the middle; no duplicate indices.
        indices = [min(len(ordered) - 1, int((i + 0.5) * len(ordered) / count)) for i in range(count)]
        selected.extend(dict(ordered[index]) for index in indices)
    return selected[:limit] if limit else selected


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "video_id", "section", "frame_index", "source_video", "image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain nonempty rows with fields {sorted(required)}")
    identifiers = [row["sample_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Manifest contains duplicate sample_id values")
    return rows


def _config(path: Path | None, manifest: Path) -> tuple[dict[str, Any], RoiSpec, str]:
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path else {}
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    roi_config = config.get("roi")
    roi_source = str(path) if roi_config else "default dish geometry (1280,920), r=680"
    if not roi_config:
        candidate = manifest.parent.parent / "pilot_config.json"
        if candidate.exists():
            pilot = json.loads(candidate.read_text(encoding="utf-8-sig"))
            roi_config = {"center": pilot["dish_center"], "radius": pilot["dish_radius"], "margin_px": 8.0}
            roi_source = str(candidate)
        else:
            roi_config = {"center": [1280, 920], "radius": 680, "margin_px": 8.0}
    center = tuple(float(x) for x in roi_config["center"])
    radius, margin = float(roi_config["radius"]), float(roi_config.get("margin_px", 8))
    if len(center) != 2 or not all(np.isfinite(x) for x in (*center, radius, margin)) or radius <= 0 or margin < 0 or margin >= radius:
        raise ValueError("ROI needs finite 2D center and 0 <= margin_px < radius")
    return config, RoiSpec(center, radius, margin), roi_source


def _method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    if "methods" in config:
        return dict(config["methods"].get(method, {}))
    if method in {"legacy_original", "v2_spatial"}:
        return {}
    return {key: value for key, value in config.items() if key != "roi"}


def _method_roi(config: dict, row: dict, default: RoiSpec, method: str) -> RoiSpec:
    """Per-video dish geometry calibration applies only to the new pipeline."""
    item = config.get("new_method_roi_by_video", {}).get(row["video_id"])
    if item is None or method in {"legacy_original", "v2_spatial"}:
        return default
    center = tuple(float(x) for x in item["center"])
    radius, margin = float(item["radius"]), float(item.get("margin_px", default.margin_px))
    if len(center) != 2 or not np.isfinite([*center, radius, margin]).all() or not 0 <= margin < radius:
        raise ValueError("Invalid per-video dish calibration")
    return RoiSpec(center, radius, margin)


def _raw_cache(cache: Path | None) -> dict:
    if cache is None:
        return {}
    cache = cache.resolve()
    run = json.loads((cache / "run_manifest.json").read_text(encoding="utf-8"))
    if run["status"] != "completed" or run["source_mode"] != "raw" or run.get("code_changed_during_run"):
        raise ValueError("Raw cache must be a completed intact raw-source run")
    rows = json.loads((cache / "samples.json").read_text(encoding="utf-8"))["samples"]
    return {row["sample_id"]: (cache, row) for row in rows}


def _load_cached_raw(row: dict, item: tuple) -> tuple[np.ndarray, dict]:
    root, old = item
    video = Path(row["source_video"]).resolve()
    if str(video) != str(Path(old["source_video"]).resolve()) or int(row["frame_index"]) != int(old["frame_index"]):
        raise ValueError("Cached image belongs to a different video/frame")
    stat = video.stat()
    if stat.st_size != old["source_file_bytes"] or stat.st_mtime_ns != old["source_file_mtime_ns"]:
        raise ValueError("Source video changed since raw extraction")
    image_path = root / old["image_path"]
    image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    if image is None or pixel_sha256(image) != old["source_pixel_sha256"]:
        raise ValueError("Cached raw image changed")
    keys = ("source_path", "decoded_frame_index", "video_fps", "video_frame_count", "source_file_bytes",
            "source_file_mtime_ns", "source_pixel_sha256", "pixel_hash_format", "image_width", "image_height",
            "biological_sampling_interval_seconds", "time_note")
    details = {key: old[key] for key in keys if key in old}
    details.update(raw_cache_path=str(image_path), raw_cache_run=str(root), registration_reapplied=False,
                   source_is_prepared_registered_image=False, reused_verified_raw_pixels=True)
    return image, details


def _run_method(image: np.ndarray, roi: RoiSpec, method: str, config: dict[str, Any]):
    if method in {"legacy_original", "v2_spatial"}:
        if config:
            raise ValueError(f"{method} uses recorded original parameters; custom overrides are not supported")
        from .baselines import run_baseline
        return run_baseline(image, roi, method)
    from .segmenter import segment
    return segment(image, roi, method=method, config=config or None)


def _relative(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _load_source(row: dict[str, str], manifest: Path, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "prepared":
        path = _relative(row["image_path"], manifest.parent).resolve()
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not decode prepared image: {path}")
        details = {"source_path": str(path), "source_file_sha256": file_sha256(path),
                   "registration_reapplied": False, "source_is_prepared_registered_image": True}
    else:
        path = _relative(row["source_video"], manifest.parent).resolve()
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise OSError(f"Could not open raw source video: {path}")
            index = int(row["frame_index"])
            if index < 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise OSError(f"Could not seek frame {index}: {path}")
            ok, image = cap.read()
            if not ok or image is None:
                raise OSError(f"Could not decode raw frame {index}: {path}")
            actual = float(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1.0
            if np.isfinite(actual) and abs(actual - index) > 0.5:
                raise OSError(f"Decoder returned frame {actual} instead of {index}: {path}")
            details = {"source_path": str(path), "decoded_frame_index": actual,
                       "video_fps": float(cap.get(cv2.CAP_PROP_FPS)),
                       "video_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                       "registration_reapplied": False, "source_is_prepared_registered_image": False}
        finally:
            cap.release()
    stat = path.stat()
    details.update(source_file_bytes=stat.st_size, source_file_mtime_ns=stat.st_mtime_ns,
                   source_pixel_sha256=pixel_sha256(image), pixel_hash_format="SHA256(BGR|shape|dtype + contiguous bytes)",
                   image_width=int(image.shape[1]), image_height=int(image.shape[0]),
                   biological_sampling_interval_seconds=None,
                   time_note="Video fps is playback rate, not an established biological sampling interval.")
    return image, details


def _code_hashes() -> dict[str, str]:
    project = Path(__file__).resolve().parents[1]
    sources = [project / "segmentation_lab" / name for name in
               ("__init__.py", "contracts.py", "segmenter.py", "baselines.py", "benchmark.py", "visualize.py")]
    sources += list((project / "plasmodium_analysis_v2" / "plasmodium_v2").glob("*.py"))
    sources += list(project.glob("plasmodium_brightfield_analysis*.py"))
    return {str(path.relative_to(project)).replace("\\", "/"): file_sha256(path) for path in sorted(sources)}


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "scipy", "scikit-image", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(), "opencv_opencl_enabled": cv2.ocl.useOpenCL()}


def _write_tables(root: Path, samples: list[dict[str, Any]], records: list[dict[str, Any]], run: dict[str, Any]) -> None:
    _json(root / "samples.json", {"dataset_id": run["run_id"], "evaluation_role": "development_seen",
                                 "development_dataset_ids": [run["run_id"]],
                                 "source_manifest": run["source_manifest"], "source_mode": run["source_mode"], "samples": samples})
    sample_keys = list(dict.fromkeys(key for sample in samples for key in sample if key != "methods"))
    with (root / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_keys)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False, default=_json_default)
                                   if isinstance(value, (list, dict)) else value)
                             for key, value in sample.items() if key != "methods"})
    metric_keys = ["sample_id", "video_id", "group", "section", "frame_index", "source_mode", "source_split",
                   "evaluation_role", "method", "status", "area_px", "roi_area_px", "area_fraction_roi",
                   "food_excluded_px", "elapsed_seconds", "qc_flags", "error", "source_pixel_sha256"]
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_keys, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "qc_flags": " | ".join(record.get("qc_flags", []))})
    _json(root / "run_manifest.json", run)


def run_benchmark(args: argparse.Namespace) -> int:
    manifest, output = args.manifest.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output must not exist; choose a new run directory: {output}")
    rows = select_samples(_read_manifest(manifest), args.per_section, args.limit)
    if not rows:
        raise ValueError("No samples selected")
    config, roi, roi_source = _config(args.config.resolve() if args.config else None, manifest)
    raw_cache = _raw_cache(getattr(args, "raw_cache", None))
    if raw_cache and args.source != "raw":
        raise ValueError("--raw-cache is only valid for --source raw")
    methods = list(dict.fromkeys(args.methods))
    invalid = set(methods) - set(METHODS)
    if invalid or not methods:
        raise ValueError(f"Unknown or empty method list: {sorted(invalid)}")
    safe_ids = [re.sub(r"[^A-Za-z0-9_.-]", "_", row["sample_id"]) for row in rows]
    if len(set(safe_ids)) != len(safe_ids) or any(value in {"", ".", ".."} for value in safe_ids):
        raise ValueError("Sample IDs are not unique safe output folder names")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    run = {"schema_version": 1, "run_id": output.name,
           "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
           "evaluation_role": "development_seen", "source_manifest": str(manifest),
           "source_manifest_sha256": file_sha256(manifest), "source_mode": args.source,
           "methods": methods, "per_section": args.per_section, "limit": args.limit,
           "selected_sample_ids": [row["sample_id"] for row in rows], "selected_sample_count": len(rows),
           "selection_rule": "Per video/section: sorted frame indices, quantile-bin centers; per_section=0 selects all.",
           "source_split_note": "Original split labels are historical metadata. ALL selected samples are development_seen.",
           "config": config, "roi": {"center": roi.center, "radius": roi.radius, "margin_px": roi.margin_px},
           "roi_source": roi_source, "code_sha256": _code_hashes(), "environment": _environment(),
           "metric_note": "Area/QC/timing are diagnostic, not accuracy. No ground-truth comparison is performed.",
           "score_note": "Uncalibrated feature evidence; baseline scores may be binary masks.",
           "biological_sampling_interval_seconds": None, "error_count": 0}
    if args.config:
        run["config_file"] = str(args.config.resolve())
        run["config_sha256"] = file_sha256(args.config.resolve())
    samples, records = [], []
    _write_tables(output, samples, records, run)
    for number, (row, safe_id) in enumerate(zip(rows, safe_ids), 1):
        sample = {**row, "dataset_id": run["run_id"], "source_split": row.get("split", ""),
                  "evaluation_role": "development_seen", "source_mode": args.source,
                  "prepared_image_path": row["image_path"], "image_path": "", "methods": {},
                  "manifest_registration_metadata_only": args.source == "raw"}
        samples.append(sample)
        folder = output / "samples" / safe_id
        folder.mkdir(parents=True)
        print(f"[{number}/{len(rows)}] {row['sample_id']} source={args.source}", flush=True)
        try:
            image, details = (_load_cached_raw(row, raw_cache[row["sample_id"]])
                              if row["sample_id"] in raw_cache else _load_source(row, manifest, args.source))
            sample.update(details)
            if not roi.mask(image.shape[:2]).any():
                raise ValueError("Configured ROI does not overlap source image")
            write_image(folder / "original.png", image)
            sample["image_path"] = str((folder / "original.png").relative_to(output)).replace("\\", "/")
            sample["raw_image_path"] = sample["image_path"]
        except Exception as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
            sample["status"] = "source_error"
            _json(folder / "source_error.json", {"error": sample["error"], "traceback": traceback.format_exc(), "source_row": row})
            for method in methods:
                record = {**{key: sample.get(key, "") for key in ("sample_id", "video_id", "group", "section", "frame_index", "source_split")},
                          "source_mode": args.source, "evaluation_role": "development_seen", "method": method,
                          "status": "source_error", "error": sample["error"], "qc_flags": ["source_read_failed"]}
                sample["methods"][method] = record
                records.append(record)
                run["error_count"] += 1
            _write_tables(output, samples, records, run)
            continue
        method_records = []
        for method in methods:
            record = {key: sample.get(key, "") for key in ("sample_id", "video_id", "group", "section", "frame_index", "source_mode", "source_split", "evaluation_role", "source_pixel_sha256")}
            record["method"] = method
            method_start = time.perf_counter()
            try:
                method_config = _method_config(config, method)
                # Isolate in-place operations by any imported baseline from the common source image.
                method_roi = _method_roi(config, row, roi, method)
                result = _run_method(image.copy(), method_roi, method, method_config)
                record["input_roi"] = {"center": method_roi.center, "radius": method_roi.radius, "margin_px": method_roi.margin_px}
                record["elapsed_seconds"] = time.perf_counter() - method_start
                result.validate(image.shape[:2])
                method_folder = folder / method
                paths = save_method_images(method_folder, image, result)
                record.update({key: str((method_folder / path).relative_to(output)).replace("\\", "/") for key, path in paths.items()})
                roi_area = int(result.roi_mask.sum())
                record.update(status="ok", area_px=int(result.mask.sum()), roi_area_px=roi_area,
                              area_fraction_roi=float(result.mask.sum() / max(1, roi_area)),
                              food_excluded_px=int(result.excluded_food.sum()), qc_flags=list(result.qc_flags),
                              diagnostics=result.diagnostics, config=method_config,
                              score_interpretation="Feature evidence, not calibrated probability; do not use as accuracy.")
                record["qc_path"] = str((method_folder / "qc.json").relative_to(output)).replace("\\", "/")
                _json(method_folder / "qc.json", record)
                del result
            except Exception as exc:
                record.update(status="error", elapsed_seconds=time.perf_counter() - method_start,
                              error=f"{type(exc).__name__}: {exc}", qc_flags=["method_failed"])
                (folder / method).mkdir(exist_ok=True)
                _json(folder / method / "error.json", {**record, "traceback": traceback.format_exc()})
                run["error_count"] += 1
                print(f"  ERROR {method}: {record['error']}", file=sys.stderr, flush=True)
            sample["methods"][method] = record
            method_records.append(record)
            records.append(record)
        good = [record for record in method_records if record["status"] == "ok"]
        if good:
            preferred = next((record for record in good if record["method"] == "adaptive_chroma"), good[0])
            sample.update(prediction_method=preferred["method"], prediction_mask_path=preferred["mask_path"], draft_mask_path=preferred["mask_path"])
        try:
            comparison = folder / "comparison.jpg"
            display_roi = RoiSpec((image.shape[1]/2, image.shape[0]/2), min(image.shape[:2])/2, 0)
            save_comparison(comparison, image, display_roi, sample, method_records, output)
            sample["comparison_path"] = str(comparison.relative_to(output)).replace("\\", "/")
        except Exception as exc:
            sample["visualization_error"] = f"{type(exc).__name__}: {exc}"
            run["error_count"] += 1
            _json(folder / "visualization_error.json", {"error": sample["visualization_error"], "traceback": traceback.format_exc()})
        sample["status"] = "ok" if len(good) == len(methods) and "visualization_error" not in sample else "error"
        del image
        _write_tables(output, samples, records, run)
    run.update(status="completed_with_errors" if run["error_count"] else "completed",
               finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start)
    try:
        run["contact_sheets"] = save_contact_sheets(output, samples)
        save_gallery(output, samples, run)
    except Exception as exc:
        run.update(status="completed_with_errors", presentation_error=f"{type(exc).__name__}: {exc}")
        run["error_count"] += 1
        _json(output / "presentation_error.json", {"error": run["presentation_error"], "traceback": traceback.format_exc()})
    # A source change during the run invalidates a claim that one code revision produced all results.
    final_hashes = _code_hashes()
    run["code_changed_during_run"] = final_hashes != run["code_sha256"]
    if run["code_changed_during_run"]:
        run["code_sha256_after"] = final_hashes
        run["error_count"] += 1
        run["status"] = "completed_with_errors"
    _write_tables(output, samples, records, run)
    # Refresh the short status banner after all integrity checks.
    if (output / "gallery.html").exists():
        save_gallery(output, samples, run)
    print(f"{run['status']}: {output} ({run['error_count']} errors)", flush=True)
    return 1 if run["error_count"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--per-section", type=int, default=1, help="Frames per video/section, 0 = all; 1 = middle")
    parser.add_argument("--source", choices=("raw", "prepared"), default="raw")
    parser.add_argument("--limit", type=int, default=0, help="0 = all selected frames")
    parser.add_argument("--config", type=Path, help='JSON: {"roi": {...}, "methods": {"adaptive_chroma": {...}}}')
    parser.add_argument("--raw-cache", type=Path, help="Reuse and verify unchanged raw frames from a prior completed run")
    args = parser.parse_args(argv)
    try:
        return run_benchmark(args)
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
