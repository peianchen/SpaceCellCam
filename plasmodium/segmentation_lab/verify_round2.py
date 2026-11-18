"""Replay a completed round-2 comparison against the final, captured source code.

Original run snapshots and results are read-only. Verification establishes
mask/food reproducibility and input integrity, not biological segmentation
accuracy. --decode-raw also re-decodes every selected source-video frame.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .benchmark import _environment, _json, _load_source, file_sha256, pixel_sha256
from .contracts import RoiSpec
from .visualize import write_image


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Recorded output escapes the run directory: {value}")
    return path


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.shape != shape or not np.isin(raw, (0, 255)).all():
        raise ValueError(f"Expected binary 0/255 mask in source coordinates: {path}")
    return raw > 0


def _mask_hash(mask: np.ndarray) -> str:
    digest = hashlib.sha256(f"binary_bool|{mask.shape}".encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(mask)))
    return digest.hexdigest()


def _geometry(value: dict) -> dict:
    return {"center": [float(x) for x in value["center"]], "radius": float(value["radius"]),
            "margin_px": float(value.get("margin_px", 8))}


def _code_hashes(names: set[str]) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: file_sha256(root / name) for name in sorted(names)}


def _video_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"source_file_bytes": stat.st_size, "source_file_mtime_ns": stat.st_mtime_ns}


def verify(args: argparse.Namespace) -> int:
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new verification output directory: {output}")
    results_path, config_path, geometry_path = run / "results.json", run / "config.json", run / "geometry.json"
    recorded = _read_json(results_path)
    config, geometry = _read_json(config_path), _read_json(geometry_path)
    samples = recorded.get("samples", [])
    if not samples or len(samples) != recorded.get("sample_count"):
        raise ValueError("Run must be complete, with nonempty samples matching sample_count")
    sample_ids = [row["sample_id"] for row in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs in results")
    if config != recorded["parameters"] or geometry["methods"]["adaptive_chroma"] != recorded["base_config"]:
        raise ValueError("Saved config/geometry disagree with parameters recorded in results")
    original_hashes = recorded.get("source_hashes", {})
    if not original_hashes or any(Path(name).name != name for name in original_hashes):
        raise ValueError("Run has no valid source snapshot hash inventory")
    names = set(original_hashes) | {"verify_round2.py", "segmenter_round2.py", "segmenter.py", "contracts.py", "benchmark.py", "visualize.py"}
    final_hashes = _code_hashes(names)
    # Import final computation only after capturing its on-disk hashes.
    from .segmenter_round2 import DEFAULTS, prepare_baseline, segment
    if _code_hashes(names) != final_hashes:
        raise RuntimeError("Source code changed while importing the final core; retry when editing has stopped")
    output.mkdir(parents=True, exist_ok=False)
    source_bundle = output / "final_source"
    source_bundle.mkdir()
    for name in sorted(names):
        shutil.copy2(Path(__file__).resolve().parent / name, source_bundle / name)
        if file_sha256(source_bundle / name) != final_hashes[name]:
            raise RuntimeError(f"Final source changed while capturing {name}")
    tracked_files: dict[str, str] = {}
    def track(path: Path) -> str:
        digest = file_sha256(path)
        key = str(path.resolve())
        if key in tracked_files and tracked_files[key] != digest:
            raise ValueError(f"Input artifact changed during verification: {path}")
        tracked_files[key] = digest
        return digest
    for path in (results_path, config_path, geometry_path):
        track(path)
    summary = {"schema_version": 1, "verification_type": "final_core_replay_not_biological_accuracy",
               "run": str(run), "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "expected_sample_count": len(samples), "decode_raw": args.decode_raw,
               "source_snapshot_statement": "Original run source snapshots are preserved. This verification recomputes outputs with the separately captured final source; it does not claim the original run executed that final source.",
               "source_video_scope": "Compare current video size/mtime with historical input manifests and again at completion. With --decode-raw, selected decoded frame pixels are also compared. No historical whole-video content hashes exist; unchanged bytes outside selected frames are not established.",
               "ground_truth_available": False, "accuracy_metrics": None,
               "recorded_run_source_sha256": original_hashes, "final_source_sha256": final_hashes,
               "source_snapshot_checks": [], "changed_source_files": [], "input_manifests": [],
               "parameters_recorded": config, "parameters_effective": {**DEFAULTS, **config},
               "base_config": recorded["base_config"], "environment": _environment(),
               "samples": [], "source_videos": {}, "errors": []}
    _json(output / "verification.json", summary)
    original_sources = {}
    try:
        for name, expected in original_hashes.items():
            snapshot = run / "source" / name
            actual = track(snapshot)
            check = {"file": name, "recorded_sha256": expected, "snapshot_sha256": actual,
                     "snapshot_matches_record": actual == expected,
                     "final_sha256": final_hashes[name], "final_matches_snapshot": actual == final_hashes[name]}
            summary["source_snapshot_checks"].append(check)
            if actual != expected:
                raise ValueError(f"Original run source snapshot changed: {name}")
            if actual != final_hashes[name]:
                summary["changed_source_files"].append(name)
        for entry in recorded["input_manifests"]:
            manifest = Path(entry["path"]).resolve()
            actual = track(manifest)
            check = {"path": str(manifest), "recorded_sha256": entry["sha256"],
                     "current_sha256": actual, "matches": actual == entry["sha256"]}
            summary["input_manifests"].append(check)
            if not check["matches"]:
                raise ValueError(f"Input manifest changed since original run: {manifest}")
            input_data = _read_json(manifest)
            for row in input_data["samples"]:
                sid = row["sample_id"]
                if sid in original_sources:
                    raise ValueError(f"Source sample appears in multiple input manifests: {sid}")
                original_sources[sid] = (row, manifest)
        if set(sample_ids) != set(original_sources):
            raise ValueError("Recorded samples and input-manifest sample set differ")
    except Exception as exc:
        summary["errors"].append(f"Preflight {type(exc).__name__}: {exc}")
        summary["status"] = "failed_preflight"
        _json(output / "verification.json", summary)
        return 1
    _json(output / "verification.json", summary)
    started = time.perf_counter()
    for number, row in enumerate(samples, 1):
        sid = row["sample_id"]
        item = {"sample_id": sid, "status": "running", "mask_checks": {}, "food_checks": {}}
        summary["samples"].append(item)
        sample_out = _inside(output, "samples/" + sid)
        sample_out.mkdir(parents=True)
        print(f"[{number}/{len(samples)}] final-core replay {sid}", flush=True)
        try:
            source, source_manifest = original_sources[sid]
            if str(source.get("source_mode", "raw")) != "raw":
                raise ValueError("Only raw source-coordinate runs are supported")
            if source["source_pixel_sha256"] != row["source_pixel_sha256"]:
                raise ValueError("Source and run pixel-hash metadata disagree")
            if str(source["source_video"]) != str(row["source_video"]) or int(source["frame_index"]) != int(row["frame_index"]):
                raise ValueError("Source video/frame metadata differs between run and input manifest")
            video = Path(source["source_video"]).resolve()
            stat = _video_stat(video)
            expected_stat = {key: int(source[key]) for key in ("source_file_bytes", "source_file_mtime_ns")}
            if stat != expected_stat:
                raise ValueError(f"Source video metadata changed since extraction: {video}")
            video_key = str(video)
            previous = summary["source_videos"].get(video_key)
            if previous and previous["stat_before"] != stat:
                raise ValueError(f"Source video metadata changed during replay: {video}")
            summary["source_videos"].setdefault(video_key, {"stat_before": stat, "historical_manifest_metadata_matches": True,
                                                          "historical_whole_file_hash_available": False, "checked_sample_ids": []})
            summary["source_videos"][video_key]["checked_sample_ids"].append(sid)
            image_path = _inside(run, row.get("original_path") or row["image_path"])
            item["original_file_sha256"] = track(image_path)
            image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
            if image is None or pixel_sha256(image) != row["source_pixel_sha256"]:
                raise ValueError("Saved original pixels differ from their recorded source hash")
            item.update(source_pixel_sha256=row["source_pixel_sha256"], source_video_metadata_matches=True,
                        saved_original_pixel_hash_matches=True, source_video=str(video), frame_index=int(row["frame_index"]))
            original_input = (source_manifest.parent / source["image_path"]).resolve()
            item["input_image_file_sha256"] = track(original_input)
            input_image = cv2.imdecode(np.fromfile(str(original_input), np.uint8), cv2.IMREAD_COLOR)
            if input_image is None or pixel_sha256(input_image) != row["source_pixel_sha256"]:
                raise ValueError("Input-manifest source PNG changed")
            del input_image
            if args.decode_raw:
                raw, details = _load_source(source, source_manifest, "raw")
                if details["source_pixel_sha256"] != row["source_pixel_sha256"] or not np.array_equal(raw, image):
                    raise ValueError("Re-decoded source video frame differs from saved raw pixels")
                item.update(raw_video_frame_bit_equal=True, decoded_frame_index=details["decoded_frame_index"])
                del raw
            else:
                item["raw_video_frame_bit_equal"] = None
            expected_geometry = dict(geometry["roi"])
            expected_geometry.update(geometry["new_method_roi_by_video"][row["video_id"]])
            if _geometry(expected_geometry) != _geometry(row["roi"]):
                raise ValueError("Recorded sample ROI disagrees with saved geometry config")
            roi = RoiSpec(**_geometry(row["roi"]))
            item["roi"] = _geometry(row["roi"])
            call_started = time.perf_counter()
            before = prepare_baseline(image, roi, base_config=recorded["base_config"])
            after = segment(image, roi, config=config, base_config=recorded["base_config"], baseline=before)
            item["recompute_seconds"] = time.perf_counter() - call_started
            for name, result in (("round1", before), ("round2", after)):
                result.validate(image.shape[:2])
                mask_path = _inside(run, row[name + "_mask_path"])
                food_path = mask_path.parent / "food_mask.png"
                for kind, path, recomputed in (("mask", mask_path, result.mask), ("food", food_path, result.excluded_food)):
                    saved_file_hash = track(path)
                    saved = _mask(path, image.shape[:2])
                    mismatch = int(np.count_nonzero(saved != recomputed))
                    check = {"bit_equal": mismatch == 0, "different_pixels": mismatch,
                             "saved_path": str(path), "saved_file_sha256": saved_file_hash,
                             "saved_mask_pixel_sha256": _mask_hash(saved), "final_mask_pixel_sha256": _mask_hash(recomputed)}
                    item[kind + "_checks"][name] = check
                    if mismatch:
                        diff_name = f"{name}_{kind}_difference.png"
                        write_image(sample_out / diff_name, (saved != recomputed).astype(np.uint8) * 255)
                        check["difference_path"] = str((sample_out / diff_name).relative_to(output)).replace("\\", "/")
                if int(result.mask.sum()) != int(row[name + "_area_px"]):
                    raise ValueError(f"{name} area differs from recorded metric")
            added, removed = int((after.mask & ~before.mask).sum()), int((before.mask & ~after.mask).sum())
            if added != int(row["added_px"]) or removed != int(row["removed_px"]):
                raise ValueError("Added/removed pixel counts differ from original results")
            item.update(added_px=added, removed_px=removed, final_qc_flags=list(after.qc_flags),
                        recorded_qc_flags=row.get("qc_flags", []),
                        qc_flags_equal=list(after.qc_flags) == row.get("qc_flags", []),
                        final_diagnostics=after.diagnostics)
            item["all_masks_and_food_bit_equal"] = all(check["bit_equal"] for kind in ("mask_checks", "food_checks") for check in item[kind].values())
            if not item["all_masks_and_food_bit_equal"]:
                raise ValueError("At least one final-core mask or food exclusion differs from original output")
            item["status"] = "passed"
            del before, after, image
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            summary["errors"].append(f"{sid}: {item['error']}")
            print(f"  FAILED {sid}: {item['error']}", file=sys.stderr, flush=True)
        _json(sample_out / "verification.json", item)
        _json(output / "verification.json", summary)
    for path, info in summary["source_videos"].items():
        try:
            info["stat_after"] = _video_stat(Path(path))
            info["unchanged_during_verification_by_stat"] = info["stat_after"] == info["stat_before"]
            if not info["unchanged_during_verification_by_stat"]:
                summary["errors"].append(f"Source video metadata changed during verification: {path}")
        except OSError as exc:
            summary["errors"].append(f"Source video unreadable after verification: {path}: {exc}")
    changed_artifacts = []
    for path, expected in tracked_files.items():
        try:
            if file_sha256(Path(path)) != expected:
                changed_artifacts.append(path)
        except OSError:
            changed_artifacts.append(path)
    summary["changed_input_artifacts_during_verification"] = changed_artifacts
    if changed_artifacts:
        summary["errors"].append("Input artifacts changed during verification")
    summary["final_code_changed_during_verification"] = _code_hashes(names) != final_hashes
    if summary["final_code_changed_during_verification"]:
        summary["errors"].append("Final source code changed during verification; replay is not one fixed revision")
    summary.update(status="passed" if not summary["errors"] else "failed",
                   checked_sample_count=len(summary["samples"]), passed_sample_count=sum(x["status"] == "passed" for x in summary["samples"]),
                   elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat(),
                   checked_artifact_sha256=tracked_files)
    with (output / "verification.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["sample_id", "status", "all_masks_and_food_bit_equal", "raw_video_frame_bit_equal", "source_video_metadata_matches",
                  "added_px", "removed_px", "qc_flags_equal", "recompute_seconds", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["samples"])
    _json(output / "verification.json", summary)
    print(f"{summary['status']}: {summary['passed_sample_count']}/{len(samples)} samples; {output}", flush=True)
    return 0 if summary["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed round2_benchmark output containing results.json")
    parser.add_argument("--output", type=Path, required=True, help="New verification directory; never overwrites the original run")
    parser.add_argument("--decode-raw", action="store_true", help="Re-decode every recorded frame from the original videos and require pixel equality")
    args = parser.parse_args(argv)
    try:
        return verify(args)
    except Exception as exc:
        print(f"Round2 verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
