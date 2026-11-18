"""Explicit human confirmation and hash-checked mask evaluation (no auto ground truth)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pixel_sha256(image):
    """Same decoded BGR/shape/dtype contract as benchmark.pixel_sha256."""
    digest = hashlib.sha256()
    digest.update(f"BGR|{image.shape}|{image.dtype}".encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(image)))
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def read_mask(path, shape):
    mask = _image(path)
    if mask.ndim != 2 or mask.shape != tuple(shape) or not np.isin(mask, [0, 255]).all():
        raise ValueError(f"Mask must be single-channel 0/255 with shape {tuple(shape)}: {path}")
    return mask > 0


def _write_mask(path, mask):
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise OSError(f"Cannot write mask: {path}")


def _resolve(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def create_package(samples_file, destination, annotation_mode="assisted"):
    """Create a new package; source images, predictions and drafts remain separate."""
    if annotation_mode not in {"assisted", "independent"}:
        raise ValueError("annotation_mode must be assisted or independent")
    samples_file, destination = Path(samples_file).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new annotation package directory: {destination}")
    if samples_file.suffix.lower() == ".csv":
        with samples_file.open(encoding="utf-8-sig", newline="") as stream:
            data = {"samples": list(csv.DictReader(stream))}
    else:
        data = json.loads(samples_file.read_text(encoding="utf-8-sig"))
        data = {"samples": data} if isinstance(data, list) else data
    rows = data.get("samples", [])
    ids = [str(row.get("sample_id", "")) for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid) for sid in ids
    ):
        raise ValueError("Unique, filename-safe sample_id values are required")
    destination.mkdir(parents=True)
    for name in ("images", "draft_masks", "confirmed_masks", "evaluation_predictions"):
        (destination / name).mkdir()
    package = {
        "schema_version": 1, "annotation_mode": annotation_mode, "exclude_food": True,
        "source_manifest": str(samples_file), "source_manifest_sha256": file_hash(samples_file),
        "dataset_id": data.get("dataset_id"),
        "evaluation_role": data.get("evaluation_role", "unspecified"),
        "development_dataset_ids": data.get("development_dataset_ids", []),
        "samples": [],
    }
    for row, sid in zip(rows, ids):
        source = _resolve(samples_file.parent, row.get("raw_image_path") or row["image_path"])
        image = _image(source)
        expected_pixels = row.get("source_pixel_sha256")
        if expected_pixels and _pixel_sha256(image) != expected_pixels:
            raise ValueError(f"Source image changed since benchmark: {sid}")
        shape = image.shape[:2]
        image_rel = f"images/{sid}{source.suffix}"
        shutil.copy2(source, destination / image_rel)
        methods = row.get("methods") or {}
        methods = json.loads(methods) if isinstance(methods, str) else methods
        if not methods and row.get("prediction_mask_path"):
            methods = {row.get("prediction_method") or "prediction": {
                "status": "ok", "mask_path": row["prediction_mask_path"]}}
        predictions = {}
        for index, (method, info) in enumerate(methods.items()):
            if not info.get("mask_path") or info.get("status") in {"failed", "error"}:
                continue
            prediction = _resolve(samples_file.parent, info["mask_path"])
            read_mask(prediction, shape)
            rel = f"evaluation_predictions/{sid}_{index}.png"
            shutil.copy2(prediction, destination / rel)
            predictions[method] = {"path": rel, "sha256": file_hash(destination / rel)}
        draft_source = row.get("draft_mask_path") if annotation_mode == "assisted" else None
        draft = read_mask(_resolve(samples_file.parent, draft_source), shape) if draft_source else np.zeros(shape, bool)
        draft_rel = f"draft_masks/{sid}.png"
        _write_mask(destination / draft_rel, draft)
        package["samples"].append({
            "sample_id": sid, "image_path": image_rel, "shape": list(shape),
            "image_sha256": file_hash(destination / image_rel),
            "source_image": str(source), "source_pixel_sha256": row.get("source_pixel_sha256"),
            "source_mode": row.get("source_mode", "unspecified"),
            "source_is_prepared_registered_image": row.get("source_is_prepared_registered_image"),
            "video_id": row.get("video_id"), "source_video": row.get("source_video"),
            "frame_index": row.get("frame_index"), "source_split": row.get("source_split", row.get("split")),
            "draft_path": draft_rel, "draft_source": draft_source, "predictions": predictions,
            "status": "saved_draft", "confirmation": None, "history": [],
        })
    _write_json(destination / "annotations.json", package)
    return destination


def _load(package_path):
    root = Path(package_path).resolve()
    return root, json.loads((root / "annotations.json").read_text(encoding="utf-8"))


def _sample(package, sample_id):
    return next(row for row in package["samples"] if row["sample_id"] == sample_id)


def _check_image(root, row):
    if file_hash(root / row["image_path"]) != row["image_sha256"]:
        raise ValueError(f"Image changed; create a new package: {row['sample_id']}")


def save_draft(package_path, sample_id, mask_path):
    root, package = _load(package_path)
    row = _sample(package, sample_id)
    _check_image(root, row)
    mask = read_mask(mask_path, row["shape"])
    _write_mask(root / row["draft_path"], mask)
    if row["confirmation"]:
        row["history"].append(row["confirmation"])
    row.update(status="saved_draft", confirmation=None)
    _write_json(root / "annotations.json", package)


def confirm_mask(package_path, sample_id, mask_path, *, reviewer, source, exclude_food):
    """Only an explicit human request may call this; draft saves never confirm."""
    if not reviewer.strip() or source not in {"independent", "assisted"} or exclude_food is not True:
        raise ValueError("Reviewer, independent/assisted source and explicit exclude_food=True are required")
    root, package = _load(package_path)
    if source != package["annotation_mode"]:
        raise ValueError("Confirmation source must match the package annotation mode")
    row = _sample(package, sample_id)
    _check_image(root, row)
    mask_path = Path(mask_path).resolve()
    if source == "independent" and (root / "evaluation_predictions") in mask_path.parents:
        raise ValueError("An algorithm prediction cannot be independently drawn ground truth")
    mask = read_mask(mask_path, row["shape"])
    digest = file_hash(mask_path)
    rel = f"confirmed_masks/{sample_id}_{digest}.png"
    if not (root / rel).exists():
        shutil.copy2(mask_path, root / rel)
    if row["confirmation"]:
        row["history"].append(row["confirmation"])
    row["confirmation"] = {
        "reviewer": reviewer.strip(), "source": source, "exclude_food": True,
        "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_sha256": row["image_sha256"], "mask_sha256": digest,
        "mask_path": rel, "mask_source": str(mask_path), "foreground_pixels": int(mask.sum()),
    }
    row["status"] = "confirmed"
    _write_json(root / "annotations.json", package)


def mask_metrics(prediction, reference):
    p, r = np.asarray(prediction, bool), np.asarray(reference, bool)
    if p.shape != r.shape:
        raise ValueError("Prediction and reference shapes differ")
    tp, fp, fn, tn = [int(np.count_nonzero(v)) for v in (p & r, p & ~r, ~p & r, ~p & ~r)]
    area = tp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / area if area else 1.0,
        "prediction_area_px": tp + fp, "reference_area_px": area,
        "signed_area_error_px": fp - fn,
        "relative_area_error": (fp - fn) / area if area else None,
    }


def evaluate_package(package_path):
    root, package = _load(package_path)
    pending = []
    for row in package["samples"]:
        _check_image(root, row)
        c = row.get("confirmation")
        if row["status"] != "confirmed" or not c:
            pending.append(row["sample_id"])
            continue
        required = (c.get("reviewer", "").strip() and c.get("confirmed_at_utc")
                    and c.get("source") == package["annotation_mode"] and c.get("exclude_food") is True)
        if not required or c.get("image_sha256") != row["image_sha256"]:
            raise ValueError(f"Invalid confirmation: {row['sample_id']}")
        if file_hash(root / c["mask_path"]) != c["mask_sha256"]:
            raise ValueError(f"Confirmed mask changed: {row['sample_id']}")
    if pending:
        return {"status": "awaiting_human_confirmation", "metrics": None,
                "pending_samples": pending, "confirmed_count": len(package["samples"]) - len(pending),
                "independent_test_eligible": False}
    if not package["samples"]:
        raise ValueError("Package has no samples")
    results, metrics = [], {}
    for row in package["samples"]:
        reference = read_mask(root / row["confirmation"]["mask_path"], row["shape"])
        for method, info in row["predictions"].items():
            if file_hash(root / info["path"]) != info["sha256"]:
                raise ValueError(f"Prediction changed: {row['sample_id']} / {method}")
            result = mask_metrics(read_mask(root / info["path"], row["shape"]), reference)
            results.append({"sample_id": row["sample_id"], "method": method, **result})
    if not results:
        raise ValueError("No predictions available for evaluation")
    for method in sorted({r["method"] for r in results}):
        selected = [r for r in results if r["method"] == method]
        metrics[method] = {"sample_count": len(selected), **{
            name + "_mean": float(np.mean([r[name] for r in selected]))
            for name in ("dice", "iou", "precision", "recall")}, **{
            name: sum(r[name] for r in selected) for name in ("tp", "fp", "fn", "tn")}}
    dev_ids, dataset_id = package.get("development_dataset_ids"), package.get("dataset_id")
    candidate = (package["annotation_mode"] == "independent" and bool(dataset_id) and bool(dev_ids)
                 and dataset_id not in dev_ids and package.get("evaluation_role") == "independent_test")
    return {"status": "evaluated_confirmed_masks", "metrics": metrics, "per_sample": results,
            "confirmed_count": len(package["samples"]), "independent_test_eligible": False,
            "independent_test_status": "requires_study_design_review" if candidate else "not_established",
            "interpretation": "Agreement with the recorded human references; no automatic biological validation claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("samples", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--mode", choices=["assisted", "independent"], default="assisted")
    for name in ("draft", "confirm"):
        p = sub.add_parser(name)
        p.add_argument("package", type=Path)
        p.add_argument("sample_id")
        p.add_argument("mask", type=Path)
        if name == "confirm":
            p.add_argument("--reviewer", required=True)
            p.add_argument("--source", required=True, choices=["independent", "assisted"])
            p.add_argument("--exclude-food", action="store_true", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("package", type=Path)
    ev.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        print(create_package(args.samples, args.destination, args.mode))
    elif args.command == "draft":
        save_draft(args.package, args.sample_id, args.mask)
    elif args.command == "confirm":
        confirm_mask(args.package, args.sample_id, args.mask, reviewer=args.reviewer,
                     source=args.source, exclude_food=args.exclude_food)
    else:
        result = evaluate_package(args.package)
        if args.out:
            _write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
