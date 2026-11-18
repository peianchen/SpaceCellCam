"""Small synthetic checks for provenance, failure visibility, and source safety."""
import argparse
import csv
import json

import cv2
import numpy as np
import pytest

from segmentation_lab import benchmark
from segmentation_lab.contracts import SegmentationResult
from segmentation_lab.visualize import mask_overlay, write_image


def test_middle_selection_keeps_each_video_section_and_all_mode():
    rows = [{"sample_id": f"{video}_{section}_{i}", "video_id": video, "section": section,
             "frame_index": str(i * 10)}
            for video in ("a", "b") for section in ("early", "late") for i in reversed(range(5))]
    selected = benchmark.select_samples(rows, 1)
    assert len(selected) == 4
    assert {row["frame_index"] for row in selected} == {"20"}
    assert len(benchmark.select_samples(rows, 0)) == 20
    assert len(benchmark.select_samples(rows, 1, 2)) == 2


def _fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.full((32, 32, 3), 90, np.uint8)
    write_image(source / "image.png", image)
    manifest = source / "manifest.csv"
    row = {"sample_id": "test_sample", "video_id": "test_video", "group": "still", "section": "early",
           "frame_index": "12", "source_video": "unused.mp4", "image_path": "image.png", "split": "test"}
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"roi": {"center": [16, 16], "radius": 14, "margin_px": 1}}))
    args = argparse.Namespace(manifest=manifest, output=tmp_path / "run", config=config,
                              methods=["adaptive_chroma", "local_lab"], per_section=1, limit=0, source="prepared")
    return args, image


def _success(image, roi, method, config):
    inside = roi.mask(image.shape[:2])
    mask = inside.copy()
    mask[12:20, 12:20] = False
    return SegmentationResult(method, mask, image.copy(), mask.astype(np.float32), mask.copy(),
                              inside, np.zeros_like(inside), [], {"synthetic": True})


def test_failure_is_recorded_nonzero_and_success_remains_reviewable(tmp_path, monkeypatch):
    args, image = _fixture(tmp_path)
    source_hash = benchmark.file_sha256(args.manifest.parent / "image.png")
    def method(image, roi, name, config):
        if name == "local_lab":
            raise RuntimeError("intentional synthetic failure")
        return _success(image, roi, name, config)
    monkeypatch.setattr(benchmark, "_run_method", method)
    monkeypatch.setattr(benchmark, "_code_hashes", lambda: {"synthetic": "fixed"})
    assert benchmark.run_benchmark(args) == 1
    run = json.loads((args.output / "run_manifest.json").read_text())
    sample = json.loads((args.output / "samples.json").read_text())["samples"][0]
    assert run["status"] == "completed_with_errors" and run["error_count"] == 1
    assert sample["evaluation_role"] == "development_seen" and sample["source_split"] == "test"
    assert sample["source_pixel_sha256"] == benchmark.pixel_sha256(image)
    assert sample["methods"]["local_lab"]["status"] == "error"
    assert (args.output / sample["methods"]["adaptive_chroma"]["mask_path"]).exists()
    assert (args.output / "gallery.html").exists()
    assert benchmark.file_sha256(args.manifest.parent / "image.png") == source_hash
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(args)


def test_holes_are_not_painted_in_overlay():
    image = np.full((24, 24, 3), 80, np.uint8)
    mask = np.zeros((24, 24), bool)
    mask[3:21, 3:21] = True
    mask[9:15, 9:15] = False
    overlay = mask_overlay(image, mask)
    assert np.array_equal(overlay[~mask], image[~mask])
    assert np.array_equal(overlay[9:15, 9:15], image[9:15, 9:15])


def test_unreadable_source_is_explicit_and_never_a_success(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path)
    args.source = "raw"
    monkeypatch.setattr(benchmark, "_code_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_load_source", lambda *a: (_ for _ in ()).throw(OSError("synthetic missing source")))
    assert benchmark.run_benchmark(args) == 1
    run = json.loads((args.output / "run_manifest.json").read_text())
    assert run["error_count"] == 2
    assert run["status"] == "completed_with_errors"
