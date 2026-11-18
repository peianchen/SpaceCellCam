"""Synthetic-only checks: these masks never constitute research annotations."""
import csv
import hashlib
import json

import cv2
import numpy as np
import pytest

from segmentation_lab.annotations import (
    confirm_mask, create_package, evaluate_package, file_hash, mask_metrics, save_draft,
)


def _fixture(tmp_path, mode="assisted", as_csv=False):
    source = tmp_path / "benchmark"
    source.mkdir()
    image = np.zeros((12, 16, 3), np.uint8)
    image[4:8, 5:9] = (20, 180, 200)
    mask = np.zeros((12, 16), np.uint8)
    mask[4:8, 5:9] = 255
    cv2.imwrite(str(source / "original.png"), image)
    cv2.imwrite(str(source / "prediction.png"), mask)
    row = {"sample_id": "synthetic_01", "raw_image_path": "original.png",
           "image_path": "original.png", "prediction_mask_path": "prediction.png",
           "draft_mask_path": "prediction.png", "prediction_method": "synthetic_method",
           "source_split": "test", "frame_index": 2}
    path = source / ("samples.csv" if as_csv else "samples.json")
    if as_csv:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    else:
        path.write_text(json.dumps({"dataset_id": "synthetic_seen", "evaluation_role": "development_seen",
                                   "development_dataset_ids": ["synthetic_seen"], "samples": [row]}))
    package = create_package(path, tmp_path / "annotations", mode)
    return package, source / "prediction.png", source / "original.png"


def _confirm(package, mask, source="assisted"):
    confirm_mask(package, "synthetic_01", mask, reviewer="SYNTHETIC_TEST_ONLY",
                 source=source, exclude_food=True)


@pytest.mark.parametrize("as_csv", [False, True])
def test_export_has_no_accuracy_without_human_confirmation(tmp_path, as_csv):
    package, prediction, original = _fixture(tmp_path, as_csv=as_csv)
    result = evaluate_package(package)
    assert result["status"] == "awaiting_human_confirmation"
    assert result["metrics"] is None and result["confirmed_count"] == 0
    manifest = json.loads((package / "annotations.json").read_text())
    row = manifest["samples"][0]
    assert row["status"] == "saved_draft" and row["confirmation"] is None
    assert file_hash(package / row["image_path"]) == file_hash(original)


def test_explicit_confirmation_is_hashed_and_never_auto_publication(tmp_path):
    package, mask, _ = _fixture(tmp_path)
    original_hash = file_hash(mask)
    _confirm(package, mask)
    result = evaluate_package(package)
    assert result["metrics"]["synthetic_method"]["dice_mean"] == 1
    assert result["independent_test_eligible"] is False
    assert result["independent_test_status"] == "not_established"
    row = json.loads((package / "annotations.json").read_text())["samples"][0]
    assert row["confirmation"]["reviewer"] == "SYNTHETIC_TEST_ONLY"
    assert row["confirmation"]["mask_sha256"] == original_hash
    assert row["confirmation"]["confirmed_at_utc"]
    assert file_hash(mask) == original_hash


def test_draft_save_revokes_confirmation_but_keeps_previous_version(tmp_path):
    package, mask, _ = _fixture(tmp_path)
    _confirm(package, mask)
    save_draft(package, "synthetic_01", mask)
    row = json.loads((package / "annotations.json").read_text())["samples"][0]
    assert row["status"] == "saved_draft" and row["confirmation"] is None
    assert len(row["history"]) == 1
    assert (package / row["history"][0]["mask_path"]).exists()
    assert evaluate_package(package)["metrics"] is None


@pytest.mark.parametrize("changed", ["image", "reference", "prediction"])
def test_changed_artifact_refuses_evaluation(tmp_path, changed):
    package, mask, _ = _fixture(tmp_path)
    _confirm(package, mask)
    row = json.loads((package / "annotations.json").read_text())["samples"][0]
    paths = {"image": row["image_path"], "reference": row["confirmation"]["mask_path"],
             "prediction": row["predictions"]["synthetic_method"]["path"]}
    target = package / paths[changed]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        evaluate_package(package)


def test_confirm_requires_person_policy_source_and_binary_shape(tmp_path):
    package, mask, _ = _fixture(tmp_path)
    for options in ({"reviewer": " "}, {"exclude_food": False}, {"source": "independent"}):
        args = dict(reviewer="SYNTHETIC_TEST_ONLY", source="assisted", exclude_food=True)
        args.update(options)
        with pytest.raises(ValueError):
            confirm_mask(package, "synthetic_01", mask, **args)
    bad = tmp_path / "bad.png"
    cv2.imwrite(str(bad), np.ones((12, 16), np.uint8))
    with pytest.raises(ValueError, match="single-channel"):
        _confirm(package, bad)
    cv2.imwrite(str(bad), np.zeros((2, 3), np.uint8))
    with pytest.raises(ValueError, match="shape"):
        _confirm(package, bad)


def test_independent_export_omits_algorithm_draft_and_blocks_prediction_path(tmp_path):
    package, mask, _ = _fixture(tmp_path, mode="independent")
    row = json.loads((package / "annotations.json").read_text())["samples"][0]
    assert row["draft_source"] is None
    assert not cv2.imread(str(package / row["draft_path"]), 0).any()
    with pytest.raises(ValueError, match="algorithm prediction"):
        _confirm(package, package / row["predictions"]["synthetic_method"]["path"], "independent")


def test_empty_reference_never_returns_zero_relative_error():
    empty = np.zeros((4, 4), bool)
    pred = empty.copy()
    pred[1:3, 1:3] = True
    both = mask_metrics(empty, empty)
    wrong = mask_metrics(pred, empty)
    missing = mask_metrics(empty, pred)
    assert both["dice"] == both["iou"] == 1
    assert both["relative_area_error"] is None
    assert wrong["dice"] == wrong["iou"] == wrong["precision"] == 0
    assert wrong["fp"] == 4 and wrong["relative_area_error"] is None
    assert wrong["signed_area_error_px"] == 4
    assert missing["dice"] == missing["recall"] == 0 and missing["fn"] == 4


def test_existing_package_is_never_overwritten(tmp_path):
    package, _, _ = _fixture(tmp_path)
    before = file_hash(package / "annotations.json")
    with pytest.raises(FileExistsError):
        create_package(tmp_path / "benchmark/samples.json", package)
    assert file_hash(package / "annotations.json") == before


def test_partial_confirmation_reports_null_not_partial_accuracy(tmp_path):
    package, mask, _ = _fixture(tmp_path)
    _confirm(package, mask)
    path = package / "annotations.json"
    data = json.loads(path.read_text())
    other = dict(data["samples"][0], sample_id="synthetic_02", status="saved_draft", confirmation=None)
    data["samples"].append(other)
    path.write_text(json.dumps(data))
    result = evaluate_package(package)
    assert result["metrics"] is None and result["confirmed_count"] == 1
    assert result["pending_samples"] == ["synthetic_02"]


@pytest.mark.parametrize("tamper", [False, True])
def test_export_checks_benchmark_pixel_hash_before_accepting_image(tmp_path, tamper):
    _, _, original = _fixture(tmp_path)
    samples = tmp_path / "benchmark/samples.json"
    data = json.loads(samples.read_text())
    image = cv2.imread(str(original), cv2.IMREAD_COLOR)
    # Independent reproduction of benchmark's declared BGR|shape|dtype contract.
    digest = hashlib.sha256(f"BGR|{image.shape}|{image.dtype}".encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(image)))
    data["samples"][0].update(source_pixel_sha256=digest.hexdigest(), source_mode="raw",
                              source_is_prepared_registered_image=False)
    samples.write_text(json.dumps(data))
    if tamper:
        image[0, 0] = (3, 2, 1)
        cv2.imwrite(str(original), image)
        with pytest.raises(ValueError, match="changed since benchmark"):
            create_package(samples, tmp_path / "checked")
    else:
        package = create_package(samples, tmp_path / "checked")
        row = json.loads((package / "annotations.json").read_text())["samples"][0]
        assert row["source_pixel_sha256"] == digest.hexdigest()
        assert row["source_mode"] == "raw" and row["source_is_prepared_registered_image"] is False
        assert evaluate_package(package)["metrics"] is None
