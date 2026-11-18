"""Headless synthetic checks; never start a window or confirm research labels."""
import json

import cv2
import numpy as np
import pytest

from segmentation_lab.annotations import create_package, evaluate_package, file_hash
from segmentation_lab.mask_editor import HEADER_HEIGHT, MaskEditor, Viewport, load_editor_sample, paint_mask, save_editor_draft


def test_zoom_and_pan_map_back_to_same_full_resolution_pixel():
    viewport = Viewport(2560, 1920, 1200, 720)
    source = (1513., 782.)
    screen = viewport.source_to_screen(*source)
    assert viewport.screen_to_source(*screen) == pytest.approx(source)
    viewport.zoom(3, screen)
    assert viewport.source_to_screen(*source) == pytest.approx(screen)
    assert viewport.screen_to_source(*screen) == pytest.approx(source)
    viewport.pan(70, -55)
    moved = viewport.source_to_screen(*source)
    assert viewport.screen_to_source(*moved) == pytest.approx(source)


def test_letterbox_header_and_off_image_coordinates_do_not_paint():
    viewport = Viewport(200, 100, 600, 600)
    assert viewport.screen_to_source(200, -5) is None
    assert viewport.screen_to_source(20, 20) is None  # letterboxed background
    assert viewport.screen_to_source(601, 300) is None
    mask = np.zeros((100, 200), bool)
    assert not paint_mask(mask, None, 3).any()
    assert not paint_mask(mask, (-1, 40), 3).any()
    assert not paint_mask(mask, (200, 50), 3).any()


def test_brush_clips_at_border_and_never_mutates_input():
    original = np.zeros((20, 30), bool)
    painted = paint_mask(original, (0, 0), 6)
    assert painted.shape == original.shape and painted.dtype == np.bool_
    assert painted[0, 0] and painted[0, 6] and painted[6, 0]
    assert not painted[-1].any() and not painted[:, -1].any()
    assert not original.any()
    erased = paint_mask(painted, (0, 0), 2, erase=True)
    assert not erased[0, 0] and painted[0, 0]
    assert erased[0, 6]


def test_stroke_interpolation_fills_motion_gaps_without_touching_other_rows():
    mask = np.zeros((40, 80), bool)
    result = paint_mask(mask, (65, 20), 2, previous=(10, 20))
    assert result[20, 10:66].all()
    assert not result[:15].any() and not result[26:].any()


def _package(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.full((80, 100, 3), 70, np.uint8)
    image[30:40, 45:55] = (20, 180, 200)
    cv2.imwrite(str(source / "original.png"), image)
    cv2.imwrite(str(source / "prediction.png"), np.zeros(image.shape[:2], np.uint8))
    data = {"dataset_id": "synthetic_editor_only", "evaluation_role": "development_seen", "samples": [{
        "sample_id": "synthetic_01", "image_path": "original.png", "draft_mask_path": "prediction.png",
        "prediction_mask_path": "prediction.png", "prediction_method": "synthetic"}]}
    (source / "samples.json").write_text(json.dumps(data))
    return create_package(source / "samples.json", tmp_path / "package"), image


def test_save_changes_only_draft_and_metadata_not_original_or_predictions(tmp_path):
    package, image = _package(tmp_path)
    loaded = load_editor_sample(package, "synthetic_01")
    row = loaded["row"]
    image_hash = file_hash(package / row["image_path"])
    prediction_hash = file_hash(package / row["predictions"]["synthetic"]["path"])
    edited = paint_mask(loaded["mask"], (50, 35), 5)
    digest = save_editor_draft(package, "synthetic_01", edited, loaded["draft_sha256"])
    current = load_editor_sample(package, "synthetic_01")
    assert current["draft_sha256"] == digest
    assert np.array_equal(current["mask"], edited)
    assert np.array_equal(current["image"], image)
    assert file_hash(package / row["image_path"]) == image_hash
    assert file_hash(package / row["predictions"]["synthetic"]["path"]) == prediction_hash
    assert current["row"]["status"] == "saved_draft" and current["row"]["confirmation"] is None
    assert evaluate_package(package)["metrics"] is None
    assert not list((package / "confirmed_masks").iterdir())
    assert not list((package / "draft_masks").glob(".editor-draft-*"))


def test_outside_draft_edit_refuses_overwrite(tmp_path):
    package, _ = _package(tmp_path)
    loaded = load_editor_sample(package, "synthetic_01")
    first = paint_mask(loaded["mask"], (20, 20), 4)
    save_editor_draft(package, "synthetic_01", first, loaded["draft_sha256"])
    second = paint_mask(loaded["mask"], (70, 60), 4)
    with pytest.raises(RuntimeError, match="changed outside"):
        save_editor_draft(package, "synthetic_01", second, loaded["draft_sha256"])
    assert np.array_equal(load_editor_sample(package, "synthetic_01")["mask"], first)


def test_changed_source_refuses_draft_save(tmp_path):
    package, _ = _package(tmp_path)
    loaded = load_editor_sample(package, "synthetic_01")
    changed = loaded["image"].copy()
    changed[0, 0] = 255
    cv2.imwrite(str(package / loaded["row"]["image_path"]), changed)
    with pytest.raises(ValueError, match="Source image has changed"):
        save_editor_draft(package, "synthetic_01", loaded["mask"], loaded["draft_sha256"])


def test_headless_render_and_missing_mouse_release_do_not_damage_source(tmp_path, monkeypatch):
    package, image = _package(tmp_path)
    def no_window(*args, **kwargs):
        raise AssertionError("No UI may be opened during this test")
    monkeypatch.setattr(cv2, "namedWindow", no_window)
    editor = MaskEditor(package, "synthetic_01", width=400, height=300)
    x, y = editor.viewport.source_to_screen(20, 20)
    editor.mouse(cv2.EVENT_LBUTTONDOWN, round(x), round(y) + HEADER_HEIGHT, cv2.EVENT_FLAG_LBUTTON, None)
    assert editor.mask[20, 20]
    x, y = editor.viewport.source_to_screen(70, 60)
    editor.mouse(cv2.EVENT_MOUSEMOVE, round(x), round(y) + HEADER_HEIGHT, 0, None)
    assert not editor.mask[60, 70] and not editor.painting
    assert editor.render().shape == (400, 400, 3)
    assert np.array_equal(editor.image, image)
