"""Local full-resolution draft editor. Saving never confirms ground truth.

Run explicitly with --package PATH --sample ID to open the OpenCV window.
Importing this module and running its unit tests never opens a window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import annotations


HEADER_HEIGHT = 100
WINDOW_NAME = "Plasmodium mask editor - DRAFT ONLY"


@dataclass
class Viewport:
    """Affine pixel-center mapping between source coordinates and a fixed canvas."""
    image_width: int
    image_height: int
    width: int = 1200
    height: int = 720
    scale: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self):
        if min(self.image_width, self.image_height, self.width, self.height) <= 0:
            raise ValueError("Image and canvas dimensions must be positive")
        self.fit()

    @property
    def fit_scale(self) -> float:
        return min(self.width / self.image_width, self.height / self.image_height)

    def fit(self) -> None:
        self.scale = self.fit_scale
        self.center_x = (self.image_width - 1) / 2
        self.center_y = (self.image_height - 1) / 2

    def source_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.center_x) * self.scale + (self.width - 1) / 2,
                (y - self.center_y) * self.scale + (self.height - 1) / 2)

    def screen_to_source(self, x: float, y: float) -> tuple[float, float] | None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        sx = (x - (self.width - 1) / 2) / self.scale + self.center_x
        sy = (y - (self.height - 1) / 2) / self.scale + self.center_y
        # Image support extends half a source pixel beyond its outer pixel centers.
        if not (-.5 <= sx < self.image_width - .5 and -.5 <= sy < self.image_height - .5):
            return None
        return (min(self.image_width - 1., max(0., sx)), min(self.image_height - 1., max(0., sy)))

    def zoom(self, factor: float, anchor: tuple[float, float] | None = None) -> None:
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Zoom factor must be positive and finite")
        ax, ay = anchor or ((self.width - 1) / 2, (self.height - 1) / 2)
        sx = (ax - (self.width - 1) / 2) / self.scale + self.center_x
        sy = (ay - (self.height - 1) / 2) / self.scale + self.center_y
        self.scale = min(max(self.scale * factor, self.fit_scale / 4), max(16., self.fit_scale))
        self.center_x = sx - (ax - (self.width - 1) / 2) / self.scale
        self.center_y = sy - (ay - (self.height - 1) / 2) / self.scale

    def pan(self, dx_screen: float, dy_screen: float) -> None:
        self.center_x = float(np.clip(self.center_x + dx_screen / self.scale, 0, self.image_width - 1))
        self.center_y = float(np.clip(self.center_y + dy_screen / self.scale, 0, self.image_height - 1))

    def affine(self) -> np.ndarray:
        ox, oy = self.source_to_screen(0, 0)
        return np.float32([[self.scale, 0, ox], [0, self.scale, oy]])


def paint_mask(mask: np.ndarray, point: tuple[float, float] | None, radius: int,
               erase: bool = False, previous: tuple[float, float] | None = None) -> np.ndarray:
    """Return a new binary mask; off-image centers are ignored and strokes clip safely."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Brush input must be a two-dimensional boolean mask")
    if not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError("Brush radius must be a positive integer in source pixels")
    edited = mask.astype(np.uint8)
    height, width = mask.shape
    def valid(value):
        return value is not None and len(value) == 2 and all(math.isfinite(float(x)) for x in value) and 0 <= value[0] < width and 0 <= value[1] < height
    if not valid(point):
        return edited.astype(bool)
    target = (int(np.clip(round(point[0]), 0, width - 1)), int(np.clip(round(point[1]), 0, height - 1)))
    color = 0 if erase else 1
    if valid(previous):
        start = (int(np.clip(round(previous[0]), 0, width - 1)), int(np.clip(round(previous[1]), 0, height - 1)))
        cv2.line(edited, start, target, color, 2 * int(radius) + 1, cv2.LINE_8)
    cv2.circle(edited, target, int(radius), color, -1, cv2.LINE_8)
    return edited.astype(bool)


def _inside(root: Path, relative: str, expected_folder: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to((root / expected_folder).resolve()):
        raise ValueError(f"Package {expected_folder} path escapes its expected directory")
    return path


def load_editor_sample(package_path: Path, sample_id: str) -> dict:
    root = Path(package_path).resolve()
    data = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    matches = [row for row in data["samples"] if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one package sample: {sample_id}")
    row = matches[0]
    image_path = _inside(root, row["image_path"], "images")
    draft_path = _inside(root, row["draft_path"], "draft_masks")
    if annotations.file_hash(image_path) != row["image_sha256"]:
        raise ValueError("Source image has changed; refusing to edit mismatched annotations")
    image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != tuple(row["shape"]):
        raise ValueError("Source image cannot be decoded or does not match package coordinates")
    mask = annotations.read_mask(draft_path, row["shape"])
    return {"root": root, "row": row, "image": image, "mask": mask,
            "draft_path": draft_path, "draft_sha256": annotations.file_hash(draft_path)}


def save_editor_draft(package_path: Path, sample_id: str, mask: np.ndarray,
                      expected_draft_sha256: str) -> str:
    """Use the public draft API; preserve originals, predictions and confirmed files."""
    current = load_editor_sample(package_path, sample_id)
    if current["draft_sha256"] != expected_draft_sha256:
        raise RuntimeError("Draft changed outside this editor; reload it before saving to avoid overwriting another edit")
    if mask.shape != current["mask"].shape or mask.dtype != np.bool_:
        raise ValueError("Edited mask must be full-resolution boolean data in source coordinates")
    descriptor, name = tempfile.mkstemp(prefix=".editor-draft-", suffix=".png", dir=str(current["draft_path"].parent))
    os.close(descriptor)
    staged = Path(name)
    try:
        ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if not ok:
            raise OSError("Could not encode the edited draft")
        encoded.tofile(str(staged))
        # This resets status to saved_draft, clears confirmation, and preserves history.
        annotations.save_draft(current["root"], sample_id, staged)
    finally:
        staged.unlink(missing_ok=True)
    return annotations.file_hash(current["draft_path"])


class MaskEditor:
    def __init__(self, package: Path, sample_id: str, width: int = 1200, height: int = 720):
        loaded = load_editor_sample(package, sample_id)
        self.package, self.sample_id = loaded["root"], sample_id
        self.image, self.mask = loaded["image"], loaded["mask"]
        self.expected_draft_sha256 = loaded["draft_sha256"]
        self.viewport = Viewport(self.image.shape[1], self.image.shape[0], width, height)
        self.radius, self.mode = 5, "overlay"
        self.dirty, self.painting, self.erasing = False, False, False
        self.previous = None
        self.cursor = None
        self.undo: list[np.ndarray] = []
        self.message = "DRAFT ONLY: S saves a draft; human confirmation is a separate command."

    def mouse(self, event, x, y, flags, _param):
        canvas_y = y - HEADER_HEIGHT
        self.cursor = (x, canvas_y)
        point = self.viewport.screen_to_source(x, canvas_y)
        if event == cv2.EVENT_MOUSEWHEEL:
            self.painting, self.previous = False, None
            delta = (flags >> 16) & 0xFFFF
            delta = delta - 65536 if delta >= 32768 else delta
            if point is not None:
                self.viewport.zoom(1.25 if delta > 0 else .8, (x, canvas_y))
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and point is not None:
            if self.mode == "original":
                self.message = "Original view is read-only; press V for overlay or B for binary before painting."
                return
            self.undo.append(self.mask.copy())
            self.undo = self.undo[-5:]
            self.painting, self.erasing = True, event == cv2.EVENT_RBUTTONDOWN
            self.previous = None
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.painting, self.previous = False, None
            return
        if self.painting and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            held_flag = cv2.EVENT_FLAG_RBUTTON if self.erasing else cv2.EVENT_FLAG_LBUTTON
            if event == cv2.EVENT_MOUSEMOVE and not (flags & held_flag):
                # A release outside the window may not deliver BUTTONUP.
                self.painting, self.previous = False, None
                return
            if point is None:
                self.previous = None
                return
            self.mask = paint_mask(self.mask, point, self.radius, self.erasing, self.previous)
            self.previous, self.dirty = point, True

    def render(self) -> np.ndarray:
        vp = self.viewport
        matrix = vp.affine()
        base = cv2.warpAffine(self.image, matrix, (vp.width, vp.height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(24, 24, 24))
        visible = cv2.warpAffine(self.mask.astype(np.uint8), matrix, (vp.width, vp.height), flags=cv2.INTER_NEAREST).astype(bool)
        if self.mode == "binary":
            body = cv2.cvtColor(visible.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        elif self.mode == "original":
            body = base
        else:
            body = base.copy()
            body[visible] = np.rint(.78 * body[visible] + .22 * np.asarray((45, 255, 60))).astype(np.uint8)
            edge = visible & ~cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            body[edge] = (45, 255, 60)
        if self.cursor and vp.screen_to_source(*self.cursor) is not None and self.mode != "original":
            cv2.circle(body, (int(self.cursor[0]), int(self.cursor[1])), max(1, round(self.radius * vp.scale)),
                       (255, 170, 50) if self.erasing else (255, 255, 255), 1, cv2.LINE_AA)
        header = np.full((HEADER_HEIGHT, vp.width, 3), (42, 37, 31), np.uint8)
        lines = [f"DRAFT - NOT CONFIRMED | {self.sample_id} | {'UNSAVED' if self.dirty else 'saved draft'}",
                 f"Left:add Right:erase | brush {self.radius}px [/] | zoom {vp.scale:.2f}x +/- wheel | pan IJKL/arrows | 0:fit",
                 "V:overlay O:original(read-only) B:binary | Z:undo | S:save draft | Esc:exit (does not save)",
                 self.message]
        for index, line in enumerate(lines):
            cv2.putText(header, line, (10, 21 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (230, 240, 250), 1, cv2.LINE_AA)
        return np.vstack([header, body])

    def key(self, key: int) -> bool:
        if key != -1:
            self.painting, self.previous = False, None
        if key == 27:
            if self.dirty:
                print("Exited without saving current edits. Previously saved draft is preserved.", flush=True)
            return False
        if key in (ord("s"), ord("S")):
            try:
                self.expected_draft_sha256 = save_editor_draft(self.package, self.sample_id, self.mask, self.expected_draft_sha256)
                self.dirty = False
                self.message = "Saved draft only. No ground truth confirmation or accuracy calculation."
                print(self.message, flush=True)
            except Exception as exc:
                self.message = f"SAVE FAILED: {type(exc).__name__}; inspect terminal. Edits remain in memory."
                print(f"Save failed: {exc}", file=sys.stderr, flush=True)
        elif key in (ord("+"), ord("=")):
            anchor = self.cursor if self.cursor and self.viewport.screen_to_source(*self.cursor) is not None else None
            self.viewport.zoom(1.25, anchor)
        elif key in (ord("-"), ord("_")):
            anchor = self.cursor if self.cursor and self.viewport.screen_to_source(*self.cursor) is not None else None
            self.viewport.zoom(.8, anchor)
        elif key == ord("0"):
            self.viewport.fit()
        elif key == ord("["):
            self.radius = max(1, self.radius - max(1, round(self.radius * .2)))
        elif key == ord("]"):
            self.radius = min(200, self.radius + max(1, round(self.radius * .2)))
        elif key in (ord("o"), ord("O"), ord("b"), ord("B"), ord("v"), ord("V")):
            self.mode = {"o": "original", "b": "binary", "v": "overlay"}[chr(key).lower()]
            self.painting, self.previous = False, None
        elif key in (ord("z"), ord("Z")) and self.undo:
            self.mask = self.undo.pop()
            self.dirty = True
        else:
            motions = {ord("j"): (-1, 0), ord("l"): (1, 0), ord("i"): (0, -1), ord("k"): (0, 1),
                       2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1),
                       65361: (-1, 0), 65363: (1, 0), 65362: (0, -1), 65364: (0, 1)}
            if key in motions:
                dx, dy = motions[key]
                self.viewport.pan(dx * self.viewport.width * .15, dy * self.viewport.height * .15)
        return True

    def run(self) -> None:
        # A fixed-size image window avoids OS window resizing invalidating mouse mapping.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse)
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.render())
                if not self.key(cv2.waitKeyEx(20)):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass  # The user may already have closed this window using its title bar.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--width", type=int, default=1200, help="Fixed image viewport width")
    parser.add_argument("--height", type=int, default=720, help="Image viewport height, excluding 100px status header")
    args = parser.parse_args(argv)
    if not (400 <= args.width <= 2400 and 300 <= args.height <= 1800):
        parser.error("Use width 400..2400 and height 300..1800 for a usable fixed viewport")
    try:
        MaskEditor(args.package, args.sample, args.width, args.height).run()
        return 0
    except Exception as exc:
        print(f"Mask editor failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
