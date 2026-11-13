"""Small OpenCV brush editor for accepting/correcting weak mask labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


class Reviewer:
    def __init__(self, dataset: Path, draft_dir: str, split: str, width: int):
        self.dataset = dataset.resolve()
        with (self.dataset / "dataset_manifest.csv").open("r", newline="", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        if split != "all":
            self.rows = [row for row in self.rows if row["split"] == split]
        self.draft_dir = self.dataset / draft_dir
        self.output_dir = self.dataset / "masks_verified"
        self.output_dir.mkdir(exist_ok=True)
        self.width = width
        self.index = 0
        self.brush = 18
        self.painting = False
        self.erase = False
        self.image = None
        self.mask = None
        self.scale = 1.0
        cv2.namedWindow("mask-review", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("mask-review", self.mouse)

    def load(self):
        row = self.rows[self.index]
        self.image = cv2.imread(str(self.dataset / row["image_path"]), cv2.IMREAD_COLOR)
        verified = self.output_dir / f"{row['sample_id']}.png"
        draft = self.draft_dir / f"{row['sample_id']}.png"
        source = verified if verified.exists() else draft
        loaded = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE) if source.exists() else None
        self.mask = ((loaded >= 128).astype(np.uint8) if loaded is not None
                     else np.zeros(self.image.shape[:2], np.uint8))
        self.scale = self.width / self.image.shape[1]

    def mouse(self, event, x, y, _flags, _param):
        if event in {cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN}:
            self.painting = True
            self.erase = event == cv2.EVENT_RBUTTONDOWN
        elif event in {cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP}:
            self.painting = False
        if self.painting:
            ox, oy = int(round(x / self.scale)), int(round(y / self.scale))
            cv2.circle(self.mask, (ox, oy), self.brush, 0 if self.erase else 1, -1)

    def save(self):
        row = self.rows[self.index]
        cv2.imwrite(str(self.output_dir / f"{row['sample_id']}.png"), self.mask.astype(np.uint8) * 255)

    def display(self):
        row = self.rows[self.index]
        overlay = self.image.copy()
        selected = self.mask > 0
        overlay[selected] = (0.45 * overlay[selected] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        title = (f"{self.index + 1}/{len(self.rows)} {row['sample_id']}  "
                 f"brush={self.brush}px  left=paint right=erase s=save n/p=move +/-=brush q=quit")
        cv2.putText(overlay, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                    (255, 255, 255), 1, cv2.LINE_AA)
        shown = cv2.resize(overlay, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("mask-review", shown)

    def run(self):
        self.load()
        while True:
            self.display()
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                self.save()
            elif key == ord("n") and self.index < len(self.rows) - 1:
                self.save()
                self.index += 1
                self.load()
            elif key == ord("p") and self.index > 0:
                self.save()
                self.index -= 1
                self.load()
            elif key in {ord("+"), ord("=")}:
                self.brush = min(150, self.brush + 3)
            elif key in {ord("-"), ord("_")}:
                self.brush = max(2, self.brush - 3)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--draft-dir", default="masks_weak_v2")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--width", type=int, default=1400)
    args = parser.parse_args()
    Reviewer(args.dataset, args.draft_dir, args.split, args.width).run()


if __name__ == "__main__":
    main()
