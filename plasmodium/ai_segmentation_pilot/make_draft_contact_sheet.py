"""Build a contact sheet from generated draft overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/draft_overlays"))
    parser.add_argument("--output", type=Path, default=Path("dataset/previews/sam2_drafts.jpg"))
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.jpg"))
    paths = paths[::max(1, args.stride)]
    if args.max_items > 0:
        paths = paths[:args.max_items]
    if not paths:
        raise RuntimeError(f"No draft overlays found in {args.input}")
    tiles = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        scale = 560 / image.shape[1]
        tile = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, tile.shape[0] - 34), (tile.shape[1], tile.shape[0]), (0, 0, 0), -1)
        cv2.putText(tile, path.stem[:48], (8, tile.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % args.columns:
        tiles.append(blank.copy())
    rows = [np.hstack(tiles[index:index + args.columns])
            for index in range(0, len(tiles), args.columns)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(args.output.resolve())


if __name__ == "__main__":
    main()
