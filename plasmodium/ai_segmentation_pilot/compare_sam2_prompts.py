"""Render a one-frame SAM 2 prompt comparison for visual go/no-go review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from generate_sam2_drafts import _prompt_seed, _select_mask, _weak_mask_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("prompt_comparison.jpg"))
    parser.add_argument("--center", nargs=2, type=int, default=(1280, 920))
    parser.add_argument("--radius", type=int, default=680)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    with (dataset / "dataset_manifest.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        row = list(csv.DictReader(handle))[args.row]
    image = cv2.imread(str(dataset / row["image_path"]), cv2.IMREAD_COLOR)
    center = tuple(args.center)
    seed, dish, box, points, labels = _prompt_seed(image, center, args.radius)

    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(args.checkpoint), device="cuda")
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    modes = {
        "points": dict(point_coords=points, point_labels=labels),
        "box": dict(box=box[None, :]),
        "points_box": dict(point_coords=points, point_labels=labels, box=box[None, :]),
        "weak_mask": dict(
            point_coords=points,
            point_labels=labels,
            box=box[None, :],
            mask_input=_weak_mask_input(seed),
        ),
    }
    panels = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, kwargs in modes.items():
            masks, scores, _ = predictor.predict(multimask_output=True, **kwargs)
            mask = _select_mask(masks, scores, seed, dish, center, args.radius)
            panel = image.copy()
            panel[mask] = (0.4 * panel[mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.putText(panel, f"{name} area={mask.mean():.3f} scores={np.round(scores, 3)}",
                        (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            panels.append(panel)
            print(name, "masks", masks.shape, "scores", scores, "area", int(mask.sum()))

    prompt = image.copy()
    prompt[seed] = (0.35 * prompt[seed] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.rectangle(prompt, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (255, 0, 0), 4)
    for point, label in zip(points, labels):
        cv2.circle(prompt, tuple(point.astype(int)), 10, (0, 255, 0) if label else (0, 0, 255), -1)
    cv2.putText(prompt, f"prompt seed={seed.mean():.3f} box={box.astype(int)}", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    panels.insert(0, prompt)
    scale = 720 / image.shape[1]
    panels = [cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for panel in panels]
    canvas = np.hstack(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", args.output.resolve(), "sample", row["sample_id"])


if __name__ == "__main__":
    main()
