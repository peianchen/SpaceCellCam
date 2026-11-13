"""Generate review-only SAM 2 draft masks from stabilized representative frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _dish_mask(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2


def _prompt_seed(image, center, radius):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dish = _dish_mask(image.shape[:2], center, int(radius * 0.92))
    central = _dish_mask(image.shape[:2], center, int(radius * 0.78))
    seed = (
        (hsv[..., 0] >= 12) & (hsv[..., 0] <= 58)
        & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 25)
        & dish
    )
    seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    labels, count = ndi.label(seed)
    choices = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < 100:
            continue
        yy, xx = np.where(component)
        distance = np.hypot(float(xx.mean()) - center[0], float(yy.mean()) - center[1])
        central_overlap = float((component & central).sum() / area)
        choices.append((central_overlap, -distance, area, label_id))
    if not choices:
        raise RuntimeError("Could not find a central yellow prompt seed")
    seed = labels == max(choices)[3]
    # Include nearby yellow components so the box covers branching beyond the core.
    expanded = cv2.dilate(seed.astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool)
    seed |= ((labels > 0) & expanded)
    yy, xx = np.where(seed)
    pad = max(70, int(round(max(xx.max() - xx.min(), yy.max() - yy.min()) * 0.18)))
    box = np.array(
        [max(0, xx.min() - pad), max(0, yy.min() - pad),
         min(image.shape[1] - 1, xx.max() + pad), min(image.shape[0] - 1, yy.max() + pad)],
        dtype=np.float32,
    )
    # Positive points span the seed; negative points protect the dish rim.
    coords = np.column_stack([xx, yy])
    # Spread positives across the seed instead of taking scan-order samples.
    positive = [coords[len(coords) // 2]]
    for _ in range(min(11, len(coords) - 1)):
        chosen = np.asarray(positive)
        distance_sq = ((coords[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
        positive.append(coords[np.argmax(distance_sq.min(axis=1))])
    positive = np.asarray(positive)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    negative = np.column_stack(
        [center[0] + radius * 0.90 * np.cos(angles), center[1] + radius * 0.90 * np.sin(angles)]
    )
    points = np.vstack([positive, negative]).astype(np.float32)
    labels_prompt = np.r_[np.ones(len(positive)), np.zeros(len(negative))].astype(np.int32)
    return seed, dish, box, points, labels_prompt


def _weak_mask_input(seed):
    """Convert a full-resolution weak mask to SAM's 256x256 logit prompt."""
    low_res = cv2.resize(seed.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
    return np.where(low_res >= 0.25, 8.0, -8.0).astype(np.float32)[None, ...]


def _select_mask(masks, scores, seed, dish, center, radius):
    rim = dish & ~_dish_mask(dish.shape, center, int(radius * 0.82))
    ranked = []
    for index, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool) & dish
        coverage = float((mask & seed).sum() / max(seed.sum(), 1))
        rim_fraction = float((mask & rim).sum() / max(mask.sum(), 1))
        ranked.append((float(scores[index]) + 0.6 * coverage - 0.35 * rim_fraction, index))
    mask = np.asarray(masks[max(ranked)[1]], dtype=bool) & dish
    labels, count = ndi.label(mask)
    touching = np.unique(labels[seed])
    touching = touching[touching > 0]
    if touching.size:
        mask = np.isin(labels, touching)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--row-indices", nargs="*", type=int)
    parser.add_argument("--center", nargs=2, type=int, default=(1280, 920))
    parser.add_argument("--radius", type=int, default=680)
    parser.add_argument(
        "--prompt-mode",
        choices=["points", "box", "points_box", "weak_mask"],
        default="weak_mask",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 2 pilot requires CUDA; CPU fallback is intentionally disabled")
    device = torch.device("cuda")
    model = build_sam2(args.model_config, str(args.checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    dataset = args.dataset.resolve()
    with (dataset / "dataset_manifest.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.row_indices:
        rows = [rows[index] for index in args.row_indices]
    if args.split != "all":
        rows = [row for row in rows if row["split"] == args.split]
    if args.limit > 0:
        rows = rows[: args.limit]
    overlay_dir = dataset / "draft_overlays"
    overlay_dir.mkdir(exist_ok=True)
    draft_dir = dataset / "masks_draft"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for number, row in enumerate(rows, 1):
            image_path = dataset / row["image_path"]
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            seed, dish, box, points, point_labels = _prompt_seed(
                image_bgr, tuple(args.center), args.radius
            )
            predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            kwargs = {"multimask_output": True}
            if args.prompt_mode in {"points", "points_box", "weak_mask"}:
                kwargs.update(point_coords=points, point_labels=point_labels)
            if args.prompt_mode in {"box", "points_box", "weak_mask"}:
                kwargs["box"] = box[None, :]
            if args.prompt_mode == "weak_mask":
                kwargs["mask_input"] = _weak_mask_input(seed)
            masks, scores, _ = predictor.predict(**kwargs)
            mask = _select_mask(masks, scores, seed, dish, tuple(args.center), args.radius)
            mask_path = draft_dir / f"{row['sample_id']}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            overlay = image_bgr.copy()
            overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.imwrite(str(overlay_dir / f"{row['sample_id']}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"{number}/{len(rows)} {row['sample_id']}", flush=True)


if __name__ == "__main__":
    main()
