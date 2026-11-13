"""Create editable AI draft masks for manual validation."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from skimage import filters, morphology

from plasmodium_brightfield_analysis import keep_largest_component


PACKAGE_DIR = Path("results_plasmodium/manual_validation_package")
RANDOM_SEED = 42
TRAIN_SAMPLES_PER_CLASS = 30000
PREDICT_CHUNK_PIXELS = 250000


def _pixel_features(source):
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hue_angle = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
    gray_mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sat_mean = cv2.GaussianBlur(hsv[:, :, 1], (0, 0), 2.0)
    return np.stack(
        [
            np.sin(hue_angle),
            np.cos(hue_angle),
            hsv[:, :, 1] / 255.0,
            hsv[:, :, 2] / 255.0,
            lab[:, :, 1] / 255.0,
            lab[:, :, 2] / 255.0,
            gray / 255.0,
            gray_mean / 255.0,
            sat_mean / 255.0,
        ],
        axis=-1,
    ).astype(np.float32)


def _sample_indices(mask, count, rng):
    indices = np.flatnonzero(mask.ravel())
    if indices.size == 0:
        return indices
    if indices.size <= count:
        return indices
    return rng.choice(indices, size=int(count), replace=False)


def _ai_redraw_mask(
    source,
    auto_mask,
    section,
    dish_center=(1280, 960),
    dish_radius=750,
):
    """Train a per-image ML classifier and redraw the entire body mask."""
    rng = np.random.default_rng(RANDOM_SEED)
    features = _pixel_features(source)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    height, width = auto_mask.shape
    yy, xx = np.ogrid[:height, :width]
    dish = (
        (xx - int(dish_center[0])) ** 2
        + (yy - int(dish_center[1])) ** 2
        <= int(dish_radius) ** 2
    )

    eroded_auto = cv2.erode(
        auto_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    confident_yellow = (
        (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 52)
        & (hsv[:, :, 1] >= 65)
        & (lab[:, :, 2] >= 142)
        & dish
    )
    positive_seed = eroded_auto & confident_yellow
    if positive_seed.sum() < 5000:
        positive_seed = eroded_auto

    far_from_auto = ~cv2.dilate(
        auto_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    ).astype(bool)
    confident_gray = (
        ((hsv[:, :, 1] <= 38) | (lab[:, :, 2] <= 136))
        & (hsv[:, :, 2] >= 45)
        & dish
        & far_from_auto
    )
    negative_seed = confident_gray

    positive_idx = _sample_indices(
        positive_seed, TRAIN_SAMPLES_PER_CLASS, rng
    )
    negative_idx = _sample_indices(
        negative_seed, TRAIN_SAMPLES_PER_CLASS, rng
    )
    if positive_idx.size < 1000 or negative_idx.size < 1000:
        raise ValueError("Insufficient AI training seeds for draft mask.")

    flat_features = features.reshape(-1, features.shape[-1])
    train_idx = np.concatenate([positive_idx, negative_idx])
    train_x = flat_features[train_idx]
    train_y = np.concatenate(
        [
            np.ones(positive_idx.size, dtype=np.uint8),
            np.zeros(negative_idx.size, dtype=np.uint8),
        ]
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=90,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.5,
        random_state=RANDOM_SEED,
    )
    model.fit(train_x, train_y)

    dish_idx = np.flatnonzero(dish.ravel())
    probabilities = np.zeros(height * width, dtype=np.float32)
    for start in range(0, dish_idx.size, PREDICT_CHUNK_PIXELS):
        chunk_idx = dish_idx[start : start + PREDICT_CHUNK_PIXELS]
        probabilities[chunk_idx] = model.predict_proba(
            flat_features[chunk_idx]
        )[:, 1]
    probability_map = probabilities.reshape(height, width)

    # This is a fresh full-image classification. The original mask is not
    # dilated or propagated into the result.
    draft = (probability_map >= 0.52) & dish
    if str(section) == "rear":
        yellow_channel = lab[:, :, 2].astype(np.float32) / 255.0
        x0 = max(0, int(dish_center[0] - dish_radius))
        x1 = min(width, int(dish_center[0] + dish_radius + 1))
        y0 = max(0, int(dish_center[1] - dish_radius))
        y1 = min(height, int(dish_center[1] + dish_radius + 1))
        ridge_response = np.zeros((height, width), dtype=np.float32)
        ridge_crop = filters.frangi(
            yellow_channel[y0:y1, x0:x1],
            sigmas=range(1, 5),
            black_ridges=False,
        )
        ridge_response[y0:y1, x0:x1] = ridge_crop
        positive_ridge = ridge_response[(ridge_response > 0) & dish]
        if positive_ridge.size > 0:
            ridge_threshold = float(np.percentile(positive_ridge, 80))
            line_network = (
                (ridge_response >= ridge_threshold)
                & (lab[:, :, 2] >= 137)
                & dish
            )
            line_network = morphology.remove_small_objects(
                line_network,
                min_size=8,
            )
            draft |= line_network
    draft = morphology.remove_small_objects(draft, min_size=20)
    draft = morphology.binary_closing(draft, morphology.disk(1))
    draft = keep_largest_component(draft)
    return draft, probability_map


def prepare_draft_masks(package_dir=PACKAGE_DIR):
    package_dir = Path(package_dir)
    manifest = pd.read_csv(package_dir / "manual_validation_manifest.csv")
    draft_dir = package_dir / "draft_masks"
    probability_dir = package_dir / "draft_probabilities"
    draft_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in manifest.itertuples(index=False):
        manual_path = Path(item.expected_manual_mask)
        source = cv2.imread(str(item.source_image), cv2.IMREAD_COLOR)
        auto = cv2.imread(str(item.auto_mask), cv2.IMREAD_GRAYSCALE)
        if source is None or auto is None:
            raise ValueError(f"Could not read frame {item.frame} draft inputs.")
        auto_mask = auto > 0

        draft, probability_map = _ai_redraw_mask(
            source,
            auto_mask,
            section=item.section,
        )
        source_kind = "ai_pixel_classifier_full_redraw"

        stem = f"frame_{int(item.frame):06d}"
        draft_path = draft_dir / f"{stem}_mask.png"
        probability_path = probability_dir / f"{stem}_probability.png"
        cv2.imwrite(str(draft_path), draft.astype(np.uint8) * 255)
        cv2.imwrite(
            str(probability_path),
            np.clip(probability_map * 255.0, 0, 255).astype(np.uint8),
        )
        intersection = int(np.sum(draft & auto_mask))
        union = int(np.sum(draft | auto_mask))
        rows.append(
            {
                "section": item.section,
                "frame": int(item.frame),
                "source_kind": source_kind,
                "auto_area_px": int(auto_mask.sum()),
                "draft_area_px": int(draft.sum()),
                "draft_area_change_fraction": float(
                    (draft.sum() - auto_mask.sum()) / max(auto_mask.sum(), 1)
                ),
                "auto_draft_iou": float(intersection / union) if union else 1.0,
                "draft_mask": str(draft_path),
                "probability_map": str(probability_path),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        package_dir / "draft_masks_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved {len(result)} editable AI draft masks to {draft_dir}")
    return result


if __name__ == "__main__":
    prepare_draft_masks()