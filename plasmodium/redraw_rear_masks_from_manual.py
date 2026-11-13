"""Redraw later rear masks from the first four corrected rear examples."""

from datetime import datetime
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from skimage import morphology

from plasmodium_brightfield_analysis import keep_largest_component
from prepare_draft_masks import _pixel_features


PACKAGE_DIR = Path("results_plasmodium/manual_validation_package")
TRAIN_EXAMPLES = 4
RANDOM_SEED = 42
SAMPLES_PER_CLASS_PER_FRAME = 45000
PREDICT_CHUNK = 250000
CONSERVATIVE_THRESHOLD = 0.70


def _features(source, draft):
    base = _pixel_features(source)
    height, width = draft.shape
    yy, xx = np.mgrid[:height, :width]
    x_norm = (xx / max(width - 1, 1)).astype(np.float32)
    y_norm = (yy / max(height - 1, 1)).astype(np.float32)
    radial = np.sqrt(
        ((xx - 1280.0) / 750.0) ** 2 + ((yy - 960.0) / 750.0) ** 2
    ).astype(np.float32)
    draft_float = draft.astype(np.float32)
    draft_local = cv2.GaussianBlur(draft_float, (0, 0), 2.0)
    return np.dstack(
        [base, x_norm, y_norm, radial, draft_float, draft_local]
    ).astype(np.float32)


def _dish(shape):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return (xx - 1280) ** 2 + (yy - 960) ** 2 <= 750**2


def _sample(mask, count, rng):
    indices = np.flatnonzero(mask.ravel())
    if indices.size <= count:
        return indices
    return rng.choice(indices, size=count, replace=False)


def _fit(examples):
    rng = np.random.default_rng(RANDOM_SEED)
    train_x = []
    train_y = []
    for source, draft, manual in examples:
        features = _features(source, draft).reshape(-1, 14)
        valid = _dish(draft.shape)
        changed = draft != manual
        positive = manual & valid
        negative = ~manual & valid
        positive_idx = np.unique(
            np.concatenate(
                [
                    _sample(positive, SAMPLES_PER_CLASS_PER_FRAME, rng),
                    _sample(changed & positive, SAMPLES_PER_CLASS_PER_FRAME, rng),
                ]
            )
        )
        negative_idx = np.unique(
            np.concatenate(
                [
                    _sample(negative, SAMPLES_PER_CLASS_PER_FRAME, rng),
                    _sample(changed & negative, SAMPLES_PER_CLASS_PER_FRAME, rng),
                ]
            )
        )
        indices = np.concatenate([positive_idx, negative_idx])
        train_x.append(features[indices])
        train_y.append(
            np.concatenate(
                [
                    np.ones(positive_idx.size, dtype=np.uint8),
                    np.zeros(negative_idx.size, dtype=np.uint8),
                ]
            )
        )
    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=140,
        max_leaf_nodes=63,
        min_samples_leaf=35,
        l2_regularization=0.8,
        random_state=RANDOM_SEED,
    )
    model.fit(np.vstack(train_x), np.concatenate(train_y))
    return model


def _predict(model, source, draft, threshold):
    features = _features(source, draft).reshape(-1, 14)
    valid = _dish(draft.shape)
    valid_idx = np.flatnonzero(valid.ravel())
    probability = np.zeros(draft.size, dtype=np.float32)
    for start in range(0, valid_idx.size, PREDICT_CHUNK):
        indices = valid_idx[start : start + PREDICT_CHUNK]
        probability[indices] = model.predict_proba(features[indices])[:, 1]
    probability = probability.reshape(draft.shape)
    mask = (probability >= threshold) & valid
    mask = morphology.remove_small_objects(mask, min_size=12)
    mask = morphology.binary_closing(mask, morphology.disk(1))
    mask = keep_largest_component(mask)
    return mask, probability


def _dice(left, right):
    denominator = int(left.sum() + right.sum())
    return 2.0 * int(np.sum(left & right)) / denominator if denominator else 1.0


def redraw_rear_masks(package_dir=PACKAGE_DIR):
    package_dir = Path(package_dir)
    manifest = pd.read_csv(package_dir / "manual_validation_manifest.csv")
    rear = manifest[manifest["section"] == "rear"].reset_index(drop=True)
    examples = []
    for row in rear.iloc[:TRAIN_EXAMPLES].itertuples(index=False):
        source = cv2.imread(str(row.source_image), cv2.IMREAD_COLOR)
        draft = cv2.imread(
            str(package_dir / "draft_masks" / f"frame_{int(row.frame):06d}_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        manual = cv2.imread(str(row.expected_manual_mask), cv2.IMREAD_GRAYSCALE)
        if source is None or draft is None or manual is None:
            raise FileNotFoundError(f"Missing training input for frame {row.frame}.")
        examples.append((source, draft > 0, manual > 0))

    threshold = CONSERVATIVE_THRESHOLD
    fold_dice = []
    fold_area_error = []
    for holdout in range(len(examples)):
        validation_model = _fit(
            [example for index, example in enumerate(examples) if index != holdout]
        )
        source, draft, manual = examples[holdout]
        predicted, _ = _predict(validation_model, source, draft, threshold)
        fold_dice.append(_dice(predicted, manual))
        fold_area_error.append(
            float((predicted.sum() - manual.sum()) / max(manual.sum(), 1))
        )
    validation_dice = float(np.mean(fold_dice))
    validation_area_error = float(np.mean(fold_area_error))
    print(
        f"Conservative threshold={threshold:.2f}, "
        f"4-fold Dice={validation_dice:.4f}, "
        f"mean area error={validation_area_error:+.4f}"
    )

    model = _fit(examples)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = package_dir / f"manual_masks_backup_rear_{timestamp}"
    review_dir = package_dir / "rear_ai_redraw_review"
    probability_dir = package_dir / "rear_ai_redraw_probabilities"
    backup_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in rear.iloc[TRAIN_EXAMPLES:].itertuples(index=False):
        source = cv2.imread(str(row.source_image), cv2.IMREAD_COLOR)
        draft_path = (
            package_dir / "draft_masks" / f"frame_{int(row.frame):06d}_mask.png"
        )
        draft = cv2.imread(str(draft_path), cv2.IMREAD_GRAYSCALE)
        manual_path = Path(row.expected_manual_mask)
        if source is None or draft is None:
            raise FileNotFoundError(f"Missing redraw input for frame {row.frame}.")
        if manual_path.exists():
            shutil.copy2(manual_path, backup_dir / manual_path.name)

        predicted, probability = _predict(
            model, source, draft > 0, threshold
        )
        cv2.imwrite(str(manual_path), predicted.astype(np.uint8) * 255)
        cv2.imwrite(
            str(probability_dir / f"frame_{int(row.frame):06d}_probability.png"),
            np.clip(probability * 255.0, 0, 255).astype(np.uint8),
        )
        overlay = source.copy()
        selected = predicted > 0
        overlay[selected] = (
            0.58 * overlay[selected]
            + 0.42 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(
            str(review_dir / f"frame_{int(row.frame):06d}_overlay.jpg"),
            overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        draft_bool = draft > 0
        rows.append(
            {
                "frame": int(row.frame),
                "draft_area_px": int(draft_bool.sum()),
                "redraw_area_px": int(predicted.sum()),
                "draft_redraw_dice": _dice(draft_bool, predicted),
                "threshold": threshold,
                "held_out_dice": validation_dice,
                "cross_validated_area_error": validation_area_error,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(
        package_dir / "rear_ai_redraw_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Redrew {len(result)} rear masks. Backup: {backup_dir}")
    return result


if __name__ == "__main__":
    redraw_rear_masks()





