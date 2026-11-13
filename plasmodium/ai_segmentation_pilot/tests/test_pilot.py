import cv2
import numpy as np

from plasmodium_ai.dataset import _split_for_group_index
from plasmodium_ai.registration import estimate_registration, warp_to_reference


def test_video_split_is_whole_video_and_deterministic():
    assert [_split_for_group_index(i, 4) for i in range(4)] == ["train", "train", "val", "test"]


def test_registration_recovers_translation_and_rotation():
    size = 256
    center = (128, 128)
    radius = 82
    rng = np.random.default_rng(42)
    yy, xx = np.ogrid[:size, :size]
    distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    image = np.zeros((size, size), np.uint8)
    ring = (distance > radius * 0.72) & (distance < radius * 1.18)
    image[ring] = rng.integers(30, 230, size=int(ring.sum()), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    reference = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    source_matrix = cv2.getRotationMatrix2D(center, 0.7, 1.0)
    source_matrix[:, 2] += (5.0, -4.0)
    moving = cv2.warpAffine(reference, source_matrix, (size, size), borderMode=cv2.BORDER_REFLECT_101)
    result = estimate_registration(
        reference, moving, center, radius,
        max_dimension=256, max_translation_px=20, max_rotation_deg=2,
    )
    aligned = warp_to_reference(moving, result)
    before = np.mean(np.abs(reference.astype(float) - moving.astype(float)))
    after = np.mean(np.abs(reference.astype(float) - aligned.astype(float)))
    assert result.applied
    assert after < before * 0.45

