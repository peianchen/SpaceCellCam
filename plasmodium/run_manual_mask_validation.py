"""Annotate 30 ground-truth masks, then calculate segmentation accuracy."""

from pathlib import Path

from prepare_draft_masks import prepare_draft_masks

from plasmodium_brightfield_analysis import (
    annotate_manual_validation_package,
    validate_manual_validation_package,
)


PACKAGE_DIR = Path("results_plasmodium/manual_validation_package")
ANNOTATE = True
REGENERATE_AI_DRAFTS = False
START_IMAGE = 25


if __name__ == "__main__":
    if ANNOTATE:
        if REGENERATE_AI_DRAFTS:
            prepare_draft_masks(PACKAGE_DIR)
        annotate_manual_validation_package(
            PACKAGE_DIR,
            display_scale=0.5,
            brush_radius=2,
            overlay_alpha=0.23,
            start_image=START_IMAGE,
        )
    else:
        validate_manual_validation_package(PACKAGE_DIR)