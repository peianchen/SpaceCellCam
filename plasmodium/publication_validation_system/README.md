# Plasmodium Publication Validation System

This directory is independent from the exploratory validation clips. It is the
pre-specified workflow for validating a frozen segmentation method against
blinded human annotations.

## Scientific boundary

- `development` videos are for parameter tuning only and cannot establish final accuracy.
- `validation` videos may be used for one final design check before freezing.
- `test` videos must be independent cultures/videos and remain untouched until the algorithm is frozen.
- Frames are nested within videos. Reporting treats the video/biological replicate as the independent unit.
- Adjacent-frame consistency is a stability result, not a ground-truth accuracy result.
- Do not show algorithm overlays or masks to annotators before annotation.
- If parameters are changed after examining test results, that test set becomes development data and a new independent test set is required.

## Files

- `config.json`: pre-specified sampling, locked parameters, metrics, and acceptance criteria.
- `videos_template.csv`: one row per independent video/culture.
- `prepare_dataset.py`: deterministic front/middle/rear consecutive-window sampling and blinding.
- `freeze_algorithm.py`: records source and parameter hashes after development is complete.
- `generate_predictions.py`: produces strict and temporal masks only after the lock is valid.
- `evaluate_validation.py`: inter/intra-rater, segmentation, topology, temporal, bootstrap CI, plots, and failure gallery.
- `synthetic_smoke_test.py`: non-biological end-to-end software test.

## Recommended sampling

Each video contributes 30 frames: two independent five-frame windows from each
of front, middle, and rear. Consecutive frames support temporal change checks.
At least two annotators draw masks from raw full-resolution images. Twenty
percent are re-presented under new blind IDs for intra-rater repeatability. The
adjudicated mask is created only after both primary annotations are complete.

## Workflow

1. Copy `videos_template.csv`, add independent videos, and assign their role before looking at sampled frames.
2. Build a blinded package:

```powershell
python prepare_dataset.py --videos videos.csv --role validation
```

3. Annotators fill `annotators/<name>/masks`. Repeat masks use the repeat blind IDs in `annotators/<name>_repeat/masks`.
4. Resolve disagreements into `adjudicated_masks` while preserving both original annotations.
5. Complete development, then freeze source and parameters:

```powershell
python freeze_algorithm.py --confirm-development-complete
```

6. Prepare the untouched test set and generate predictions:

```powershell
python prepare_dataset.py --videos videos.csv --role test
python generate_predictions.py --role test
python evaluate_validation.py --role test
```

The normal prediction command refuses to run if the source or parameter hash no
longer matches the lock. `--allow-development` exists only for software testing
and marks its outputs as ineligible for publication claims.

## Primary outputs

- `frame_metrics.csv`: Dice, IoU, precision, recall, area error, boundary F1, HD95, clDice, skeleton and artifact metrics.
- `temporal_metrics.csv`: expansion/retraction agreement and front-motion error for consecutive annotated frames.
- `inter_rater_metrics.csv` and `intra_rater_metrics.csv`.
- `summary_by_method.csv`: video-clustered bootstrap confidence intervals.
- `acceptance_summary.csv`: pre-specified pass/fail checks.
- `validation_report.md`, Bland-Altman plots, and a worst-case failure gallery.

Passing the thresholds supports the measured endpoints only. Area-mask success
does not by itself validate branchpoints, endpoints, vein metrics, or biological
interpretation; those require their own agreement results.

Before annotation, both annotators must read and sign off on
`ANNOTATION_PROTOCOL.md`. The default acceptance gate also requires at least
three independent test biological replicates; a larger sample may be required
by the study design or power calculation.
