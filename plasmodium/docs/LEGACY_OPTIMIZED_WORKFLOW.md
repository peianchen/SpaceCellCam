> 舊 optimized 單檔操作紀錄；不是目前建議的新開發入口。現行流程見 [CURRENT_WORKFLOW.md](CURRENT_WORKFLOW.md)。以下 official 字樣屬歷史原文。

# Plasmodium bright-field analysis official workflow

## Official script

Use this file as the current official entrypoint:

```text
plasmodium_brightfield_analysis_optimized.py
```

This file is the formal candidate because it contains the latest integrated pipeline:

- HSV/body segmentation with temporal recovery support
- full formal morphology and dynamics analysis
- overlay video output
- early/middle/late ROI contraction analysis
- tracker-like auxiliary motion analysis
- front/middle/rear validation clips
- multicore batch processing controls
- clear console run configuration

Older scripts should be treated as development snapshots unless you are comparing history:

- `plasmodium_brightfield_analysis.py`
- `plasmodium_brightfield_analysis_multicore.py`
- `plasmodium_brightfield_analysis_complete.py`
- `plasmodium_sample_*.py`
- `*_temporal_*.py`, `*_labclahe.py`, `*_hsv_only_*.py`

## What to edit before running

Open `plasmodium_brightfield_analysis_optimized.py` and edit only this main block unless you are developing the algorithm:

```python
if __name__ == "__main__":
    INPUT_MODE = "video"
    INPUT_PATH = r"...your_video.mp4"
    OUTPUT_DIR = r"results_plasmodium"
    RUN_MODE = "formal_only"
```

The current official default is:

```python
RUN_MODE = "formal_only"
```

That means Spyder Run will analyze the full video and export formal CSV, plots, debug examples, and `results_plasmodium/plasmodium_overlay.mp4`.

## Recommended run modes

### 1. Full formal analysis only

Use this for real data output:

```python
RUN_MODE = "formal_only"
```

Outputs include:

- formal metrics CSV
- summary plots
- debug examples
- `plasmodium_overlay.mp4`
- contraction ROI scalar columns if ROIs are available

### 2. Formal analysis plus front/middle/rear validation clips

Use this when you want formal data and visual QA in the same run:

```python
RUN_MODE = "formal_with_validation"
```

This takes longer than `formal_only`.

### 3. Front/middle/rear validation plus formal-style data plots

Use this when tuning or checking segmentation quality without running the whole-video formal analysis:

```python
RUN_MODE = "validation_all"
```

This uses one formal section-analysis pass per front/middle/rear window. The same pass writes validation clips/stills and formal-style data plots, so the video is not segmented twice for validation_all. The `validation_clips/<section>/<section>_color_mask_overlay.mp4` videos use the older lightweight validation visual style for easier manual checking, while the CSV and summary plots still come from the formal temporal analysis path.

Outputs:

```text
results_plasmodium/validation_clips/
results_plasmodium/validation_analysis/front/front_formal_metrics.csv
results_plasmodium/validation_analysis/front/front_plasmodium_summary.png
results_plasmodium/validation_analysis/middle/middle_formal_metrics.csv
results_plasmodium/validation_analysis/middle/middle_plasmodium_summary.png
results_plasmodium/validation_analysis/rear/rear_formal_metrics.csv
results_plasmodium/validation_analysis/rear/rear_plasmodium_summary.png
results_plasmodium/validation_analysis/validation_analysis_summary.csv
```

`VALIDATION_ANALYZE_SECTION_DATA` is automatically `True` only for `validation_all`, so single-section modes remain faster by default. For efficiency, validation_all now uses the formal temporal section pass to create both `validation_clips/` and `validation_analysis/` outputs. Each section is processed sequentially, but each section pass uses `VALIDATION_WORKERS_PER_SECTION = CPU_WORKERS` internally for batched temporal preparation. This avoids unstable nested multiprocessing while still using the configured CPU workers.

### 4. Rear-only validation

Use this when the late/rear segment has pale branches or unstable segmentation:

```python
RUN_MODE = "validation_rear"
```

### 5. ROI contraction only

Use this when you want to redraw or test contraction ROIs without running the full formal pipeline:

```python
RUN_MODE = "contraction_roi"
```

Recommended ROI behavior:

```python
USE_ROI_SELECTION_UI = True
REUSE_SAVED_CONTRACTION_ROIS = False
```

Set `REUSE_SAVED_CONTRACTION_ROIS = True` only when you want to reuse the previous ROI definitions.

### 6. Complete development run

Use this only when you intentionally want everything:

```python
RUN_MODE = "complete"
```

This runs ROI contraction/tracker sampling, full formal analysis, overlay video, and validation clips. It is the slowest mode.

## Edge-frame handling

The official workflow does not use the first and last unstable frames as formal data points. `TEMPORAL_SKIP_EDGE_FRAMES = 10` is applied to formal video outputs and validation windows so temporal recovery and frame-difference metrics do not depend on incomplete first/last-frame context. The video reader may still look at nearby frames as temporal halo, but CSV rows, plots, and validation section windows start after the edge skip and stop before the final edge skip.
## Multicore settings

The official default is:

```python
ENABLE_CPU_PARALLEL = True
CPU_WORKERS = max(8, int(os.cpu_count() or 8))
LOW_MEMORY_VIDEO_STREAM = True
PARALLEL_BATCH_FRAMES = 32
SAMPLE_PARALLEL_STRATEGY = "hybrid"
```

Meaning:

- use at least 8 workers if the machine has them
- stream video in low-memory batches
- split ROI contraction work across CPU cores
- let `validation_all` use `CPU_WORKERS` inside each front/middle/rear formal validation pass
- keep formal output resolution at original analysis scale

## Current formal segmentation policy

The official script currently uses:

```python
SEGMENTATION_PROFILE = "hsv_circle"
USE_TEMPORAL_BODY_RECOVERY = True
FORMAL_USE_TEMPORAL_BODY_RECOVERY = True when formal or ROI mode runs
```

Interpretation:

- strict HSV still defines the main body seed
- temporal recovery helps pale branches
- output should still be visually checked using validation clips
- paper-grade accuracy still requires manual-mask validation with Dice/IoU/Precision/Recall

## Manual ground-truth validation modes

Use these modes to measure real segmentation accuracy against human masks. This is the next required step before claiming paper-grade accuracy.

### 1. Export images for manual masks

```python
RUN_MODE = "manual_validation_export"
```

This creates:

```text
results_plasmodium/manual_validation_package/source_images/
results_plasmodium/manual_validation_package/auto_masks/
results_plasmodium/manual_validation_package/overlays/
results_plasmodium/manual_validation_package/manual_masks/
results_plasmodium/manual_validation_package/manual_validation_manifest.csv
```

Default sampling:

```python
MANUAL_VALIDATION_SAMPLES_PER_SECTION = 10
MANUAL_VALIDATION_SECTION_FRAMES = 300
```

That means 10 frames from front, 10 from middle, and 10 from rear. The auto masks and overlays are for checking, but the final manual mask should represent the human ground truth.

### 2. Draw or correct masks

Option A: use the built-in editor:

```python
RUN_MODE = "manual_validation_annotate"
```

Option B: edit the mask PNG files using any image editor. Save binary foreground masks here:

```text
results_plasmodium/manual_validation_package/manual_masks/
```

The expected names are already listed in `manual_validation_manifest.csv`.

### 3. Score automatic segmentation

After manual masks exist, run:

```python
RUN_MODE = "manual_validation_score"
```

This outputs:

```text
results_plasmodium/manual_validation_package/manual_validation_metrics.csv
results_plasmodium/manual_validation_package/manual_validation_summary.csv
```

The key columns are:

- `iou`
- `dice`
- `precision`
- `recall`
- `area_error_fraction`
- `confidence_level`

Recommended interpretation:

- Dice >= 0.90: high confidence
- Dice 0.80-0.90: usable for trend analysis, but describe limitations
- Dice < 0.80: segmentation needs revision before formal claims
- low precision: background/rim false positives
- low recall: missed pale plasmodium branches

Important: this validates the body mask only. Skeleton, endpoints, branchpoints, vein metrics, and biological interpretation still need separate checks if they are used as paper endpoints.
## Quick safety check after editing

Run this after any code change:

```powershell
python -m py_compile plasmodium_brightfield_analysis_optimized.py
```

In this Codex environment the checked command is:

```powershell
C:\Users\chen8\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile plasmodium_brightfield_analysis_optimized.py
```

## Practical next step

For the next real video, use:

```python
RUN_MODE = "formal_with_validation"
```

If the validation clips look acceptable, use `formal_only` for batch processing many videos.








