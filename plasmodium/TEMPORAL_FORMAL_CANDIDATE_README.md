# Temporal Formal Candidate

Run:

```powershell
python plasmodium_brightfield_analysis_temporal_formal.py
```

The script is independent from `plasmodium_brightfield_analysis.py`. It imports
the established segmentation/metric functions but writes to a separate output
directory so the strict formal run is never overwritten.

## Default output

Directory: `results_plasmodium_temporal_formal_candidate/`

- `plasmodium_overlay_temporal_candidate.mp4`: 60 FPS, full-resolution temporal mask overlay.
- `plasmodium_metrics_temporal_candidate.csv`: temporal primary columns plus `strict_*` controls.
- `strict_temporal_comparison_summary.csv`: overall/front/middle/rear comparison.
- `strict_temporal_comparison.png`: strict and temporal trend comparison.
- `plasmodium_summary_temporal_candidate.png`: temporal morphology/dynamics summary.
- `plasmodium_summary_temporal_candidate_veins.png`: sparse trusted/raw vein trends when available.
- `debug_examples/`: strict, candidate, temporal-added, temporal-removed, and comparison masks.
- `plasmodium_qc_flags.csv`: automatic anomaly flags.
- `temporal_technical_readiness.json`: internal technical checks, not accuracy evidence.
- `temporal_run_configuration.json`: source hashes, exact parameters, versions, and run metadata.

## Important columns

Unsuffixed morphology/dynamic columns (`area`, `dA_dt`, `explore_ratio`,
`skeleton_length`, and so on) use the temporal candidate mask. The original HSV
measurements remain in `strict_*` columns.

- `temporal_valid=False`: first/last four frames lack full symmetric context; strict mask is used.
- `temporal_dynamic_valid=False`: temporal change metrics touching an invalid edge frame must not be interpreted.
- `temporal_added_*`: pixels present only in temporal mask.
- `temporal_removed_*`: transient strict pixels removed by final temporal consensus.
- `strict_temporal_dice`: same-frame internal agreement; this is not ground-truth Dice.
- `temporal_adjacent_iou`: stability between consecutive temporal masks; this is not accuracy.

## Promotion rule

Keep `FORMAL_STATUS = "candidate_pending_independent_validation"` while tuning or
reviewing this run. It may become the formal primary method only after:

1. the algorithm source and parameters are frozen;
2. an untouched independent test set is prepared;
3. blinded annotations and adjudication are complete;
4. `publication_validation_system/evaluate_validation.py` reports the temporal primary method passed;
5. no parameter is changed after viewing final test results.

If any parameter changes after the final test is inspected, that test set becomes
development data and a new independent test set is required.

## Performance

The default is eight CPU workers, 16 core frames per batch, and four halo frames
on each side. Peak source-frame memory is bounded to roughly 24 frames. Setting
`CPU_WORKERS = 1` enables a tested sequential fallback but is substantially
slower.
