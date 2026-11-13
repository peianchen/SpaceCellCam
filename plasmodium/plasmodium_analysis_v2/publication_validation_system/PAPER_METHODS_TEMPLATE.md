# Methods and Validation Reporting Template

Replace bracketed values only after the untouched test-set run is complete.

## Segmentation validation

The segmentation pipeline was developed using [N development videos from N
independent cultures] and finalized using [N validation videos]. All source code
and segmentation parameters were frozen before the independent test set was
sampled. The final test set comprised [N videos from N independent cultures]
under [conditions]. No test video or biological replicate was used during
parameter selection.

For each video, two non-overlapping five-frame windows were selected by a fixed
random seed from each temporal third (front, middle, and rear), yielding 30
full-resolution frames per video. The first and last four source frames were not
eligible because symmetric temporal context was required. Sampling and blind-ID
generation were performed by `prepare_dataset.py`.

Two annotators independently delineated visible occupied plasmodium from raw
images while blinded to algorithm output. [20%] of frames were repeated under
new blind identifiers to estimate intra-rater agreement. Disagreements were
resolved into an adjudicated consensus after both annotation rounds. The target
definition and exclusion rules were fixed before test annotation.

Strict HSV and temporal-recovery masks were compared with adjudicated masks
using Dice, IoU, precision, recall, signed and absolute area error, boundary F1
at [3] pixels, HD95, clDice, skeleton length error, endpoint and branchpoint
count error, rim false-positive fraction, and internal-hole false-positive
fraction. Consecutive frames were additionally evaluated for expansion,
retraction, net area change, and front-motion error. Agreement was reported for
the entire test set and separately for front, middle, and rear phases.

Frames were treated as repeated observations nested within biological
replicates. Confidence intervals were estimated by resampling independent
biological replicates with replacement using [2000] cluster-bootstrap
iterations. Area agreement was also assessed with Bland-Altman analysis. The
pre-specified acceptance thresholds and all worst-performing frames were
reported without post hoc frame exclusion.

## Reproducibility record

- Analysis source SHA-256: `[from prediction_run_metadata.json]`
- Prediction runner SHA-256: `[from prediction_run_metadata.json]`
- Parameter SHA-256: `[from prediction_run_metadata.json]`
- Evaluation and metric code SHA-256: `[from evaluation_status.json]`
- Software versions: `[from prediction_run_metadata.json]`
- Dataset role and eligibility: `[from evaluation_status.json]`

Do not state that skeleton, vein, or dynamic outputs are validated unless their
endpoint-specific results are reported and acceptable.
