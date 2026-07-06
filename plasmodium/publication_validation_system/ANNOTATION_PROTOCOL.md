# Pre-specified Manual Annotation Protocol

Complete this protocol before opening the sampled test images. Record any rule
change with a date and version. Do not change the rule after predictions or test
metrics have been viewed.

## Target definition

The primary target is the **visible occupied plasmodium mask** in the dish.

Include:

- visible yellow/green plasmodial sheets;
- visible connected tubes and thin branches, including pale branches when their boundary can be distinguished from substrate;
- dense plasmodial regions even when internal texture is heterogeneous.

Exclude:

- gray/transparent agar or substrate gaps between branches, including gaps enclosed by the network;
- dish rim, reflection arcs, glare, labels, dust, and objects outside the analysis dish circle;
- background pixels included only because a nearby branch is present;
- an underlying food/inoculum object when it is visually distinguishable from plasmodial tissue.

Before annotation begins, the investigators must document whether the central
food/inoculum is biologically part of the desired measurement. The default rule
above excludes the non-plasmodial object but includes visible plasmodium growing
over it. Apply the same decision to every video and annotator.

## Drawing procedure

1. Work from the full-resolution raw image only. Algorithm masks and overlays remain hidden.
2. Zoom as needed, but do not apply image enhancement that changes the decision rule.
3. Draw the occupied tissue; preserve visible internal background holes.
4. Save a single-channel PNG with foreground 255 and background 0, using the unchanged blind filename and image dimensions.
5. Do not discuss a frame with the other annotator before both primary masks are complete.
6. Annotate repeat images independently without looking up the corresponding primary image.
7. Log genuinely ambiguous boundaries in a separate annotation note; do not silently alter the target definition.

## Adjudication

After both primary and repeat rounds are complete, compare disagreements on raw
images. An adjudicator creates one consensus mask in `adjudicated_masks` while
preserving the two original masks. Record the reason for systematic disagreement
(pale branch, internal gap, rim/reflection, glare, or central object). The
adjudicated mask is the reference for algorithm metrics; A-vs-B remains the
human agreement baseline.

## Quality control

- Mask dimensions must match the raw image exactly.
- Check binary values and inspect overlays only after the primary mask is saved.
- Review all large A-vs-B disagreements and all final algorithm outliers.
- Never replace a difficult frame merely because its metric is poor.
