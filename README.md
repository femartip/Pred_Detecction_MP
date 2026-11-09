# Microfibre IoU Meta-Learner

This repo contains the local microfibre IoU meta-learning pipelines with a
fold-clean 5-fold Mask R-CNN evaluation protocol. It currently supports:

- `pre-detection`: raw-image features only
- `post-detection`: detector-output features, with optional DINO feature
  augmentation for comparison

## Fold-clean methodology

For each dataset:

1. Clean the provided train/test COCO fold files by filtering annotations to
   the image IDs present in each split.
2. Run the local fold-specific Mask R-CNN checkpoint on the held-out images.
3. Rebuild the per-image IoU target using only the detector assigned to that
   held-out fold.
4. Recompute the features required by the selected stage.
5. Train the retained regressors only on the other folds.
6. Aggregate out-of-fold predictions across all five folds.

For `pre-detection`, detector outputs are used only to define the target.
Predictors use only raw-image features: hand-crafted descriptors and frozen
DINOv2 embeddings.

For `post-detection`, the same fold-clean detector is also used to extract:

- `Fconf`: confidence-derived summary features
- `FTTA`: test-time augmentation consistency features
- `FMT`: multi-threshold stability features

No `FCNN` features are used. The post-detection stage additionally evaluates
whether concatenating DINOv2 embeddings improves performance.

## Outputs

`pre_detection` writes:

```text
outputs/pre_detection/ca/fold_clean_targets.csv
outputs/pre_detection/ca/embeddings/*.csv
outputs/pre_detection/ca/handcrafted_features.csv
outputs/pre_detection/ca/results.csv
outputs/pre_detection/ca/oof_predictions.csv
outputs/pre_detection/glass/fold_clean_targets.csv
outputs/pre_detection/glass/embeddings/*.csv
outputs/pre_detection/glass/handcrafted_features.csv
outputs/pre_detection/glass/results.csv
outputs/pre_detection/glass/oof_predictions.csv
outputs/pre_detection/summary.csv
```

`post_detection` writes:

```text
outputs/post_detection/ca/fold_clean_targets.csv
outputs/post_detection/ca/post_detection_features.csv
outputs/post_detection/ca/embeddings/dinov2_vitb14_cv2_roi518.csv
outputs/post_detection/ca/embeddings/dinov2_vitb14.csv
outputs/post_detection/ca/results.csv
outputs/post_detection/ca/oof_predictions.csv
outputs/post_detection/glass/fold_clean_targets.csv
outputs/post_detection/glass/post_detection_features.csv
outputs/post_detection/glass/embeddings/dinov2_vitb14_cv2_roi518.csv
outputs/post_detection/glass/embeddings/dinov2_vitb14.csv
outputs/post_detection/glass/results.csv
outputs/post_detection/glass/oof_predictions.csv
outputs/post_detection/summary.csv
```

## Install

```bash
poetry install --with dev
```

## Run

Fix the fold files:

```bash
poetry run python Utils/fix_coco_fold_splits.py
```

Run both datasets:

```bash
poetry run python -m microfibre_iou.cli --stage both --dataset both
```

Run only pre-detection:

```bash
poetry run python -m microfibre_iou.cli --stage pre --dataset ca
poetry run python -m microfibre_iou.cli --stage pre --dataset glass
```

Run only post-detection:

```bash
poetry run python -m microfibre_iou.cli --stage post --dataset ca
poetry run python -m microfibre_iou.cli --stage post --dataset glass
```
