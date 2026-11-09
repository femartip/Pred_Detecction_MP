# Microfibre IoU Meta-Learner

This repo contains the local microfibre IoU meta-learning pipelines using existing Mask R-CNN checkpoints and five meta-regression folds. It supports:

- `pre-detection`: raw-image features only
- `post-detection`: detector-output features, with optional DINO feature
  augmentation for comparison

## Evaluation Methodology

For each dataset:

1. Match pinned HF images to the original checkpoint training records.
   Use an unseen-image detector when available; otherwise assign an existing
   detector and explicitly flag training exposure.
2. Run the assigned local Mask R-CNN checkpoint on every image.
3. Rebuild the per-image IoU target using only the detector assigned to that
   evaluation fold.
4. Recompute the features required by the selected stage.
5. Train the retained regressors only on the other folds.
6. Aggregate out-of-fold predictions across all five folds.


For `pre-detection`, detector outputs are used only to define the target.
Predictors use only raw-image features: hand-crafted descriptors and frozen
DINOv2 embeddings.

For `post-detection`, the same assigned detector is also used to extract:

- `Fconf`: confidence-derived summary features
- `FTTA`: test-time augmentation consistency features
- `FMT`: multi-threshold stability features

No `FCNN` features are used. The post-detection stage additionally evaluates
whether concatenating DINOv2 embeddings improves performance.

## Outputs

`pre_detection` writes:

```text
outputs/pre_detection/ca/iou_targets.csv
outputs/pre_detection/ca/embeddings/*.csv
outputs/pre_detection/ca/handcrafted_features.csv
outputs/pre_detection/ca/results.csv
outputs/pre_detection/ca/oof_predictions.csv
outputs/pre_detection/glass/iou_targets.csv
outputs/pre_detection/glass/embeddings/*.csv
outputs/pre_detection/glass/handcrafted_features.csv
outputs/pre_detection/glass/results.csv
outputs/pre_detection/glass/oof_predictions.csv
outputs/pre_detection/summary.csv
```

`post_detection` writes:

```text
outputs/post_detection/ca/iou_targets.csv
outputs/post_detection/ca/post_detection_features.csv
outputs/post_detection/ca/embeddings/dinov2_vitb14_cv2_roi518.csv
outputs/post_detection/ca/results.csv
outputs/post_detection/ca/oof_predictions.csv
outputs/post_detection/glass/iou_targets.csv
outputs/post_detection/glass/post_detection_features.csv
outputs/post_detection/glass/embeddings/dinov2_vitb14_cv2_roi518.csv
outputs/post_detection/glass/results.csv
outputs/post_detection/glass/oof_predictions.csv
outputs/post_detection/summary.csv
```

## Install Dependencies

```bash
poetry install --with dev
```

## Download And Prepare

Download both datasets and model bundles with the Hugging Face CLI:

```bash
poetry run hf download femartip/microfibres_CA_filter --repo-type dataset --revision 502903a9aada7b67fe823a6aee5889231a1006c8 --local-dir data/downloads/ca
poetry run hf download femartip/microfibres_Glass_filter --repo-type dataset --revision 857c4e632cd82e04ebe55e02a3ce5afcb04f50a6 --local-dir data/downloads/glass
poetry run hf download femartip/microfibres-ca-mask-rcnn --local-dir models/ca
poetry run hf download femartip/microfibres-glass-mask-rcnn --local-dir models/glass
poetry run python scripts/cli.py prepare --dataset both
```

Preparation is local-only and recomputes images, COCO annotations and evaluation
splits under `data/Dataset/{Dataset_CA,Dataset_vidrio}/`. It verifies bundle
checksums and HF-to-original image mappings, preserves original held-out
assignments, and balances remaining images with seed 42. All 1287 CA and 1203
Glass images are included; 309 CA and 86 Glass images were seen by their assigned
detector during training and are explicitly flagged. This is not a fully
detector-held-out evaluation.

The pipelines load downloaded checkpoints directly from
`models/{ca,glass}/fold{0..4}/` and read the prepared splits without modifying them.
Use `--dataset ca` or `--dataset glass` to prepare only one dataset. Re-running
preparation overwrites the generated files, but does not run inference or extract
features. Original detector training/test records remain in the model bundles;
the generated test files are the expanded meta-regression evaluation partitions.

## Run

```text
HF CLI downloads → prepare_data.py
├── pre_detection_pipeline.py:  target IoU → validation → HC + DINO → regression
└── post_detection_pipeline.py: target IoU → validation → conf/TTA/MT + DINO → regression
```

Both pipelines call `evaluation_protocol.py` for validation and
`regression_utils.py` for fitting and metrics. They run independently through
`scripts/cli.py`; neither requires the other to run first.

Prepare once after downloading, then run either stage for both datasets:

```bash
poetry run python scripts/cli.py prepare --dataset both
poetry run python scripts/cli.py run --stage pre --dataset both
poetry run python scripts/cli.py run --stage post --dataset both
```

Or run both stages:

```bash
poetry run python scripts/cli.py run --stage both --dataset both
```

Use `--dataset ca` or `--dataset glass` for one dataset. Add `--reuse-existing`
to refit regressors from existing targets/features; omit it for a fresh run.
Archived subset results cannot be reused with the full-population splits.
