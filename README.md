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
Predictors use only raw-image information: hand-crafted descriptors, frozen
DINOv2/ResNet embeddings, or the optional fine-tuned image networks.

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

## Trainable Image Baselines

After pre-detection targets have been generated, run the new image-only comparisons:

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python scripts/cli.py finetune --dataset both
```

This evaluates frozen ResNet-50 + PCA(256)/Ridge, fine-tuned ResNet-50, and
fine-tuned DINOv2 ViT-B/14. It reads the existing pre-detection IoU targets and
never reruns Mask R-CNN or HC extraction. To run just one method:

```bash
poetry run python scripts/cli.py finetune --dataset ca --models resnet50_finetuned
```

All three use the same raw-image candidate ROI as frozen DINO, with ImageNet
normalization. ResNet uses ImageNet V2 pretrained weights (2048-d), a 232-pixel
short-side resize and 224-pixel center crop; DINO retains its 560/518 bicubic
preprocessing (768-d).
## Ablation Studies

```bash
poetry run python scripts/cli.py ablation
```

Equivalently: `poetry run python scripts/ablation.py`. This runs cross-dataset
transfer and individual-feature analysis from saved features, without a GPU.


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

## Baseline Comparison

After post-detection has completed:

```bash
poetry run python scripts/cli.py compare --dataset both
```

Choose a subset with `--baselines` (space-separated):

```bash
poetry run python scripts/cli.py compare --dataset both --baselines ensemble
poetry run python scripts/cli.py compare --dataset both --baselines confidence tta rca
poetry run python scripts/cli.py compare --dataset both --baselines dropout
```

Choices are `confidence`, `tta`, `rca`, `dropout`, and `ensemble`. Without this
argument, confidence, TTA, RCA and dropout run. Only selected baselines are
recomputed; saved results, score columns and protocol files for unselected
baselines are preserved. Existing scores must match the current image IDs, folds
and targets. The combined summary includes all saved results for both datasets.
Detector-based baselines require working CUDA and fail immediately if it cannot
initialize; there is no CPU fallback. RCA scoring remains CPU-based, and
`--reuse-existing` requires no GPU. Existing CPU timings are not GPU benchmarks.


### Coverage And Computational Cost

`results.csv` and `summary.csv` report accepted-set mean IoU and absolute gain
over accepting all images at 20% and 40% coverage, plus the accepted image count
and actual coverage after rounding up. Selection uses calibrated OOF predictions.

Scoring cost is recorded per image for all required passes together, including
mask aggregation and comparison. `scoring_seconds_mean`, `_median`, `_std` and
`_sum` summarize these totals across images. 

Rebuild coverage and cost summaries from saved scores without inference:

```bash
poetry run python scripts/cli.py compare --dataset both --reuse-existing
```

Older saved confidence/TTA scores have no extraction timings; with
`--reuse-existing`, these remain missing, not zero. To measure their costs, run
those baselines normally:

```bash
poetry run python scripts/cli.py compare --dataset both --baselines confidence tta
```

By default, every selected baseline recomputes its scores and records scoring
time. Only `--reuse-existing` skips extraction/inference and rebuilds the metrics
and timing summaries from saved data.
