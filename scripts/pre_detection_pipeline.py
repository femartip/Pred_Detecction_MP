"Runs Mask R-CNN, extracts hand-crafted features, feature embeddings and saves metrics"
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DATASETS, fold_detector_configs
from .detectron2_inference import build_iou_targets
from .embedding_extraction import recompute_embeddings
from .evaluation_protocol import evaluation_metadata
from .handcrafted_features import recompute_handcrafted_features
from .regression_utils import (
    build_fdino_estimator,
    build_regressors,
    cv_predict_with_folds,
    target_fold_indices,
)

STAGE = "pre-detection"


def run_pre_detection_pipeline(dataset: str, reuse_existing: bool = False) -> pd.DataFrame:
    config = DATASETS[dataset]
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing:
        targets = pd.read_csv(config.output_dir / "iou_targets.csv").sort_values("row_id").reset_index(drop=True)
    else:
        targets = build_iou_targets(
            image_root=config.image_root,
            fold_configs=fold_detector_configs(config),
            output_dir=config.output_dir,
        )
    metadata = evaluation_metadata(targets, config.split_root, config.output_dir)
    if reuse_existing:
        embedding_dir = config.output_dir / "embeddings"
        embeddings = {
            "dinov2_vitb14_cv2_roi518.csv": _load_feature_table(embedding_dir / "dinov2_vitb14_cv2_roi518.csv"),
            "dinov2_vitb14.csv": _load_feature_table(embedding_dir / "dinov2_vitb14.csv"),
        }
        handcrafted = _load_feature_table(config.output_dir / "handcrafted_features.csv")
    else:
        embedding_dir = config.output_dir / "embeddings"
        embeddings = recompute_embeddings(targets, embedding_dir, csv_names=("dinov2_vitb14_cv2_roi518.csv", "dinov2_vitb14.csv"))
        handcrafted = recompute_handcrafted_features(targets, config.output_dir / "handcrafted_features.csv")
    y = targets["iou_target"].to_numpy(np.float32)
    row_ids = targets["row_id"]
    fold_indices = target_fold_indices(targets["fold"].to_numpy(np.int64))

    rows: list[dict[str, float | str]] = []
    preds: dict[str, np.ndarray] = {
        "row_id": row_ids.to_numpy(), "iou_target": y,
        "fold": targets["fold"].to_numpy(),
        "detector_seen_during_training": targets["detector_seen_during_training"].to_numpy(),
    }
    for model_name, estimator in build_regressors().items():
        result, pred = cv_predict_with_folds(STAGE, config.name, "fhc", model_name, estimator, handcrafted, y, fold_indices)
        rows.append({**result.__dict__, **metadata})
        preds[f"fhc__{model_name}"] = pred
        write_outputs(config.output_dir, rows, preds)

    dino_views = {
        "fdino_roi": embeddings["dinov2_vitb14_cv2_roi518.csv"],
        "fdino_full": embeddings["dinov2_vitb14.csv"],
    }
    for view, x in dino_views.items():
        model_name = "pca256_ridge"
        estimator = build_fdino_estimator()
        result, pred = cv_predict_with_folds(STAGE, config.name, view, model_name, estimator, x, y, fold_indices)
        rows.append({**result.__dict__, **metadata})
        preds[f"{view}__{model_name}"] = pred
        write_outputs(config.output_dir, rows, preds)
        for model_name, estimator in build_regressors().items():
            result, pred = cv_predict_with_folds(STAGE, config.name, view, model_name, estimator, x, y, fold_indices)
            rows.append({**result.__dict__, **metadata})
            preds[f"{view}__{model_name}"] = pred
            write_outputs(config.output_dir, rows, preds)

    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def write_outputs(output_dir: Path, rows: list[dict[str, float | str]], preds: dict[str, np.ndarray]) -> None:
    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(preds).to_csv(output_dir / "oof_predictions.csv", index=False)


def _load_feature_table(path: Path) -> np.ndarray:
    table = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
    return table.drop(columns=["row_id"]).to_numpy(dtype=np.float32)
