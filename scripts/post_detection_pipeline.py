from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .detectron2_inference import build_iou_targets
from .datasets import DATASETS, fold_detector_configs
from .embedding_extraction import recompute_embeddings
from .evaluation_protocol import evaluation_metadata
from .post_detection_features import extract_post_detection_features
from .regression_utils import build_regressors, cv_predict_with_folds, target_fold_indices

OUTPUT_ROOT = Path("outputs/post_detection")
STAGE = "post-detection"
ROI_DINO_CSV = "dinov2_vitb14_cv2_roi518.csv"


def run_post_detection_pipeline(dataset: str, reuse_existing: bool = False) -> pd.DataFrame:
    config = replace(DATASETS[dataset], output_dir=OUTPUT_ROOT / dataset)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing:
        targets = pd.read_csv(config.output_dir / "iou_targets.csv").sort_values("row_id").reset_index(drop=True)
    else:
        targets = build_iou_targets(
            image_root=config.image_root,
            fold_configs=fold_detector_configs(config),
            output_dir=config.output_dir,
            score_thresh=0.5,
        )
    metadata = evaluation_metadata(targets, config.split_root, config.output_dir)
    embedding_dir = config.output_dir / "embeddings"
    feature_csv = config.output_dir / "post_detection_features.csv"
    if reuse_existing:
        post_table = pd.read_csv(feature_csv).sort_values("row_id").reset_index(drop=True)
        dino = _load_feature_table(embedding_dir / ROI_DINO_CSV)
    else:
        post_table = extract_post_detection_features(targets, fold_detector_configs(config), feature_csv)
        dino = recompute_embeddings(targets, embedding_dir, csv_names=(ROI_DINO_CSV,))[ROI_DINO_CSV]

    y = targets["iou_target"].to_numpy("float32")
    row_ids = targets["row_id"]
    fold_indices = target_fold_indices(targets["fold"].to_numpy("int64"))

    conf_cols = [col for col in post_table.columns if col.startswith("conf_")]
    tta_cols = [col for col in post_table.columns if col.startswith("tta_")]
    mt_cols = [col for col in post_table.columns if col.startswith("mt_")]

    feature_sets = {
        view: post_table[columns].to_numpy(dtype=np.float32)
        for view, columns in {
            "conf_only": conf_cols,
            "conf_mt": conf_cols + mt_cols,
            "conf_tta": conf_cols + tta_cols,
            "full": conf_cols + tta_cols + mt_cols,
        }.items()
    }
    feature_sets["post_plus_fdino_roi"] = np.hstack([feature_sets["full"], dino])

    rows: list[dict[str, float | str]] = []
    preds: dict[str, object] = {
        "row_id": row_ids.to_numpy(), "iou_target": y,
        "fold": targets["fold"].to_numpy(),
        "detector_seen_during_training": targets["detector_seen_during_training"].to_numpy(),
    }
    for view, x in feature_sets.items():
        for model_name, estimator in build_regressors().items():
            result, pred = cv_predict_with_folds(STAGE, config.name, view, model_name, estimator, x, y, fold_indices)
            rows.append({**result.__dict__, **metadata})
            preds[f"{view}__{model_name}"] = pred
            write_outputs(config.output_dir, rows, preds)

    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def write_outputs(output_dir: Path, rows: list[dict[str, float | str]], preds: dict[str, object]) -> None:
    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(preds).to_csv(output_dir / "oof_predictions.csv", index=False)


def _load_feature_table(path: Path) -> np.ndarray:
    table = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
    return table.drop(columns=["row_id"]).to_numpy(dtype=np.float32)
