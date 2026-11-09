from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils.fix_coco_fold_splits import SplitDatasetConfig, clean_dataset_splits
from .detectron2_inference import DetectorFoldConfig, build_fold_clean_targets
from .embedding_extraction import recompute_embeddings
from .fold_clean_pre_detection_pipeline import DATASETS, DatasetConfig, fold_detector_configs
from .post_detection_features import extract_post_detection_features
from .regression_utils import build_post_detection_estimators, cv_predict_with_folds, target_fold_indices

OUTPUT_ROOT = Path("outputs/post_detection")
STAGE = "post-detection"
ROI_DINO_CSV = "dinov2_vitb14_cv2_roi518.csv"


def run_fold_clean_post_detection_pipeline(dataset: str, reuse_existing: bool = False) -> pd.DataFrame:
    base = DATASETS[dataset]
    config = DatasetConfig(
        name=base.name,
        dataset_root=base.dataset_root,
        image_root=base.image_root,
        split_root=base.split_root,
        model_root=base.model_root,
        output_dir=OUTPUT_ROOT / dataset,
    )
    clean_dataset_splits(SplitDatasetConfig(name=config.name, root=config.split_root))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing:
        targets = pd.read_csv(config.output_dir / "fold_clean_targets.csv").sort_values("row_id").reset_index(drop=True)
        post_table = pd.read_csv(config.output_dir / "post_detection_features.csv").sort_values("row_id").reset_index(drop=True)
        embeddings = {
            ROI_DINO_CSV: _load_feature_table(config.output_dir / "embeddings" / ROI_DINO_CSV),
        }
    else:
        fold_configs = fold_detector_configs(config)
        targets = build_fold_clean_targets(
            dataset=config.name,
            image_root=config.image_root,
            fold_configs=fold_configs,
            output_dir=config.output_dir,
            score_thresh=0.5,
        )
        feature_csv = config.output_dir / "post_detection_features.csv"
        extract_post_detection_features(targets, fold_configs, feature_csv)
        post_table = pd.read_csv(feature_csv).sort_values("row_id").reset_index(drop=True)
        embeddings = recompute_embeddings(targets, config.output_dir / "embeddings", csv_names=(ROI_DINO_CSV,))

    y = targets["iou_target"].to_numpy("float32")
    row_ids = targets["row_id"]
    fold_indices = target_fold_indices(targets["fold"].to_numpy("int64"))

    conf_cols = [col for col in post_table.columns if col.startswith("conf_")]
    tta_cols = [col for col in post_table.columns if col.startswith("tta_")]
    mt_cols = [col for col in post_table.columns if col.startswith("mt_")]

    conf_only = post_table[conf_cols].to_numpy(dtype=np.float32)
    conf_mt = post_table[conf_cols + mt_cols].to_numpy(dtype=np.float32)
    conf_tta = post_table[conf_cols + tta_cols].to_numpy(dtype=np.float32)
    full = post_table[conf_cols + tta_cols + mt_cols].to_numpy(dtype=np.float32)

    feature_sets = {
        "conf_only": conf_only,
        "conf_mt": conf_mt,
        "conf_tta": conf_tta,
        "full": full,
        "post_plus_fdino_roi": np.hstack([full, embeddings[ROI_DINO_CSV]]),
    }

    rows: list[dict[str, float | str]] = []
    preds: dict[str, object] = {"row_id": row_ids.to_numpy(), "iou_target": y}
    for view, x in feature_sets.items():
        for model_name, estimator in build_post_detection_estimators().items():
            result, pred = cv_predict_with_folds(STAGE, config.name, view, model_name, estimator, x, y, fold_indices)
            rows.append(result.__dict__)
            preds[f"{view}__{model_name}"] = pred
            write_outputs(config.output_dir, rows, preds)

    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def write_outputs(output_dir: Path, rows: list[dict[str, float | str]], preds: dict[str, object]) -> None:
    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(preds).to_csv(output_dir / "oof_predictions.csv", index=False)


def _load_feature_table(path: Path) -> np.ndarray:
    table = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
    return table.drop(columns=["row_id"]).to_numpy(dtype=np.float32)
