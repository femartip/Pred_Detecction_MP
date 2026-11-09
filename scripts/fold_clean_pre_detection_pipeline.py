from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils.fix_coco_fold_splits import SplitDatasetConfig, clean_dataset_splits
from .detectron2_inference import DetectorFoldConfig, build_fold_clean_targets
from .embedding_extraction import recompute_embeddings
from .handcrafted_features import recompute_handcrafted_features
from .regression_utils import (
    build_fdino_estimator,
    build_post_detection_estimators,
    cv_predict_with_folds,
    target_fold_indices,
)

OUTPUT_ROOT = Path("outputs/pre_detection")
STAGE = "pre-detection"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_root: Path
    image_root: Path
    split_root: Path
    model_root: Path
    output_dir: Path


DATASETS: dict[str, DatasetConfig] = {
    "ca": DatasetConfig(
        name="ca",
        dataset_root=Path("data/Dataset/Dataset_CA"),
        image_root=Path("data/Dataset/Dataset_CA/images"),
        split_root=Path("data/Dataset/Dataset_CA"),
        model_root=Path("models/CA_final_model"),
        output_dir=OUTPUT_ROOT / "ca",
    ),
    "glass": DatasetConfig(
        name="glass",
        dataset_root=Path("data/Dataset/Dataset_vidrio"),
        image_root=Path("data/Dataset/Dataset_vidrio/images"),
        split_root=Path("data/Dataset/Dataset_vidrio"),
        model_root=Path("models/Glass_final_model"),
        output_dir=OUTPUT_ROOT / "glass",
    ),
}


def run_fold_clean_pre_detection_pipeline(dataset: str, reuse_existing: bool = False) -> pd.DataFrame:
    config = DATASETS[dataset]
    clean_dataset_splits(_split_cleanup_config(config))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing:
        targets = pd.read_csv(config.output_dir / "fold_clean_targets.csv").sort_values("row_id").reset_index(drop=True)
        embedding_dir = config.output_dir / "embeddings"
        embeddings = {
            "dinov2_vitb14_cv2_roi518.csv": _load_feature_table(embedding_dir / "dinov2_vitb14_cv2_roi518.csv"),
            "dinov2_vitb14.csv": _load_feature_table(embedding_dir / "dinov2_vitb14.csv"),
        }
        handcrafted = _load_feature_table(config.output_dir / "handcrafted_features.csv")
    else:
        targets = build_fold_clean_targets(
            dataset=config.name,
            image_root=config.image_root,
            fold_configs=fold_detector_configs(config),
            output_dir=config.output_dir,
        )
        embedding_dir = config.output_dir / "embeddings"
        embeddings = recompute_embeddings(targets, embedding_dir, csv_names=("dinov2_vitb14_cv2_roi518.csv", "dinov2_vitb14.csv"))
        handcrafted = recompute_handcrafted_features(targets, config.output_dir / "handcrafted_features.csv")
    y = targets["iou_target"].to_numpy(np.float32)
    row_ids = targets["row_id"]
    fold_indices = target_fold_indices(targets["fold"].to_numpy(np.int64))

    rows: list[dict[str, float | str]] = []
    preds: dict[str, np.ndarray] = {"row_id": row_ids.to_numpy(), "iou_target": y}
    for model_name, estimator in build_post_detection_estimators().items():
        result, pred = cv_predict_with_folds(STAGE, config.name, "fhc", model_name, estimator, handcrafted, y, fold_indices)
        rows.append(result.__dict__)
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
        rows.append(result.__dict__)
        preds[f"{view}__{model_name}"] = pred
        write_outputs(config.output_dir, rows, preds)
        for model_name, estimator in build_post_detection_estimators().items():
            result, pred = cv_predict_with_folds(STAGE, config.name, view, model_name, estimator, x, y, fold_indices)
            rows.append(result.__dict__)
            preds[f"{view}__{model_name}"] = pred
            write_outputs(config.output_dir, rows, preds)

    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def fold_detector_configs(config: DatasetConfig) -> list[DetectorFoldConfig]:
    split_dir = config.split_root / "clean_splits"
    fold_configs: list[DetectorFoldConfig] = []
    for fold in range(5):
        if config.name == "ca":
            checkpoint_path = config.model_root / str(fold) / f"model_final0_{fold}.pth"
        else:
            checkpoint_path = config.model_root / f"model_final0_{fold}.pth"
        fold_configs.append(
            DetectorFoldConfig(
                config_path=config.model_root / f"config_0_{fold}.yaml",
                checkpoint_path=checkpoint_path,
                test_coco_path=split_dir / f"test_coco_{fold}_fold.json",
            )
        )
    return fold_configs


def write_outputs(output_dir: Path, rows: list[dict[str, float | str]], preds: dict[str, np.ndarray]) -> None:
    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(preds).to_csv(output_dir / "oof_predictions.csv", index=False)


def _load_feature_table(path: Path) -> np.ndarray:
    table = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
    return table.drop(columns=["row_id"]).to_numpy(dtype=np.float32)


def _split_cleanup_config(config: DatasetConfig):
    return SplitDatasetConfig(name=config.name, root=config.split_root)
