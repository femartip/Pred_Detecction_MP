"""Shared dataset locations and per-fold detector inputs."""

from dataclasses import dataclass
from pathlib import Path

from .detectron2_inference import DetectorFoldConfig


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
        model_root=Path("models/ca"),
        output_dir=Path("outputs/pre_detection/ca"),
    ),
    "glass": DatasetConfig(
        name="glass",
        dataset_root=Path("data/Dataset/Dataset_vidrio"),
        image_root=Path("data/Dataset/Dataset_vidrio/images"),
        split_root=Path("data/Dataset/Dataset_vidrio"),
        model_root=Path("models/glass"),
        output_dir=Path("outputs/pre_detection/glass"),
    ),
}


def fold_detector_configs(config: DatasetConfig) -> list[DetectorFoldConfig]:
    return [
        DetectorFoldConfig(
            config_path=config.model_root / f"fold{fold}/config.yaml",
            checkpoint_path=config.model_root / f"fold{fold}/model_final.pth",
            test_coco_path=config.split_root / f"test_coco_{fold}_fold.json",
        )
        for fold in range(5)
    ]
