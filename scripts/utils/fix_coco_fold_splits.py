from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SplitDatasetConfig:
    name: str
    root: Path


SPLIT_DATASETS = {
    "ca": SplitDatasetConfig(name="ca", root=Path("data/Dataset/Dataset_CA")),
    "glass": SplitDatasetConfig(name="glass", root=Path("data/Dataset/Dataset_vidrio")),
}


def clean_all_dataset_splits() -> None:
    for config in SPLIT_DATASETS.values():
        clean_dataset_splits(config)


def clean_dataset_splits(config: SplitDatasetConfig) -> None:
    full_coco = json.loads((config.root / "coco_format.json").read_text())
    clean_dir = config.root / "clean_splits"
    clean_dir.mkdir(parents=True, exist_ok=True)
    categories = full_coco.get("categories", [])
    info = full_coco.get("info", {})
    licenses = full_coco.get("licenses", [])

    for fold in range(5):
        for split_name in ("train", "test"):
            source = config.root / f"{split_name}_coco_{fold}_fold.json"
            split_coco = json.loads(source.read_text())
            image_ids = {int(image["id"]) for image in split_coco["images"]}
            annotations = [ann for ann in full_coco["annotations"] if int(ann["image_id"]) in image_ids]
            cleaned = {
                "info": info,
                "licenses": licenses,
                "images": split_coco["images"],
                "annotations": annotations,
                "categories": categories,
            }
            target = clean_dir / f"{split_name}_coco_{fold}_fold.json"
            target.write_text(json.dumps(cleaned))


def main() -> None:
    clean_all_dataset_splits()
    print("Clean fold files written to data/Dataset/*/clean_splits")


if __name__ == "__main__":
    main()
