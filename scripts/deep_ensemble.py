"""Build each fold ensemble from its target detector and two new detectors."""

import gc
import json
from time import perf_counter

import pandas as pd
import torch


def train_deep_ensemble(config, fold_configs):
    from dataclasses import replace

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.utils.env import seed_all_rng

    if not torch.cuda.is_available():
        raise RuntimeError("Deep-ensemble training requires CUDA in the Poetry environment.")
    mapping = pd.read_csv(config.model_root / "hf_image_mapping.csv")
    pretrained = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    members_by_fold = []
    for fold, base in enumerate(fold_configs):
        root = config.model_root / f"fold{fold}"
        original = json.loads((root / "train_coco.json").read_text())
        original_ids = {image["id"] for image in original["images"]}
        expected = set(mapping.loc[mapping["original_image_id"].isin(original_ids), "hf_row_id"])
        train = json.loads((config.split_root / f"train_coco_{fold}_fold.json").read_text())
        assert {image["id"] for image in train["images"]} == expected, f"Training mapping mismatch: {root}"
        dataset_name = f"ensemble_{config.name}_fold{fold}"
        members = [base]
        for member, seed in enumerate((42, 43), start=1):
            directory = root / f"member{member}"
            directory.mkdir(parents=True, exist_ok=True)
            train_path = directory / "train_coco.json"
            train_path.write_text(json.dumps(train))
            if member == 1:
                register_coco_instances(dataset_name, {}, str(train_path.resolve()), str(config.image_root.resolve()))

            cfg = get_cfg()
            cfg.merge_from_file(str(base.config_path))
            cfg.MODEL.WEIGHTS = pretrained
            cfg.MODEL.DEVICE = "cuda"
            cfg.DATASETS.TRAIN = (dataset_name,)
            cfg.DATASETS.TEST = ()
            cfg.TEST.EVAL_PERIOD = 0
            cfg.SEED = seed
            cfg.OUTPUT_DIR = str(directory.resolve())
            (directory / "config.yaml").write_text(cfg.dump())
            record = {
                "status": "training", "seed": seed, "fold": fold,
                "initial_weights": pretrained,
                "target_detector": str(base.checkpoint_path),
                "target_detector_is_member": True,
                "training_ids": "HF IDs mapped to the original detector training partition",
                "original_training_images": len(original_ids),
                "available_training_images": len(expected),
                "unavailable_original_training_images": len(original_ids) - len(expected),
                "device": torch.cuda.get_device_name(),
                "training_timing_scope": "trainer initialization, pretrained loading, optimization and checkpoint writing; excludes dataset preparation",
            }
            record_path = directory / "training.json"
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"[ensemble {config.name}] fold {fold}, member {member}, seed {seed}: "
                  f"{len(expected)} images, {cfg.SOLVER.MAX_ITER} iterations", flush=True)
            seed_all_rng(seed)
            torch.cuda.synchronize()
            start = perf_counter()
            trainer = DefaultTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            torch.cuda.synchronize()
            record.update(status="complete", training_seconds=perf_counter() - start)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            members.append(replace(base, config_path=directory / "config.yaml", checkpoint_path=directory / "model_final.pth"))
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
        members_by_fold.append(members)
    return members_by_fold
