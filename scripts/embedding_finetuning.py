"""Frozen ResNet control and end-to-end image-only quality predictors."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .datasets import DATASETS
from .embedding_extraction import (
    EMBEDDING_DIMS, MODEL_NAMES, RESNET_CSV, build_embedding_model, crop_candidate_roi,
    image_to_dino_tensor, image_to_resnet_tensor, recompute_embeddings,
)
from .evaluation_protocol import evaluation_metadata
from .pre_detection_pipeline import write_outputs
from .regression_utils import SEED, build_fdino_estimator, cv_predict_with_folds, metrics_row, target_fold_indices

cv2.setNumThreads(1)


class QualityImages(Dataset):
    def __init__(self, targets, backbone):
        self.targets = targets.reset_index(drop=True)
        self.transform = image_to_dino_tensor if backbone == "dinov2" else image_to_resnet_tensor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        record = self.targets.iloc[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(crop_candidate_roi(image.convert("RGB"))).squeeze(0)
        return tensor, np.float32(record.iou_target), int(record.row_id)


def image_loader(targets, backbone, batch_size, workers, seed, shuffle=False):
    return DataLoader(
        QualityImages(targets, backbone), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate(model, loader):
    model.eval()
    total_loss, row_ids, predictions = 0.0, [], []
    with torch.inference_mode():
        for images, values, ids in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                predicted = model(images.to("cuda", non_blocking=True)).flatten().float().sigmoid()
                loss = F.mse_loss(predicted, values.to("cuda"), reduction="sum")
            total_loss += loss.item()
            row_ids.extend(ids.tolist())
            predictions.extend(predicted.cpu().tolist())
    return total_loss / len(loader.dataset), np.asarray(row_ids), np.asarray(predictions, dtype=np.float32)


def fit_outer_fold(targets, train, test, backbone, directory, fold, epochs, patience, batch_size, workers):
    seed = SEED + int(fold)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    inner_train, validation = train_test_split(train, test_size=0.2, random_state=seed)
    assert not set(inner_train).intersection(validation)
    assert not set(test).intersection(np.concatenate([inner_train, validation]))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "split.json").write_text(json.dumps({
        "seed": seed,
        "train_row_ids": targets.iloc[inner_train]["row_id"].tolist(),
        "validation_row_ids": targets.iloc[validation]["row_id"].tolist(),
        "test_row_ids": targets.iloc[test]["row_id"].tolist(),
    }, indent=2) + "\n")
    train_loader = image_loader(targets.iloc[inner_train], backbone, batch_size, workers, seed, shuffle=True)
    validation_loader = image_loader(targets.iloc[validation], backbone, batch_size, workers, seed)
    model = build_embedding_model(backbone, num_classes=1).to("cuda")
    head = list(model.get_classifier().parameters())
    head_ids = {id(parameter) for parameter in head}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": 1e-5},
        {"params": head, "lr": 1e-3},
    ], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, values, _ in tqdm(train_loader, desc=f"[{backbone} fold {fold}] epoch {epoch}", unit="batch"):
            images, values = images.to("cuda", non_blocking=True), values.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                predicted = model(images).flatten().float().sigmoid()
                loss = F.mse_loss(predicted, values)
            assert torch.isfinite(loss).item(), "Non-finite training loss"
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(values)
        validation_loss, _, _ = evaluate(model, validation_loader)
        assert np.isfinite(validation_loss), "Non-finite validation loss"
        history.append({"epoch": epoch, "train_mse": train_loss / len(inner_train), "validation_mse": validation_loss})
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"[{backbone} fold {fold}] epoch {epoch}: validation MSE={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(model.state_dict(), directory / "best_model.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    del optimizer, scaler, train_loader, validation_loader
    model.load_state_dict(torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True))
    # The outer test images are evaluated only after checkpoint selection is complete.
    test_loader = image_loader(targets.iloc[test], backbone, batch_size, workers, seed)
    _, row_ids, predicted = evaluate(model, test_loader)
    selection = {"fold": int(fold), "selected_epoch": best_epoch, "validation_mse": best_loss,
                 "n_train": len(inner_train), "n_validation": len(validation), "n_test": len(test)}
    (directory / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    del model, test_loader
    torch.cuda.empty_cache()
    return row_ids, predicted, selection


def run_embedding_finetuning(dataset, models, reuse_existing=False, epochs=30, patience=5, batch_size=4, workers=2):
    assert epochs > 0 and patience > 0 and batch_size > 0 and workers >= 0
    if not reuse_existing and not torch.cuda.is_available():
        raise RuntimeError("Embedding experiments require working CUDA; CPU fallback is disabled.")
    config = DATASETS[dataset]
    output = config.output_dir
    targets = pd.read_csv(output / "iou_targets.csv").sort_values("row_id").reset_index(drop=True)
    metadata = evaluation_metadata(targets, config.split_root, output)
    y = targets["iou_target"].to_numpy(np.float32)
    assert np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    folds = target_fold_indices(targets["fold"].to_numpy())
    base_columns = ["row_id", "iou_target", "fold", "detector_seen_during_training"]
    rows, predictions = [], targets[base_columns].to_dict("list")
    for method in dict.fromkeys(models):
        if method == "resnet50_frozen":
            path = output / "embeddings" / RESNET_CSV
            if reuse_existing:
                table = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
                assert table["row_id"].equals(targets["row_id"])
                x = table.drop(columns="row_id").to_numpy(np.float32)
            else:
                x = recompute_embeddings(targets, path.parent, csv_names=(RESNET_CSV,))[RESNET_CSV]
            result, predicted = cv_predict_with_folds("pre-detection", dataset, "fresnet50_roi", "pca256_ridge",
                                                     build_fdino_estimator(), x, y, folds)
            rows.append({**result.__dict__, **metadata})
            predictions["fresnet50_roi__pca256_ridge"] = predicted
        else:
            backbone = {"resnet50_finetuned": "resnet50", "dinov2_finetuned": "dinov2"}[method]
            directory = output / "finetuning" / backbone
            if reuse_existing:
                protocol = json.loads((directory / "protocol.json").read_text())
                assert protocol["status"] == "complete", "Fine-tuning did not complete all outer folds"
                table = pd.read_csv(directory / "oof_predictions.csv").sort_values("row_id").reset_index(drop=True)
                pd.testing.assert_frame_equal(targets[base_columns], table[base_columns], check_exact=False, rtol=1e-6)
                predicted = table["predicted_iou"].to_numpy(np.float32)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                protocol = {
                    "status": "training", "stage": "pre-detection", "backbone": MODEL_NAMES[backbone],
                    "backbone_updated": True, "inputs": "raw-image candidate ROI only; no detector features or masks",
                    "input_preprocessing": "same deterministic ROI and normalization as frozen embeddings; no augmentation",
                    "output": "one linear head followed by sigmoid; MSE loss",
                    "optimizer": "AdamW", "backbone_lr": 1e-5, "head_lr": 1e-3, "weight_decay": 0.01,
                    "max_epochs": epochs, "patience": patience, "batch_size": batch_size, "amp": "float16",
                    "seed": "42 + outer fold", "inner_validation_fraction": 0.2,
                    "selection": "lowest inner validation MSE; outer images never used for selection",
                    "refit": "none; selected checkpoint remains trained on 80% of the outer training portion",
                    "device": torch.cuda.get_device_name(), "folds": [], **metadata,
                }
                (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
                predicted = np.full(len(targets), np.nan, dtype=np.float32)
                for fold, (train, test) in folds.items():
                    ids, values, selection = fit_outer_fold(targets, train, test, backbone, directory / f"fold{fold}",
                                                            fold, epochs, patience, batch_size, workers)
                    assert np.array_equal(ids, targets.iloc[test]["row_id"].to_numpy())
                    predicted[test] = values
                    protocol["folds"].append(selection)
                    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
                table = targets[base_columns].copy()
                table["predicted_iou"] = predicted
                table.to_csv(directory / "oof_predictions.csv", index=False)
                protocol["status"] = "complete"
                (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
            assert np.isfinite(predicted).all()
            fold_r2 = [r2_score(y[test], predicted[test]) for _, test in folds.values()]
            result = metrics_row("pre-detection", dataset, method, "end_to_end", EMBEDDING_DIMS[backbone], y, predicted, fold_r2)
            rows.append({**result.__dict__, **metadata})
            predictions[f"{method}__end_to_end"] = predicted
        write_outputs(output, rows, predictions)
    return pd.DataFrame(rows).sort_values("r2", ascending=False)
