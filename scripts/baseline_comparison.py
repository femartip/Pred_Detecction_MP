"""Fold-calibrated scalar quality baselines, with optional ensemble consistency."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import combinations
from multiprocessing import get_context
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from .datasets import DATASETS, fold_detector_configs
from .detectron2_inference import build_predictor, polygons_to_mask, predictions_to_mask, run_predictor_instances
from .evaluation_protocol import evaluation_metadata
from .post_detection_features import apply_tta, mask_iou
from .regression_utils import build_regressors, metrics_row, target_fold_indices

SEED = 42
LEVELS = 16
PIXELS_PER_CLASS = 2048
REFERENCES_PER_FOLD = 5
WORKERS = 8
DROPOUT_PROBABILITY = 0.3
DROPOUT_PASSES = 10
cv2.setNumThreads(1)
CODEBOOK = np.array(np.unravel_index(np.arange(LEVELS ** 3), (LEVELS,) * 3)).T.astype(np.float32)


def pixel_codes(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    features = np.stack([gray, cv2.blur(gray, (5, 5)), cv2.blur(gray, (21, 21))], axis=-1)
    bins = np.minimum((features * LEVELS).astype(np.int32), LEVELS - 1)
    return np.ravel_multi_index(bins.reshape(-1, 3).T, (LEVELS,) * 3)


def initialize_references(foreground, background, pools):
    global REFERENCE_FOREGROUND, REFERENCE_BACKGROUND, REFERENCE_POOLS
    cv2.setNumThreads(1)
    REFERENCE_FOREGROUND = foreground
    REFERENCE_BACKGROUND = background
    REFERENCE_POOLS = pools


def rca_score(job):
    row_id, fold, image_path, height, width, rles = job
    start = perf_counter()
    codes = pixel_codes(image_path)
    pseudo_labels = predictions_to_mask(rles, height, width).ravel()
    rng = np.random.default_rng(SEED + row_id)
    if pseudo_labels.any() and not pseudo_labels.all():
        sampled = np.concatenate([
            rng.choice(indices, min(PIXELS_PER_CLASS, len(indices)), replace=False)
            for indices in (np.flatnonzero(~pseudo_labels), np.flatnonzero(pseudo_labels))
        ])
        classifier = RandomForestClassifier(
            n_estimators=50, max_depth=30, class_weight="balanced", random_state=SEED,
            n_jobs=1,
        ).fit(CODEBOOK[codes[sampled]], pseudo_labels[sampled])
        predicted = classifier.predict(CODEBOOK)
    else:
        predicted = np.full(len(CODEBOOK), pseudo_labels.all())
    # Histogram counts give exact full-resolution IoUs for these quantized features.
    intersection = REFERENCE_FOREGROUND @ predicted
    union = REFERENCE_FOREGROUND.sum(axis=1) + REFERENCE_BACKGROUND @ predicted
    ious = np.divide(intersection, union, out=np.ones_like(intersection), where=union != 0)
    values = ious[REFERENCE_POOLS[:, fold]].max(axis=1)
    return row_id, values, perf_counter() - start


def extract_dropout_scores(targets, fold_configs, output):
    import torch
    from torch.nn import functional as F

    rows = []
    for fold, group in targets.groupby("fold", sort=True):
        predictor = build_predictor(fold_configs[int(fold)], score_thresh=0.5)
        head = predictor.model.roi_heads.box_head
        relus = [layer for name, layer in head.named_children() if name.startswith("fc_relu")]
        assert len(relus) == 2, "Expected the two-FC Mask R-CNN box head"

        def inject_dropout(module, inputs, output):
            return F.dropout(output, p=DROPOUT_PROBABILITY, training=True)

        # Only these activations are stochastic; the detector remains in eval mode.
        handles = [layer.register_forward_hook(inject_dropout) for layer in relus]
        try:
            for row in tqdm(group.itertuples(index=False), total=len(group), desc=f"[dropout fold {fold}]", unit="img"):
                image = cv2.imread(row.image_path)
                if image is None:
                    raise FileNotFoundError(row.image_path)
                torch.manual_seed(SEED + int(row.row_id))
                start = perf_counter()
                masks = [run_predictor_instances(image, predictor)["mask"] for _ in range(DROPOUT_PASSES)]
                score = np.mean([mask_iou(a, b) for a, b in combinations(masks, 2)])
                rows.append({"row_id": int(row.row_id), "dropout_injection": float(score),
                             "dropout_seconds": perf_counter() - start,
                             "dropout_device": str(next(predictor.model.parameters()).device)})
        finally:
            for handle in handles:
                handle.remove()
        del predictor
    table = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    table.to_csv(output / "dropout_scores.csv", index=False)
    return table


def extract_ensemble_scores(targets, members_by_fold, output):
    rows = []
    for fold, group in targets.groupby("fold", sort=True):
        predictors = [build_predictor(member, score_thresh=0.5) for member in members_by_fold[int(fold)]]
        for row in tqdm(group.itertuples(index=False), total=len(group), desc=f"[ensemble fold {fold}]", unit="img"):
            image = cv2.imread(row.image_path)
            if image is None:
                raise FileNotFoundError(row.image_path)
            start = perf_counter()
            masks = [run_predictor_instances(image, predictor)["mask"] for predictor in predictors]
            score = np.mean([mask_iou(a, b) for a, b in combinations(masks, 2)])
            rows.append({"row_id": int(row.row_id), "ensemble_consistency": float(score),
                         "ensemble_seconds": perf_counter() - start,
                         "ensemble_device": str(next(predictors[0].model.parameters()).device)})
        del predictors
    table = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    table.to_csv(output / "ensemble_scores.csv", index=False)
    return table


def extract_image_baseline_scores(targets, fold_configs, selected, output):
    rows = []
    for fold, group in targets.groupby("fold", sort=True):
        predictor = build_predictor(fold_configs[int(fold)], score_thresh=0.5)
        for row in tqdm(group.itertuples(index=False), total=len(group), desc=f"[baseline timing fold {fold}]", unit="img"):
            image = cv2.imread(row.image_path)
            if image is None:
                raise FileNotFoundError(row.image_path)
            record = {"row_id": int(row.row_id)}
            if "confidence" in selected:
                start = perf_counter()
                prediction = run_predictor_instances(image, predictor)
                record["detection_confidence"] = float(prediction["scores_for_stats"].mean())
                record["confidence_seconds"] = perf_counter() - start
                record["confidence_device"] = str(next(predictor.model.parameters()).device)
            if "tta" in selected:
                start = perf_counter()
                masks = [inverse(run_predictor_instances(view, predictor)["mask"]) for view, inverse in apply_tta(image)]
                record["tta_consistency"] = float(np.mean([mask_iou(a, b) for a, b in combinations(masks, 2)]))
                record["tta_seconds"] = perf_counter() - start
                record["tta_device"] = str(next(predictor.model.parameters()).device)
            rows.append(record)
        del predictor
    table = pd.DataFrame(rows).sort_values("row_id").set_index("row_id")
    table.to_csv(output / "image_baseline_scores.csv")
    return table


def extract_rca_scores(targets, config, source, output, scores, workers):
    coco = json.loads((config.dataset_root / "coco_format.json").read_text())
    images = {image["id"]: image for image in coco["images"]}
    polygons = defaultdict(list)
    for annotation in coco["annotations"]:
        polygons[annotation["image_id"]].extend(annotation["segmentation"])

    protocol = {
        "method": "2D Random Forest adaptation of RCA; maximum reference IoU followed by fold-trained Ridge calibration",
        "source": "https://doi.org/10.1109/TMI.2017.2665165",
        "pixel_features": ["grayscale", "5x5 mean", "21x21 mean"],
        "levels_per_feature": LEVELS, "pixels_per_class": PIXELS_PER_CLASS,
        "classifier": {"n_estimators": 50, "max_depth": 30, "class_weight": "balanced", "random_state": SEED},
        "mask_resolution": "original; same polygon rasterization as target IoU",
        "reference_selection": "5 images sampled from each other evaluation fold, seed 42 + held-out fold",
        "training_references": "20 images balanced across folds excluding both the outer test fold and the scored image's own fold; seed 142 + 5*outer_fold + image_fold",
        "pixel_sampling_seed": "42 + row_id",
        "empty_prediction": "constant-background reverse classifier; zero IoU on annotated references",
        "single_class_prediction": "constant-class reverse classifier",
        "rca_seconds": "per-image worker time, including image features, reverse training and reference scoring; excludes reference setup",
        "references": {},
        "calibration": {"estimator": "RobustScaler + RidgeCV", "alphas": [0.01, 0.1, 1, 10, 100],
                        "cv": 3, "prediction_clipping": [0, 1], "selected_alphas": {}},
    }
    groups = {int(fold): group for fold, group in targets.groupby("fold", sort=True)}
    for outer in groups:
        protocol["references"][str(outer)] = {}
        for fold in groups:
            rng = np.random.default_rng(SEED + outer if fold == outer else SEED + 100 + 5 * outer + fold)
            available = [k for k in groups if k not in (outer, fold)]
            total = 4 * REFERENCES_PER_FOLD
            reference_ids = [
                int(row_id)
                for position, other in enumerate(available)
                for row_id in rng.choice(groups[other]["row_id"].to_numpy(),
                                         total // len(available) + (position < total % len(available)), replace=False)
            ]
            assert not set(reference_ids).intersection(targets.loc[targets["fold"].isin([outer, fold]), "row_id"])
            protocol["references"][str(outer)][str(fold)] = reference_ids

    reference_ids = sorted({i for outer in protocol["references"].values() for ids in outer.values() for i in ids})
    positions = {row_id: position for position, row_id in enumerate(reference_ids)}
    pools = np.array([
        [[positions[i] for i in protocol["references"][str(outer)][str(fold)]] for fold in groups]
        for outer in groups
    ])
    foreground, background = [], []
    print(f"[RCA {dataset}] preparing {len(reference_ids)} reference images", flush=True)
    for row_id in tqdm(reference_ids, desc=f"[RCA {dataset} references]", unit="img"):
        image = images[row_id]
        codes = pixel_codes(config.image_root / Path(image["file_name"]).name)
        mask = polygons_to_mask(polygons[row_id], image["height"], image["width"]).ravel()
        foreground.append(np.bincount(codes[mask], minlength=len(CODEBOOK)))
        background.append(np.bincount(codes[~mask], minlength=len(CODEBOOK)))
    predictions = defaultdict(list)
    for fold in groups:
        for prediction in json.loads((source / f"fold{fold}/coco_instances_results.json").read_text()):
            predictions[prediction["image_id"]].append(prediction["segmentation"])
    jobs = [
        (int(row.row_id), int(row.fold), row.image_path, images[row.row_id]["height"], images[row.row_id]["width"], predictions[row.row_id])
        for row in targets.itertuples(index=False)
    ]
    rca_inputs = pd.DataFrame(index=scores.index, columns=[f"outer{fold}" for fold in groups], dtype=float)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=initialize_references,
        initargs=(np.asarray(foreground, dtype=np.float64), np.asarray(background, dtype=np.float64), pools),
    ) as pool:
        for row_id, values, seconds in tqdm(pool.map(rca_score, jobs), total=len(jobs), desc=f"[RCA {dataset}]", unit="img"):
            rca_inputs.loc[row_id] = values
            scores.loc[row_id, ["rca_rf", "rca_seconds"]] = [values[int(scores.loc[row_id, "fold"])], seconds]
    rca_inputs.to_csv(output / "rca_calibration_inputs.csv", index_label="row_id")
    scores["rca_device"] = "cpu"

    return rca_inputs, protocol


def run_baseline_comparison(dataset, workers=WORKERS, baselines=None, reuse_existing=False):
    names = {
        "confidence": "detection_confidence", "tta": "tta_consistency",
        "rca": "rca_rf", "dropout": "dropout_injection", "ensemble": "ensemble_consistency",
    }
    selected = list(dict.fromkeys(baselines if baselines is not None else ("confidence", "tta", "rca", "dropout")))
    methods = [names[name] for name in selected]
    config = DATASETS[dataset]
    fold_configs = fold_detector_configs(config)
    needs_detector = set(selected).intersection(("confidence", "tta", "dropout", "ensemble"))
    if not reuse_existing and needs_detector:
        for fold in fold_configs:
            for path in (fold.config_path, fold.checkpoint_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing detector input: {path}")
    source = Path("outputs/post_detection") / dataset
    output = Path("outputs/comparison") / dataset
    output.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(source / "iou_targets.csv").sort_values("row_id").reset_index(drop=True)
    metadata = evaluation_metadata(targets, config.split_root, output)
    scores = targets[["row_id", "fold", "detector_seen_during_training", "iou_target"]].set_index("row_id", drop=False)
    if (output / "quality_scores.csv").exists():
        saved = pd.read_csv(output / "quality_scores.csv").sort_values("row_id").set_index("row_id", drop=False)
        # Preserve unselected baselines only when they describe the same evaluation.
        pd.testing.assert_frame_equal(scores, saved[scores.columns], check_exact=False, rtol=1e-12, atol=1e-12)
        scores = saved

    if not reuse_existing and set(selected).intersection(("confidence", "tta")):
        timed = extract_image_baseline_scores(targets, fold_configs, selected, output)
        assert timed.index.equals(scores.index)
        scores[timed.columns] = timed

    protocol = {"calibration": {
        "estimator": "RobustScaler + RidgeCV", "alphas": [0.01, 0.1, 1, 10, 100],
        "cv": 3, "prediction_clipping": [0, 1], "selected_alphas": {},
    }}
    if "rca" in selected:
        if reuse_existing:
            rca_inputs = pd.read_csv(output / "rca_calibration_inputs.csv").set_index("row_id")
            assert rca_inputs.index.equals(scores.index)
            protocol = json.loads((output / "rca_protocol.json").read_text())
        else:
            rca_inputs, protocol = extract_rca_scores(targets, config, source, output, scores, workers)
    if "dropout" in selected and not reuse_existing:
        dropout = extract_dropout_scores(targets, fold_configs, output)
        assert dropout["row_id"].equals(targets["row_id"])
        scores[dropout.columns.drop("row_id")] = dropout.set_index("row_id")
    if "ensemble" in selected and not reuse_existing:
        from .deep_ensemble import train_deep_ensemble

        members_by_fold = train_deep_ensemble(config, fold_configs)
        ensemble_scores = extract_ensemble_scores(targets, members_by_fold, output)
        assert ensemble_scores["row_id"].equals(targets["row_id"])
        scores[ensemble_scores.columns.drop("row_id")] = ensemble_scores.set_index("row_id")

    y = scores["iou_target"].to_numpy(np.float32)
    folds = target_fold_indices(scores["fold"].to_numpy())
    results = []
    calibrator = build_regressors()["ridge"]
    for method in methods:
        values = np.zeros_like(y)
        fit_seconds, predict_seconds = 0.0, 0.0
        protocol["calibration"]["selected_alphas"][method] = {}
        for outer, (train, test) in folds.items():
            raw = rca_inputs[f"outer{outer}"].to_numpy(np.float32) if method == "rca_rf" else scores[method].to_numpy(np.float32)
            assert np.isfinite(raw).all()
            start = perf_counter()
            model = clone(calibrator).fit(raw[train, None], y[train])
            fit_seconds += perf_counter() - start
            start = perf_counter()
            values[test] = np.clip(model.predict(raw[test, None]), 0, 1)
            predict_seconds += perf_counter() - start
            protocol["calibration"]["selected_alphas"][method][str(outer)] = float(model["ridge"].alpha_)
        scores[f"{method}_calibrated"] = values
        fold_r2 = [r2_score(y[test], values[test]) for _, test in folds.values()]
        result = asdict(metrics_row("post-detection", dataset, method, "ridge_calibrated", 1, y, values, fold_r2))
        timing_column, passes, scope = {
            "detection_confidence": ("confidence_seconds", 1, "one detector pass and mean confidence"),
            "tta_consistency": ("tta_seconds", 5, "five detector passes, transforms and pairwise mask scoring"),
            "rca_rf": ("rca_seconds", 0, "CPU worker time for reverse features, training and scoring; excludes detector and reference setup"),
            "dropout_injection": ("dropout_seconds", 10, "ten detector passes and pairwise mask scoring"),
            "ensemble_consistency": ("ensemble_seconds", 3, "three detector passes and pairwise mask scoring"),
        }[method]
        times = scores[timing_column] if timing_column in scores else pd.Series(np.nan, index=scores.index)
        measured = np.isfinite(times).all()
        device_column = timing_column.replace("_seconds", "_device")
        devices = ", ".join(sorted(scores[device_column].dropna().unique())) if device_column in scores else ("cpu" if method == "rca_rf" else "not recorded")
        result.update({
            "mean_target_iou": float(y.mean()),
            "scoring_seconds_mean": float(times.mean()) if measured else np.nan,
            "scoring_seconds_median": float(times.median()) if measured else np.nan,
            "scoring_seconds_std": float(times.std(ddof=0)) if measured else np.nan,
            "scoring_seconds_sum": float(times.sum()) if measured else np.nan,
            "detector_passes_timed": passes if measured else np.nan,
            "timing_scope": scope if measured else "not measured; score read from saved features",
            "timing_device": devices if measured else "not measured",
            "timing_excludes": "model loading, image reading and shared setup; first-image warmup is not excluded",
            "calibration_fit_seconds": fit_seconds,
            "calibration_predict_seconds_per_image": predict_seconds / len(y),
        })
        if method == "ensemble_consistency":
            if reuse_existing:
                saved_ensemble = json.loads((output / "ensemble_protocol.json").read_text())
                assert saved_ensemble["target_detector_is_member"], "Saved ensemble excludes the target detector; recompute this baseline."
                training_seconds = saved_ensemble["training_seconds_total"]
            else:
                training_seconds = sum(json.loads((m.checkpoint_path.parent / "training.json").read_text())["training_seconds"]
                                       for members in members_by_fold for m in members[1:])
            result["training_seconds_total"] = training_seconds
        for coverage in (20, 40):
            accepted = np.argsort(values)[::-1][:int(np.ceil(len(y) * coverage / 100))]
            result[f"accepted_iou_{coverage}"] = float(y[accepted].mean())
            result[f"gain_{coverage}"] = float(y[accepted].mean() - y.mean())
            result[f"accepted_n_{coverage}"] = len(accepted)
            result[f"actual_coverage_{coverage}"] = len(accepted) / len(y)
        results.append({**result, **metadata})
    table = pd.DataFrame(results)
    if (output / "results.csv").exists():
        saved = pd.read_csv(output / "results.csv")
        table = pd.concat([saved.loc[~saved["view"].isin(methods)], table], ignore_index=True)
    scores.to_csv(output / "quality_scores.csv", index=False)
    if "dropout" in selected:
        if reuse_existing:
            dropout_protocol = json.loads((output / "dropout_protocol.json").read_text())
        else:
            dropout_protocol = {
                "method": "post-hoc dropout injection; checkpoints trained without dropout",
                "source": "https://doi.org/10.1016/j.ins.2023.119356",
                "placement": "after fc_relu1 and fc_relu2 of the ROI box head; all other modules in eval mode",
                "probability": DROPOUT_PROBABILITY, "passes": DROPOUT_PASSES,
                "scaling": "PyTorch inverted dropout (retained activations divided by 1-p)",
                "seed": "42 + row_id, set once before the ten passes",
                "score_threshold": 0.5,
                "raw_score": "mean of 45 pairwise native-resolution union-mask IoUs",
                "empty_union_iou": 1.0,
                "target": "existing deterministic detector union-mask IoU, not the stochastic mean prediction",
                "calibration": "same fold-trained RobustScaler + RidgeCV as other scalar baselines",
                "timing": "ten full detector passes and pairwise scoring; excludes model loading and image reading",
            }
        dropout_protocol["selected_alphas"] = protocol["calibration"]["selected_alphas"].pop("dropout_injection")
        (output / "dropout_protocol.json").write_text(json.dumps(dropout_protocol, indent=2) + "\n")
    if "ensemble" in selected:
        if reuse_existing:
            ensemble_protocol = json.loads((output / "ensemble_protocol.json").read_text())
            ensemble_protocol["selected_alphas"] = protocol["calibration"]["selected_alphas"].pop("ensemble_consistency")
        else:
            ensemble_protocol = {
                "method": "original fold detector plus two independently trained detectors; mask consistency, not probability averaging",
                "source": "https://www.sciencedirect.com/science/article/pii/S1361841524003177",
                "members": [[{"config": str(m.config_path), "checkpoint": str(m.checkpoint_path)} for m in members] for members in members_by_fold],
                "training": "two new detectors per fold on the mapped HF training partition, seeds 42/43; independent fine-tuning from COCO pretraining; original detector unchanged",
                "target_detector_is_member": True,
                "training_seconds_total": training_seconds,
                "training_cost_scope": "two new members per fold; excludes the historical cost of training the original detector",
                "score_threshold": 0.5,
                "raw_score": "mean of three pairwise native-resolution union-mask IoUs",
                "empty_union_iou": 1.0,
                "target": "existing deterministic detector union-mask IoU; original detector is ensemble member 0",
                "calibration": "same fold-trained RobustScaler + RidgeCV as other scalar baselines",
                "selected_alphas": protocol["calibration"]["selected_alphas"].pop("ensemble_consistency"),
                "timing": "three full detector passes and pairwise scoring; excludes model loading and image reading",
            }
        (output / "ensemble_protocol.json").write_text(json.dumps(ensemble_protocol, indent=2) + "\n")
    if "rca" in selected:
        (output / "rca_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    table.to_csv(output / "results.csv", index=False)
    return table
