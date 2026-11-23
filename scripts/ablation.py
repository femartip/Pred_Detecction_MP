"""Cross-dataset transfer and held-out individual-feature analysis."""

from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.datasets import DATASETS
from scripts.evaluation_protocol import evaluation_metadata
from scripts.regression_utils import SEED, build_regressors, target_fold_indices


def load_post_data(dataset):
    source = Path("outputs/post_detection") / dataset
    targets = pd.read_csv(source / "iou_targets.csv").sort_values("row_id").reset_index(drop=True)
    features = pd.read_csv(source / "post_detection_features.csv").sort_values("row_id").reset_index(drop=True)
    reference = pd.read_csv(source / "oof_predictions.csv").sort_values("row_id").reset_index(drop=True)
    assert targets["row_id"].equals(features["row_id"])
    assert targets["row_id"].equals(reference["row_id"])
    assert targets["fold"].equals(reference["fold"])
    assert targets["detector_seen_during_training"].equals(reference["detector_seen_during_training"])
    np.testing.assert_allclose(targets["iou_target"], reference["iou_target"], rtol=1e-6)
    groups = [[column for column in features if column.startswith(prefix)] for prefix in ("conf_", "tta_", "mt_")]
    assert [len(columns) for columns in groups] == [11, 6, 12]
    columns = [column for family in groups for column in family]
    return targets, features.set_index("row_id"), reference["full__random_forest"].to_numpy(np.float32), columns


def run_cross_dataset_transfer():
    output = Path("outputs/transfer")
    output.mkdir(parents=True, exist_ok=True)
    data, feature_columns, exposure = {}, {}, {}
    for dataset, config in DATASETS.items():
        targets, features, reference, columns = load_post_data(dataset)
        feature_columns[dataset] = columns
        directory = output / dataset
        directory.mkdir(parents=True, exist_ok=True)
        exposure[dataset] = evaluation_metadata(targets, config.split_root, directory)
        data[dataset] = (targets, features, reference)

    columns = feature_columns["ca"]
    assert set(columns) == set(feature_columns["glass"]), "Feature families differ between datasets"
    rows = []
    for source, destination in (("ca", "glass"), ("glass", "ca")):
        train_targets, train_features, _ = data[source]
        targets, features, within_predictions = data[destination]
        x_train = train_features[columns].to_numpy(np.float32)
        x_test = features[columns].to_numpy(np.float32)
        y_train = train_targets["iou_target"].to_numpy(np.float32)
        y = targets["iou_target"].to_numpy(np.float32)
        assert all(np.isfinite(values).all() for values in (x_train, x_test, y_train, y, within_predictions))
        model = build_regressors()["random_forest"]
        print(f"[transfer] {source.upper()} -> {destination.upper()}: "
              f"fit {len(y_train)} source images, evaluate {len(y)} destination images", flush=True)
        model.fit(x_train, y_train)
        predictions = np.clip(model.predict(x_test).astype(np.float32), 0, 1)

        table = targets[["row_id", "fold", "detector_seen_during_training", "iou_target"]].copy()
        table["predicted_iou"] = predictions
        table["within_dataset_oof_iou"] = within_predictions
        table.to_csv(output / f"{source}_to_{destination}_predictions.csv", index=False)

        for setting, training_dataset, values in (
            ("cross_dataset_transfer", source, predictions),
            ("within_dataset_oof", destination, within_predictions),
        ):
            row = {
                "stage": "post-detection", "setting": setting,
                "source_dataset": training_dataset, "target_dataset": destination,
                "view": "full", "model": "random_forest", "n_features": len(columns),
                "n_train": len(y_train) if setting == "cross_dataset_transfer" else np.nan,
                "r2": float(r2_score(y, values)),
                "spearman": float(spearmanr(y, values).statistic),
                "mae": float(mean_absolute_error(y, values)),
                "rmse": float(np.sqrt(mean_squared_error(y, values))),
                "mean_target_iou": float(y.mean()),
                **exposure[destination],
            }
            for coverage in (20, 40):
                accepted = np.argsort(values)[::-1][:int(np.ceil(len(y) * coverage / 100))]
                row[f"accepted_iou_{coverage}"] = float(y[accepted].mean())
                row[f"gain_{coverage}"] = float(y[accepted].mean() - y.mean())
                row[f"accepted_n_{coverage}"] = len(accepted)
                row[f"actual_coverage_{coverage}"] = len(accepted) / len(y)
            rows.append(row)

    protocol = {
        "experiment": "bidirectional cross-dataset transfer of the quality regressor",
        "features": columns,
        "estimator": build_regressors()["random_forest"].get_params(),
        "training": "all source-dataset images; fixed hyperparameters; no destination fitting, tuning or recalibration",
        "preprocessing": "none; identical ordered raw feature columns in both datasets",
        "prediction_clipping": [0, 1],
        "target": "saved image-level union-mask IoU from each destination image's assigned detector",
        "detectors": "dataset-specific detectors retained; not zero-shot transfer of the segmentation pipeline",
        "within_dataset_reference": "existing full__random_forest five-fold OOF predictions; no refitting",
        "training_size_caveat": "transfer fits all source images; within-dataset reference fits four folds, so training sizes are not matched",
        "n_train": "source image count for transfer; missing for within-dataset OOF because training counts vary by fold",
        "coverage": "rank destination images by predicted IoU and retain ceil(n * coverage / 100)",
        "detector_exposure": exposure,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    return results



def run_individual_feature_analysis():
    output = Path("outputs/feature_analysis")
    output.mkdir(parents=True, exist_ok=True)
    records, baselines, metadata = [], [], {}
    for dataset, config in DATASETS.items():
        targets, features, reference, columns = load_post_data(dataset)
        directory = output / dataset
        directory.mkdir(parents=True, exist_ok=True)
        metadata[dataset] = evaluation_metadata(targets, config.split_root, directory)
        x = features[columns].to_numpy(np.float32)
        y = targets["iou_target"].to_numpy(np.float32)
        assert all(np.isfinite(values).all() for values in (x, y, reference))
        for fold, (train, test) in target_fold_indices(targets["fold"].to_numpy()).items():
            print(f"[features {dataset}] fold {fold}: 29 features, 10 held-out permutations each", flush=True)
            model = build_regressors()["random_forest"].fit(x[train], y[train])
            model.set_params(n_jobs=1)
            predictions = model.predict(x[test])
            np.testing.assert_allclose(predictions, reference[test], rtol=1e-5, atol=1e-6)
            baselines.append({"dataset": dataset, "fold": int(fold), "n_test": len(test),
                              "baseline_r2": float(r2_score(y[test], predictions))})
            importance = permutation_importance(
                model, x[test], y[test], scoring="r2", n_repeats=10,
                random_state=SEED + int(fold), n_jobs=1,
            )
            for feature, drops in zip(columns, importance.importances):
                for repeat, drop in enumerate(drops):
                    records.append({"dataset": dataset, "fold": int(fold), "feature": feature,
                                    "repeat": repeat, "r2_drop": float(drop)})
            pd.DataFrame(records).to_csv(output / "permutation_scores.csv", index=False)
            pd.DataFrame(baselines).to_csv(output / "fold_scores.csv", index=False)

    scores = pd.DataFrame(records)
    summary = scores.groupby(["dataset", "feature"], as_index=False).agg(
        importance_mean=("r2_drop", "mean"), importance_std=("r2_drop", "std"),
    ).sort_values(["dataset", "importance_mean"], ascending=[True, False])
    summary["rank"] = summary.groupby("dataset").cumcount() + 1
    summary.to_csv(output / "feature_importance.csv", index=False)
    protocol = {
        "method": "held-out permutation importance of the full 29-feature RF",
        "features": columns,
        "estimator": build_regressors()["random_forest"].get_params(),
        "folds": "same five within-dataset evaluation folds as the main experiments",
        "fit": "training folds only; fixed RF settings; no feature selection or tuning",
        "score": "held-out baseline R2 minus R2 after shuffling one feature",
        "repeats_per_fold": 10, "seed": "42 + fold",
        "aggregation": "equal-weight mean and sample standard deviation over five folds and ten repeats",
        "prediction_check": "refitted RF predictions must match saved full__random_forest OOF predictions",
        "parallelism": "RF fitting uses existing settings; predictions/permutations run sequentially",
        "interpretation": "marginal model reliance, not causal attribution; correlated features can share importance",
        "detector_exposure": metadata,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return summary


def run_ablation():
    transfer = run_cross_dataset_transfer()
    importance = run_individual_feature_analysis()
    print("\nCross-dataset transfer")
    print(transfer[["setting", "source_dataset", "target_dataset", "r2", "spearman", "gain_20"]].to_string(index=False))
    print("\nTop five features per dataset (held-out R2 drop)")
    print(importance.groupby("dataset", sort=False).head(5).to_string(index=False))


if __name__ == "__main__":
    run_ablation()
