from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

SEED = 42


@dataclass(frozen=True)
class Result:
    stage: str
    dataset: str
    view: str
    model: str
    n_features: int
    r2: float
    fold_r2_mean: float
    fold_r2_std: float
    mae: float
    rmse: float
    spearman: float


def target_fold_indices(folds: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    index = np.arange(len(folds))
    return {fold: (index[folds != fold], index[folds == fold]) for fold in sorted(np.unique(folds))}


def cv_predict_with_folds(
    stage: str,
    dataset: str,
    view: str,
    model_name: str,
    estimator: object,
    x: np.ndarray,
    y: np.ndarray,
    fold_indices: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[Result, np.ndarray]:
    pred = np.zeros_like(y, dtype=np.float32)
    folds = []
    for train_idx, val_idx in fold_indices.values():
        fitted = fit_with_safe_pca(estimator, x[train_idx], y[train_idx])
        pred[val_idx] = np.clip(np.asarray(fitted.predict(x[val_idx]), dtype=np.float32), 0.0, 1.0)
        folds.append(r2_score(y[val_idx], pred[val_idx]))
    return metrics_row(stage, dataset, view, model_name, x.shape[1], y, pred, folds), pred


def nested_stack_with_folds(
    stage: str,
    dataset: str,
    stack_name: str,
    stacker: object,
    specs: list[tuple[str, np.ndarray, object]],
    y: np.ndarray,
    fold_indices: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[Result, np.ndarray]:
    pred = np.zeros_like(y, dtype=np.float32)
    folds = []
    for outer_fold in sorted(fold_indices):
        outer_train, outer_val = fold_indices[outer_fold]
        train_meta = np.zeros((len(outer_train), len(specs)), dtype=np.float32)
        val_meta = np.zeros((len(outer_val), len(specs)), dtype=np.float32)
        pos = {int(idx): row for row, idx in enumerate(outer_train)}
        inner = KFold(n_splits=5, shuffle=True, random_state=SEED + 100 + outer_fold)

        for col, (_name, x, estimator) in enumerate(specs):
            x_train = x[outer_train]
            y_train = y[outer_train]
            for inner_train, inner_val in inner.split(x_train):
                model = fit_with_safe_pca(estimator, x_train[inner_train], y_train[inner_train])
                inner_pred = np.clip(model.predict(x_train[inner_val]), 0.0, 1.0)
                inner_ids = outer_train[inner_val]
                for idx, value in zip(inner_ids, inner_pred):
                    train_meta[pos[int(idx)], col] = float(value)

            model = fit_with_safe_pca(estimator, x_train, y_train)
            val_meta[:, col] = np.clip(model.predict(x[outer_val]), 0.0, 1.0)

        meta_model = clone(stacker)
        meta_model.fit(train_meta, y[outer_train])
        pred[outer_val] = np.clip(np.asarray(meta_model.predict(val_meta), dtype=np.float32), 0.0, 1.0)
        folds.append(r2_score(y[outer_val], pred[outer_val]))

    return metrics_row(stage, dataset, "nested_roi_stack", stack_name, len(specs), y, pred, folds), pred


def fit_with_safe_pca(estimator: object, x: np.ndarray, y: np.ndarray):
    fitted = clone(estimator)
    if isinstance(fitted, Pipeline):
        for name, step in fitted.named_steps.items():
            if isinstance(step, PCA):
                fitted.set_params(**{f"{name}__n_components": min(step.n_components, x.shape[0], x.shape[1])})
    fitted.fit(x, y)
    return fitted


def metrics_row(
    stage: str,
    dataset: str,
    view: str,
    model_name: str,
    n_features: int,
    y: np.ndarray,
    pred: np.ndarray,
    folds: Iterable[float],
) -> Result:
    folds = list(folds)
    sp = spearmanr(y, pred).statistic
    return Result(
        stage=stage,
        dataset=dataset,
        view=view,
        model=model_name,
        n_features=n_features,
        r2=float(r2_score(y, pred)),
        fold_r2_mean=float(np.mean(folds)),
        fold_r2_std=float(np.std(folds, ddof=1)),
        mae=float(mean_absolute_error(y, pred)),
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        spearman=float(0.0 if np.isnan(sp) else sp),
    )


def build_fdino_estimator() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=256, random_state=SEED)),
            ("ridge", RidgeCV(alphas=np.logspace(-5, 5, 31))),
        ]
    )


def build_fhc_estimator() -> Pipeline:
    return Pipeline(
        [
            ("scale", RobustScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-5, 5, 31))),
        ]
    )


def build_post_detection_estimators() -> dict[str, object]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required for the post-detection pipeline") from exc

    return {
        "ridge": Pipeline(
            [
                ("scale", RobustScaler()),
                ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=SEED,
        ),
    }
