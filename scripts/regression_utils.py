from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor

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


def build_regressors() -> dict[str, object]:
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
        "xgboost": XGBRegressor(
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
