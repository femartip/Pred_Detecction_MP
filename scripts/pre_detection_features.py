from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def crop_candidate_roi(image: Image.Image) -> Image.Image:
    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=45, sigmaY=45)
    corrected = gray - background
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    response = np.abs(corrected) + 0.25 * np.hypot(grad_x, grad_y)
    threshold = max(float(np.percentile(response, 98.8)), float(response.mean() + 2.0 * response.std()))
    mask = (response >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 20:
            keep[labels == label] = 1
    if keep.sum() == 0:
        return image

    ys, xs = np.where(keep > 0)
    h, w = keep.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    margin_y = max(48, int(0.25 * (y1 - y0)))
    margin_x = max(48, int(0.25 * (x1 - x0)))
    y0, y1 = max(0, y0 - margin_y), min(h, y1 + margin_y)
    x0, x1 = max(0, x0 - margin_x), min(w, x1 + margin_x)
    return image.crop((x0, y0, x1, y1))


def resize_short_side(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    if min(w, h) == size:
        return image
    scale = size / min(w, h)
    return image.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return image.crop((left, top, left + size, top + size))


def image_to_dino_tensor(image: Image.Image) -> torch.Tensor:
    image = resize_short_side(image, 560)
    image = center_crop(image, 518)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def candidate_models(n_features: int) -> dict[str, object]:
    alphas = np.logspace(-5, 5, 31)
    models: dict[str, object] = {"ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))}
    for n in (256, 512, 768):
        if n < min(n_features, 1800):
            models[f"pca{n}_ridge"] = make_pipeline(
                StandardScaler(),
                PCA(n_components=n, random_state=SEED),
                RidgeCV(alphas=alphas),
            )
    return models


def load_embedding(path: Path, row_ids: pd.Series) -> np.ndarray:
    df = pd.read_csv(path).sort_values("row_id")
    expected = row_ids.sort_values().to_numpy()
    if not np.array_equal(df["row_id"].to_numpy(), expected):
        raise ValueError(f"row_id mismatch in {path}")
    return df.drop(columns=["row_id"]).to_numpy(dtype=np.float32)

