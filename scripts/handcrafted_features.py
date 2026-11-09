from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor, sobel
from tqdm.auto import tqdm


PATCH_GRID = 4


def recompute_handcrafted_features(targets: pd.DataFrame, output_csv: Path) -> np.ndarray:
    rows: list[dict[str, float | int]] = []
    for record in tqdm(
        targets.itertuples(index=False),
        total=len(targets),
        desc="[handcrafted]",
        unit="img",
    ):
        with Image.open(record.image_path) as image:
            feats = extract_all_handcrafted(np.asarray(image.convert("RGB")))
        rows.append({"row_id": int(record.row_id), **feats})

    table = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    return table.drop(columns=["row_id"]).to_numpy(dtype=np.float32)


def extract_all_handcrafted(img_rgb: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats: dict[str, float] = {}
    feats.update(extract_intensity_features(gray))
    feats.update(extract_glcm_features(gray))
    feats.update(extract_lbp_features(gray))
    feats.update(extract_edge_features(gray))
    feats.update(extract_fft_features(gray))
    feats.update(extract_gabor_features(gray))
    feats.update(extract_hog_features(gray))
    feats.update(extract_patch_features(gray))
    return feats


def extract_intensity_features(gray: np.ndarray) -> dict[str, float]:
    flat = gray.flatten().astype(np.float32)
    return {
        "int_mean": float(flat.mean()),
        "int_std": float(flat.std()),
        "int_p5": float(np.percentile(flat, 5)),
        "int_p25": float(np.percentile(flat, 25)),
        "int_p50": float(np.percentile(flat, 50)),
        "int_p75": float(np.percentile(flat, 75)),
        "int_p95": float(np.percentile(flat, 95)),
        "int_skew": float(_safe_moment(flat, 3)),
        "int_kurt": float(_safe_moment(flat, 4) - 3.0),
        "int_range": float(flat.max() - flat.min()),
    }


def extract_glcm_features(gray: np.ndarray) -> dict[str, float]:
    g64 = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(
        g64,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    feats = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop).flatten()
        feats[f"glcm_{prop}_mean"] = float(vals.mean())
        feats[f"glcm_{prop}_std"] = float(vals.std())
    return feats


def extract_lbp_features(gray: np.ndarray, n_points: int = 24, radius: int = 3, n_bins: int = 32) -> dict[str, float]:
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_points + 2), density=True)
    return {f"lbp_{i}": float(v) for i, v in enumerate(hist)}


def extract_edge_features(gray: np.ndarray) -> dict[str, float]:
    edge = sobel(gray.astype(np.float32) / 255.0).flatten()
    return {
        "edge_mean": float(edge.mean()),
        "edge_std": float(edge.std()),
        "edge_ratio": float((edge > 0.05).mean()),
        "edge_p75": float(np.percentile(edge, 75)),
        "edge_p95": float(np.percentile(edge, 95)),
    }


def extract_fft_features(gray: np.ndarray) -> dict[str, float]:
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))) ** 2
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = float(mag.sum()) + 1e-9
    low = float(mag[radius <= min(h, w) * 0.08].sum())
    mid = float(mag[(radius > min(h, w) * 0.08) & (radius <= min(h, w) * 0.22)].sum())
    high = float(mag[radius > min(h, w) * 0.22].sum())
    return {
        "fft_low_ratio": float(low / total),
        "fft_mid_ratio": float(mid / total),
        "fft_high_ratio": float(high / total),
        "fft_log_energy": float(np.log1p(total)),
    }


def extract_gabor_features(gray: np.ndarray) -> dict[str, float]:
    img_f = gray.astype(np.float32) / 255.0
    feats = {}
    for freq in (0.1, 0.3):
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            real, _ = gabor(img_f, frequency=freq, theta=theta)
            key = f"gabor_f{freq:.1f}_t{theta:.2f}"
            feats[f"{key}_mean"] = float(real.mean())
            feats[f"{key}_std"] = float(real.std())
    return feats


def extract_hog_features(gray: np.ndarray) -> dict[str, float]:
    resized = cv2.resize(gray, (128, 128))
    fd = hog(
        resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    ).reshape(-1, 9)
    feats = {f"hog_bin{b}_mean": float(fd[:, b].mean()) for b in range(9)}
    feats.update({f"hog_bin{b}_std": float(fd[:, b].std()) for b in range(9)})
    feats["hog_energy"] = float((fd**2).sum())
    return feats


def extract_patch_features(gray: np.ndarray, n: int = PATCH_GRID) -> dict[str, float]:
    h, w = gray.shape
    ph, pw = h // n, w // n
    feats = {}
    patch_means: list[float] = []
    patch_stds: list[float] = []
    patch_edges: list[float] = []
    for r in range(n):
        for c in range(n):
            patch = gray[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw].astype(np.float32)
            pmean = float(patch.mean())
            pstd = float(patch.std())
            pedge = float(sobel(patch / 255.0).mean())
            patch_means.append(pmean)
            patch_stds.append(pstd)
            patch_edges.append(pedge)
            feats[f"patch_{r}{c}_mean"] = pmean
            feats[f"patch_{r}{c}_std"] = pstd
            feats[f"patch_{r}{c}_edge"] = pedge
    feats["patch_mean_var"] = float(np.var(patch_means))
    feats["patch_std_var"] = float(np.var(patch_stds))
    feats["patch_edge_var"] = float(np.var(patch_edges))
    feats["patch_mean_cv"] = float(np.std(patch_means) / (np.mean(patch_means) + 1e-9))
    return feats


def _safe_moment(values: np.ndarray, order: int) -> float:
    centered = values - values.mean()
    scale = values.std()
    if scale < 1e-9:
        return 0.0
    return float(np.mean((centered / scale) ** order))
