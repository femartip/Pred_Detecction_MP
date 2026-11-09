from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .detectron2_inference import DetectorFoldConfig, build_predictor, run_predictor_instances

BASE_SCORE_THRESH = 0.5
LOW_SCORE_THRESH = 0.2
MT_THRESHOLDS = (0.2, 0.3, 0.5, 0.7)


def extract_post_detection_features(
    targets: pd.DataFrame,
    fold_configs: list[DetectorFoldConfig],
    output_csv: Path,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold, fold_targets in targets.groupby("fold", sort=True):
        base_predictor = build_predictor(fold_configs[int(fold)], score_thresh=BASE_SCORE_THRESH)
        low_predictor = build_predictor(fold_configs[int(fold)], score_thresh=LOW_SCORE_THRESH)
        for record in tqdm(
            fold_targets.itertuples(index=False),
            total=len(fold_targets),
            desc=f"[post fold {fold}]",
            unit="img",
        ):
            img_bgr = cv2.imread(record.image_path)
            if img_bgr is None:
                raise FileNotFoundError(record.image_path)
            rows.append({
                "row_id": int(record.row_id),
                **extract_confidence_features(img_bgr, base_predictor),
                **extract_tta_features(img_bgr, base_predictor),
                **extract_multi_threshold_features(img_bgr, low_predictor),
            })

    table = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    return table


def extract_confidence_features(img_bgr: np.ndarray, predictor) -> dict[str, float]:
    outputs = run_predictor_instances(img_bgr, predictor)
    n_inst = outputs["n_inst"]
    # Preserve the existing zero-score placeholder for empty detections.
    scores = outputs["scores_for_stats"]
    hist, _ = np.histogram(scores, bins=10, range=(0.0, 1.0))
    hist = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist * np.log(hist + 1e-9)))
    return {
        "conf_n_inst": float(n_inst),
        "conf_score_mean": float(scores.mean()),
        "conf_score_max": float(scores.max()),
        "conf_score_min": float(scores.min()),
        "conf_score_std": float(scores.std()),
        "conf_score_p25": float(np.percentile(scores, 25)),
        "conf_score_p75": float(np.percentile(scores, 75)),
        "conf_score_entropy": entropy,
        "conf_n_high_conf": float((scores >= 0.7).sum()),
        "conf_n_low_conf": float((scores < 0.5).sum()),
        "conf_pred_area": float(outputs["area"]),
    }


def extract_tta_features(img_bgr: np.ndarray, predictor) -> dict[str, float]:
    tta_masks = []
    tta_counts = []
    tta_areas = []
    for aug_img, inverse_fn in apply_tta(img_bgr):
        outputs = run_predictor_instances(aug_img, predictor)
        tta_masks.append(inverse_fn(outputs["mask"]))
        tta_counts.append(outputs["n_inst"])
        tta_areas.append(outputs["area"])

    pairwise_ious = [mask_iou(a, b) for a, b in combinations(tta_masks, 2)]
    # Historical feature name: this measures foreground occupancy, not agreement.
    agreement = float(np.mean(np.stack(tta_masks, axis=0), axis=0).mean())
    return {
        "tta_n_inst_std": float(np.std(tta_counts)),
        "tta_area_std": float(np.std(tta_areas)),
        "tta_area_mean": float(np.mean(tta_areas)),
        "tta_iou_mean": float(np.mean(pairwise_ious)),
        "tta_iou_std": float(np.std(pairwise_ious)),
        "tta_agreement": agreement,
    }


def extract_multi_threshold_features(img_bgr: np.ndarray, predictor) -> dict[str, float]:
    outputs = run_predictor_instances(img_bgr, predictor)
    scores = outputs["scores"]
    masks = outputs["masks"]
    feats: dict[str, float] = {}
    counts = []
    areas = []
    for thresh in MT_THRESHOLDS:
        keep = scores >= thresh
        n_inst = int(keep.sum())
        if n_inst == 0 or masks is None:
            area = 0.0
        else:
            area = float(masks[keep].any(axis=0).mean())
        suffix = int(thresh * 10)
        feats[f"mt_n_inst_{suffix}"] = float(n_inst)
        feats[f"mt_area_{suffix}"] = area
        counts.append(float(n_inst))
        areas.append(area)

    feats["mt_n_inst_range"] = float(max(counts) - min(counts))
    feats["mt_n_inst_std"] = float(np.std(counts))
    feats["mt_area_range"] = float(max(areas) - min(areas))
    feats["mt_area_std"] = float(np.std(areas))
    return feats


def apply_tta(img_bgr: np.ndarray):
    return [
        (img_bgr, lambda mask: mask),
        (cv2.flip(img_bgr, 1), lambda mask: cv2.flip(mask.astype(np.uint8), 1).astype(bool)),
        (cv2.flip(img_bgr, 0), lambda mask: cv2.flip(mask.astype(np.uint8), 0).astype(bool)),
        (
            cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
            lambda mask: cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool),
        ),
        (
            cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
            lambda mask: cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool),
        ),
    ]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return 1.0 if union == 0 else float(inter / union)
