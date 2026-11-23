"Calles inference"
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DetectorFoldConfig:
    config_path: Path
    checkpoint_path: Path
    test_coco_path: Path


def build_iou_targets(
    image_root: Path,
    fold_configs: list[DetectorFoldConfig],
    output_dir: Path,
    score_thresh: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for fold, fold_config in enumerate(fold_configs):
        print(f"[detector] fold {fold}: {fold_config.checkpoint_path}", flush=True)
        split_coco = json.loads(fold_config.test_coco_path.read_text())
        predictions = run_fold_inference(image_root, fold_config, output_dir / f"fold{fold}", score_thresh=score_thresh)

        gt_by_image: dict[int, list[list[float]]] = {}
        for ann in split_coco["annotations"]:
            gt_by_image.setdefault(int(ann["image_id"]), []).extend(ann.get("segmentation", []))

        pred_by_image: dict[int, list[dict[str, object]]] = {}
        for pred in predictions:
            pred_by_image.setdefault(int(pred["image_id"]), []).append(pred["segmentation"])

        for image in split_coco["images"]:
            row_id = int(image["id"])
            width = int(image["width"])
            height = int(image["height"])
            rows.append(
                {
                    "row_id": row_id,
                    "image_path": str(image_root / Path(str(image["file_name"])).name),
                    "fold": fold,
                    "detector_seen_during_training": bool(image["detector_seen_during_training"]),
                    "iou_target": union_mask_iou(
                        gt_by_image.get(row_id, []),
                        pred_by_image.get(row_id, []),
                        height,
                        width,
                    ),
                }
            )

    targets = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    targets.to_csv(output_dir / "iou_targets.csv", index=False)
    return targets


def run_fold_inference(
    image_root: Path,
    fold_config: DetectorFoldConfig,
    output_dir: Path,
    score_thresh: float = 0.5,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor(fold_config, score_thresh=score_thresh)

    split_coco = json.loads(fold_config.test_coco_path.read_text())
    predictions: list[dict[str, object]] = []
    for image in tqdm(split_coco["images"], desc=f"[detector] {output_dir.name}"):
        image_id = int(image["id"])
        image_path = image_root / Path(str(image["file_name"])).name
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(image_path)
        instances = predictor(bgr)["instances"].to("cpu")
        if len(instances) == 0:
            continue

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy().astype(np.uint8)
        for box, score, category_id, mask in zip(boxes, scores, classes, masks):
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            x0, y0, x1, y1 = box.tolist()
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(score),
                    "segmentation": rle,
                }
            )

    (output_dir / "coco_instances_results.json").write_text(json.dumps(predictions))
    return predictions


def build_predictor(fold_config: DetectorFoldConfig, score_thresh: float):
    if not torch.cuda.is_available():
        raise RuntimeError("Mask R-CNN inference requires working CUDA; CPU fallback is disabled.")
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(str(fold_config.config_path))
    cfg.MODEL.WEIGHTS = str(fold_config.checkpoint_path)
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return DefaultPredictor(cfg)


def run_predictor_instances(bgr: np.ndarray, predictor) -> dict[str, object]:
    instances = predictor(bgr)["instances"].to("cpu")
    n_inst = len(instances)
    if n_inst == 0:
        zero_mask = np.zeros(bgr.shape[:2], dtype=bool)
        return {
            "n_inst": 0,
            "scores": np.array([], dtype=np.float32),
            "scores_for_stats": np.array([0.0], dtype=np.float32),
            "masks": None,
            "mask": zero_mask,
            "area": 0.0,
        }

    scores = instances.scores.numpy().astype(np.float32)
    masks = instances.pred_masks.numpy().astype(bool)
    union_mask = masks.any(axis=0)
    return {
        "n_inst": n_inst,
        "scores": scores,
        "scores_for_stats": scores,
        "masks": masks,
        "mask": union_mask,
        "area": float(union_mask.mean()),
    }


def union_mask_iou(polygons: list[list[float]], pred_rles: list[dict[str, object]], height: int, width: int) -> float:
    gt_mask = polygons_to_mask(polygons, height, width)
    pred_mask = predictions_to_mask(pred_rles, height, width)
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return 1.0 if union == 0 else float(inter / union)


def polygons_to_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def predictions_to_mask(pred_rles: list[dict[str, object]], height: int, width: int) -> np.ndarray:
    if not pred_rles:
        return np.zeros((height, width), dtype=bool)
    decoded = mask_utils.decode(pred_rles)
    if decoded.ndim == 2:
        return decoded.astype(bool)
    return decoded.any(axis=2).astype(bool)
