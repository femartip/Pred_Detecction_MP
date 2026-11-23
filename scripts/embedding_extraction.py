"""Frozen image embeddings and shared raw-image ROI preprocessing."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from tqdm.auto import tqdm

RESNET_CSV = "resnet50_imagenet_roi224.csv"
MODEL_NAMES = {"resnet50": "resnet50.tv2_in1k", "dinov2": "vit_base_patch14_dinov2"}
EMBEDDING_DIMS = {"resnet50": 2048, "dinov2": 768}


def build_embedding_model(backbone: str, num_classes: int = 0):
    options = {"dynamic_img_size": True} if backbone == "dinov2" else {}
    return timm.create_model(MODEL_NAMES[backbone], pretrained=True, num_classes=num_classes, **options)


def recompute_embeddings(
    targets: pd.DataFrame,
    embedding_dir: Path,
    csv_names: tuple[str, ...] = ("dinov2_vitb14_cv2_roi518.csv", "dinov2_vitb14.csv"),
) -> dict[str, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("Embedding extraction requires working CUDA; CPU fallback is disabled.")
    device = torch.device("cuda")
    embedding_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "dinov2_vitb14_cv2_roi518.csv": ("dinov2", True, "cv2_roi"),
        "dinov2_vitb14.csv": ("dinov2", False, "vitb14"),
        RESNET_CSV: ("resnet50", True, "resnet50_roi"),
    }
    result = {}
    with torch.inference_mode():
        for backbone in dict.fromkeys(specs[name][0] for name in csv_names):
            print(f"[embeddings] loading {MODEL_NAMES[backbone]} on {device}", flush=True)
            model = build_embedding_model(backbone).to(device).eval()
            transform = image_to_dino_tensor if backbone == "dinov2" else image_to_resnet_tensor
            for csv_name in csv_names:
                name, roi, prefix = specs[csv_name]
                if name != backbone:
                    continue
                rows = []
                for start in tqdm(range(0, len(targets), 16), desc=f"[embeddings] {csv_name}"):
                    batch = targets.iloc[start:start + 16]
                    tensors = []
                    for record in batch.itertuples(index=False):
                        with Image.open(record.image_path) as image:
                            image = image.convert("RGB")
                            if roi:
                                image = crop_candidate_roi(image)
                            tensors.append(transform(image).squeeze(0))
                    values = model(torch.stack(tensors).to(device)).cpu().numpy().astype(np.float32)
                    rows.extend([int(row_id), *value.tolist()] for row_id, value in zip(batch["row_id"], values))
                columns = ["row_id", *[f"{prefix}_{i:04d}" for i in range(EMBEDDING_DIMS[backbone])]]
                table = pd.DataFrame(rows, columns=columns).sort_values("row_id").reset_index(drop=True)
                table.to_csv(embedding_dir / csv_name, index=False)
                result[csv_name] = table.drop(columns="row_id").to_numpy(np.float32)
            del model
    return result


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


def resize_short_side(image: Image.Image, size: int, interpolation=Image.Resampling.BICUBIC) -> Image.Image:
    w, h = image.size
    if min(w, h) == size:
        return image
    scale = size / min(w, h)
    return image.resize((round(w * scale), round(h * scale)), interpolation)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return image.crop((left, top, left + size, top + size))


def image_to_dino_tensor(image: Image.Image) -> torch.Tensor:
    return image_to_tensor(image, 560, 518, Image.Resampling.BICUBIC)


def image_to_resnet_tensor(image: Image.Image) -> torch.Tensor:
    return image_to_tensor(image, 232, 224, Image.Resampling.BILINEAR)


def image_to_tensor(image, short_side, crop_size, interpolation):
    image = center_crop(resize_short_side(image, short_side, interpolation), crop_size)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
