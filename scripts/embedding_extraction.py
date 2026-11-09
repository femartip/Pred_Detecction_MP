from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .pre_detection_features import crop_candidate_roi, image_to_dino_tensor, resize_short_side


@dataclass(frozen=True)
class EmbeddingSpec:
    csv_name: str
    dim: int
    batch_size: int
    build_model: Callable[[torch.device], tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], str]]


def recompute_embeddings(
    targets: pd.DataFrame,
    embedding_dir: Path,
    csv_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dir.mkdir(parents=True, exist_ok=True)
    selected = set(csv_names) if csv_names is not None else None
    result: dict[str, np.ndarray] = {}
    for spec in embedding_specs(device):
        if selected is not None and spec.csv_name not in selected:
            continue
        table = compute_embedding_table(targets, embedding_dir / spec.csv_name, spec)
        result[spec.csv_name] = table.drop(columns=["row_id"]).to_numpy(np.float32)
    return result


def embedding_specs(device: torch.device) -> list[EmbeddingSpec]:
    return [
        EmbeddingSpec("dinov2_vitb14_cv2_roi518.csv", 768, 16, lambda d=device: build_dino_model("vit_base_patch14_dinov2", d, roi_dino_tensor)),
        EmbeddingSpec("dinov2_vitb14.csv", 768, 16, lambda d=device: build_dino_model("vit_base_patch14_dinov2", d, full_dino_tensor)),
        EmbeddingSpec("dinov2_vitl14_crop.csv", 1024, 8, lambda d=device: build_dino_model("vit_large_patch14_dinov2", d, full_dino_tensor)),
        EmbeddingSpec("dinov2_vitb_warp518.csv", 768, 16, lambda d=device: build_dino_model("vit_base_patch14_dinov2", d, warp_dino_tensor)),
        EmbeddingSpec("dinov2_vitb_fivecrop518.csv", 3840, 8, lambda d=device: build_dino_model("vit_base_patch14_dinov2", d, fivecrop_dino_tensor)),
        EmbeddingSpec("convnext_tiny_imagenet.csv", 768, 32, lambda d=device: build_timm_model("convnext_tiny", d)),
        EmbeddingSpec("efficientnet_v2_s_imagenet.csv", 1280, 32, lambda d=device: build_timm_model("tf_efficientnetv2_s", d)),
        EmbeddingSpec("swin_t_imagenet.csv", 768, 32, lambda d=device: build_timm_model("swin_tiny_patch4_window7_224", d)),
    ]


def compute_embedding_table(targets: pd.DataFrame, output_csv: Path, spec: EmbeddingSpec) -> pd.DataFrame:
    print(f"[embeddings] {output_csv.name}: loading model", flush=True)
    model, image_transform, prefix = spec.build_model()
    rows: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(targets), spec.batch_size):
            batch = targets.iloc[start : start + spec.batch_size]
            rows.extend(encode_batch(batch, model, image_transform))
            if (start // spec.batch_size) % 10 == 0:
                print(f"[embeddings] {output_csv.name}: {min(len(rows), len(targets))}/{len(targets)}", flush=True)
    columns = ["row_id", *[f"{prefix}_{idx:04d}" for idx in range(spec.dim)]]
    table = pd.DataFrame(rows, columns=columns).sort_values("row_id").reset_index(drop=True)
    table.to_csv(output_csv, index=False)
    print(f"[embeddings] {output_csv.name}: written", flush=True)
    return table


def encode_batch(
    batch: pd.DataFrame,
    model: torch.nn.Module,
    image_transform: Callable[[Image.Image], torch.Tensor],
) -> list[list[float]]:
    rows: list[list[float]] = []
    tensors: list[torch.Tensor] = []
    row_ids: list[int] = []
    for record in batch.itertuples(index=False):
        with Image.open(record.image_path) as image:
            tensor = image_transform(image.convert("RGB"))
        tensors.append(tensor)
        row_ids.append(int(record.row_id))

    first = tensors[0]
    device = next(model.parameters()).device
    if first.ndim == 4:
        flat = torch.cat(tensors, dim=0).to(device)
        values = forward_embedding(model, flat).detach().cpu().numpy().astype(np.float32)
        crops = first.shape[0]
        values = values.reshape(len(tensors), crops * values.shape[1])
    else:
        stacked = torch.stack(tensors, dim=0).to(device)
        values = forward_embedding(model, stacked).detach().cpu().numpy().astype(np.float32)

    for row_id, value in zip(row_ids, values):
        rows.append([row_id, *value.reshape(-1).tolist()])
    return rows


def build_dino_model(
    model_name: str,
    device: torch.device,
    image_transform: Callable[[Image.Image], torch.Tensor],
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], str]:
    model = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True).to(device).eval()
    return model, image_transform, dino_prefix(model_name, image_transform)


def build_timm_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], str]:
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return model, transform, imagenet_prefix(model_name)


def dino_prefix(model_name: str, image_transform: Callable[[Image.Image], torch.Tensor]) -> str:
    if image_transform is roi_dino_tensor:
        return "cv2_roi"
    if image_transform is warp_dino_tensor:
        return "vitb_warp518"
    if image_transform is fivecrop_dino_tensor:
        return "vitb_fivecrop518"
    if "large" in model_name:
        return "vitl14_crop"
    return "vitb14"


def imagenet_prefix(model_name: str) -> str:
    if model_name == "convnext_tiny":
        return "convnext_tiny"
    if model_name == "tf_efficientnetv2_s":
        return "efficientnet_v2_s"
    return "swin_t"


def forward_embedding(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    features = model.forward_features(tensor)
    if hasattr(model, "forward_head"):
        return model.forward_head(features, pre_logits=True)
    return model(tensor)


def full_dino_tensor(image: Image.Image) -> torch.Tensor:
    return image_to_dino_tensor(image).squeeze(0)


def roi_dino_tensor(image: Image.Image) -> torch.Tensor:
    return image_to_dino_tensor(crop_candidate_roi(image)).squeeze(0)


def warp_dino_tensor(image: Image.Image) -> torch.Tensor:
    warped = image.resize((518, 518), Image.Resampling.BICUBIC)
    return normalize_rgb_tensor(TF.to_tensor(warped))


def fivecrop_dino_tensor(image: Image.Image) -> torch.Tensor:
    resized = resize_short_side(image, 560)
    crops = TF.five_crop(resized, 518)
    return torch.stack([normalize_rgb_tensor(TF.to_tensor(crop)) for crop in crops], dim=0)


def normalize_rgb_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std
