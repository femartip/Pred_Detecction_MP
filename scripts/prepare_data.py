"""Prepare local HF downloads for the CA and Glass experiments."""

from collections import defaultdict
import hashlib
from io import BytesIO
import json

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq


DATASETS = {"ca": "Dataset_CA", "glass": "Dataset_vidrio"}


def image_keys(coco):
    polygons = defaultdict(list)
    for annotation in coco["annotations"]:
        polygons[annotation["image_id"]].extend(annotation["segmentation"])
    return {
        image["id"]: (image["width"], image["height"], tuple(sorted(
            np.asarray(p, dtype=np.float32).tobytes() for p in polygons[image["id"]]
        ))) for image in coco["images"]
    }


def prepare_dataset(dataset, downloads, models, destination):
    bundle = models / dataset
    for filename, expected in json.loads((bundle / "manifest.json").read_text()).items():
        with (bundle / filename).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"], filename

    root = destination / DATASETS[dataset]
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((bundle / "hf_source.json").read_text())
    images, annotations = [], []
    for filename in source["parquet_files"]:
        for batch in pq.ParquetFile(downloads / dataset / filename).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                image_id = len(images)
                data = row["image"]["bytes"]
                with Image.open(BytesIO(data)) as image:
                    width, height = image.size
                    suffix = {"JPEG": ".jpg", "PNG": ".png"}[image.format]
                name = f"{image_id:05d}{suffix}"
                (image_dir / name).write_bytes(data)
                images.append({"id": image_id, "file_name": name, "width": width, "height": height})
                for points in row["mask"]["polygons"]["points"]:
                    polygon = [float(value) for point in points for value in point]
                    xs, ys = polygon[::2], polygon[1::2]
                    area = abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
                                   for i in range(len(xs)))) / 2
                    annotations.append({
                        "id": len(annotations), "image_id": image_id, "category_id": 1,
                        "segmentation": [polygon], "iscrowd": 0, "area": area,
                        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    })
    assert len(images) == source["image_count"]
    assert len(annotations) == source["annotation_count"]
    source["detector_splits"] = "Original membership mapped to HF IDs; expanded evaluation records detector exposure."
    coco = {"info": source, "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "fibre", "supercategory": "fibre"}]}
    mapping_table = pd.read_csv(bundle / "hf_image_mapping.csv")
    assert not mapping_table["hf_row_id"].duplicated().any()
    mapping = dict(zip(mapping_table["hf_row_id"], mapping_table["original_image_id"]))
    assert set(mapping) == set(range(len(images))) and len(set(mapping.values())) == len(images)
    original_keys = image_keys(json.loads((bundle / "original_coco.json").read_text()))
    assert all(key == original_keys[mapping[i]] for i, key in image_keys(coco).items())
    train_sets, test_sets = [], []
    for fold in range(5):
        train_sets.append({i["id"] for i in json.loads((bundle / f"fold{fold}/train_coco.json").read_text())["images"]})
        test_sets.append({i["id"] for i in json.loads((bundle / f"fold{fold}/test_coco.json").read_text())["images"]})
    eligible = {i: [k for k in range(5) if old_id not in train_sets[k]] for i, old_id in mapping.items()}
    assignments = {}
    for i, old_id in mapping.items():
        held_out = [k for k in eligible[i] if old_id in test_sets[k]]
        assert len(held_out) <= 1
        if held_out:
            assignments[i] = held_out[0]
    counts = [sum(k == fold for k in assignments.values()) for fold in range(5)]
    # Assign unseen images first, then images seen by all five detectors.
    for remaining in ([i for i in mapping if i not in assignments and eligible[i]],
                      [i for i in mapping if not eligible[i]]):
        for i in np.random.default_rng(42).permutation(remaining):
            fold = min(eligible[i] or list(range(5)), key=lambda k: (counts[k], k))
            assignments[int(i)] = fold
            counts[fold] += 1
    assert set(assignments) == set(mapping)
    for fold in range(5):
        for split, ids in (
            ("train", {i for i, old_id in mapping.items() if old_id in train_sets[fold]}),
            ("test", {i for i, k in assignments.items() if k == fold}),
        ):
            split_coco = {**coco, "images": [
                {**image, "detector_seen_during_training": mapping[image["id"]] in train_sets[fold]}
                for image in images if image["id"] in ids
            ], "annotations": [a for a in annotations if a["image_id"] in ids]}
            (root / f"{split}_coco_{fold}_fold.json").write_text(json.dumps(split_coco))
    pd.DataFrame([
        {"row_id": i, "original_image_id": old_id, "fold": assignments[i],
         "eligible_checkpoints": ";".join(map(str, eligible[i])),
         "detector_seen_during_training": old_id in train_sets[assignments[i]], "status": "evaluated"}
        for i, old_id in mapping.items()
    ]).to_csv(root / "evaluation_membership.csv", index=False)
    (root / "coco_format.json").write_text(json.dumps(coco))
    (root / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"{dataset}: {len(images)} images, {sum(not v for v in eligible.values())} detector-seen; folds {counts}")
