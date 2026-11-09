"""Validate evaluation membership and expose detector training overlap in outputs."""

import json
from pathlib import Path

import pandas as pd


def evaluation_metadata(targets: pd.DataFrame, split_root: Path, output_dir: Path) -> dict:
    columns = ["row_id", "fold", "detector_seen_during_training"]
    if not set(columns).issubset(targets.columns):
        raise ValueError("Targets lack detector exposure metadata; rerun without --reuse-existing.")
    membership = pd.read_csv(split_root / "evaluation_membership.csv")
    expected = membership.loc[membership["status"] == "evaluated", columns].sort_values("row_id").reset_index(drop=True)
    actual = targets[columns].sort_values("row_id").reset_index(drop=True)
    if actual["row_id"].duplicated().any() or not actual.equals(expected):
        raise ValueError("Saved targets do not match current evaluation membership; rerun without --reuse-existing.")
    seen = int(actual["detector_seen_during_training"].sum())
    metadata = {
        "evaluation_scope": "mixed_detector_exposure" if seen else "detector_unseen_only",
        "n_images": len(actual),
        "n_detector_seen": seen,
        "n_detector_unseen": len(actual) - seen,
    }
    protocol = {**metadata, "source": json.loads((split_root / "source.json").read_text())}
    (output_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return metadata
