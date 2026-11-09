"""Command line entry point for local microfibre IoU experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support direct execution as well as module and installed entry points.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare data and run microfibre IoU experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare local HF downloads")
    run = commands.add_parser("run", help="Run pre/post-detection experiments")
    for command in (prepare, run):
        command.add_argument("--dataset", choices=["ca", "glass", "both"], default="both")
    prepare.add_argument("--downloads", type=Path, default=Path("data/downloads"))
    prepare.add_argument("--models", type=Path, default=Path("models"))
    prepare.add_argument("--destination", type=Path, default=Path("data/Dataset"))
    run.add_argument("--stage", choices=["pre", "post", "both"], default="pre")
    run.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    datasets = ["ca", "glass"] if args.dataset == "both" else [args.dataset]
    if args.command == "prepare":
        from scripts.prepare_data import prepare_dataset

        for dataset in datasets:
            prepare_dataset(dataset, args.downloads, args.models, args.destination)
        return

    import pandas as pd
    from scripts.pre_detection_pipeline import run_pre_detection_pipeline
    from scripts.post_detection_pipeline import run_post_detection_pipeline

    stage_runs = []
    if args.stage in {"pre", "both"}:
        stage_runs.append(("pre-detection", "outputs/pre_detection/summary.csv", run_pre_detection_pipeline))
    if args.stage in {"post", "both"}:
        stage_runs.append(("post-detection", "outputs/post_detection/summary.csv", run_post_detection_pipeline))

    for stage_label, summary_path, runner in stage_runs:
        all_results: list[pd.DataFrame] = []
        for dataset in datasets:
            results = runner(dataset, reuse_existing=args.reuse_existing)
            all_results.append(results)
            print(f"\n=== {stage_label.upper()} {dataset.upper()} ===")
            print(results.head(10).to_string(index=False))

        if len(all_results) > 1:
            summary = pd.concat(all_results, ignore_index=True)
            summary.to_csv(summary_path, index=False)
            print(f"\n=== {stage_label.upper()} Combined ===")
            print(summary.sort_values(["dataset", "r2"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
