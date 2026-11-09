"""Command line entry point for local microfibre IoU experiments."""

from __future__ import annotations

import argparse

import pandas as pd

from .fold_clean_pre_detection_pipeline import run_fold_clean_pre_detection_pipeline
from .post_detection_pipeline import run_fold_clean_post_detection_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run microfibre IoU experiments")
    parser.add_argument("--dataset", choices=["ca", "glass", "both"], default="both")
    parser.add_argument("--stage", choices=["pre", "post", "both"], default="pre")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    datasets = ["ca", "glass"] if args.dataset == "both" else [args.dataset]
    stage_runs = []
    if args.stage in {"pre", "both"}:
        stage_runs.append(("pre-detection", "outputs/pre_detection/summary.csv", run_fold_clean_pre_detection_pipeline))
    if args.stage in {"post", "both"}:
        stage_runs.append(("post-detection", "outputs/post_detection/summary.csv", run_fold_clean_post_detection_pipeline))

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
