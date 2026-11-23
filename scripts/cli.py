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
    compare = commands.add_parser("compare", help="Evaluate calibrated confidence, TTA, RCA and dropout-injection baselines")
    commands.add_parser("ablation", help="Evaluate cross-dataset RF transfer and held-out feature importance")
    images = commands.add_parser("embedding_finetune", help="Evaluate frozen ResNet and fine-tuned ResNet/DINO pre-detection models")
    for command in (prepare, run, compare, images):
        command.add_argument("--dataset", choices=["ca", "glass", "both"], default="both")
    prepare.add_argument("--downloads", type=Path, default=Path("data/downloads"))
    prepare.add_argument("--models", type=Path, default=Path("models"))
    prepare.add_argument("--destination", type=Path, default=Path("data/Dataset"))
    run.add_argument("--stage", choices=["pre", "post", "both"], default="pre")
    run.add_argument("--reuse-existing", action="store_true")
    compare.add_argument("--workers", type=int, default=8)
    compare.add_argument("--baselines", nargs="+", choices=["confidence", "tta", "rca", "dropout", "ensemble"],
                         default=["confidence", "tta", "rca", "dropout"], help="Run only selected baselines; ensemble trains two new detectors per fold alongside the original")
    compare.add_argument("--reuse-existing", action="store_true", help="Rebuild metrics and cost summaries from saved scores; no inference")
    images.add_argument("--models", nargs="+", choices=["resnet50_frozen", "resnet50_finetuned", "dinov2_finetuned"],
                        default=["resnet50_frozen", "resnet50_finetuned", "dinov2_finetuned"])
    images.add_argument("--epochs", type=int, default=30)
    images.add_argument("--patience", type=int, default=5)
    images.add_argument("--batch-size", type=int, default=4)
    images.add_argument("--workers", type=int, default=2)
    images.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "ablation":
        from scripts.ablation import run_ablation

        run_ablation()
        return

    datasets = ["ca", "glass"] if args.dataset == "both" else [args.dataset]
    if args.command == "prepare":
        from scripts.prepare_data import prepare_dataset

        for dataset in datasets:
            prepare_dataset(dataset, args.downloads, args.models, args.destination)
        return

    import pandas as pd
    if args.command == "embedding_finetune":
        from scripts.embedding_finetuning import run_embedding_finetuning

        results = pd.concat([run_embedding_finetuning(dataset, args.models, reuse_existing=args.reuse_existing,
                            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                            workers=args.workers) for dataset in datasets], ignore_index=True)
        paths = [Path("outputs/pre_detection") / dataset / "results.csv" for dataset in ("ca", "glass")]
        pd.concat([pd.read_csv(path) for path in paths if path.exists()], ignore_index=True).to_csv("outputs/pre_detection/summary.csv", index=False)
        print(results.to_string(index=False))
        return

    if args.command == "compare":
        from scripts.baseline_comparison import run_baseline_comparison

        results = pd.concat([run_baseline_comparison(dataset, workers=args.workers, baselines=args.baselines,
                            reuse_existing=args.reuse_existing) for dataset in datasets], ignore_index=True)
        tables = [Path("outputs/comparison") / dataset / "results.csv" for dataset in ("ca", "glass")]
        pd.concat([pd.read_csv(path) for path in tables if path.exists()], ignore_index=True).to_csv("outputs/comparison/summary.csv", index=False)
        print(results.to_string(index=False))
        return

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
