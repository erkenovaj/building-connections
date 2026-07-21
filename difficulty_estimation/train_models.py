"""Train/evaluate difficulty estimators and write comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from difficulty_estimation.workflow.data import DEFAULT_ROOT, write_artifacts
from difficulty_estimation.workflow.modeling import (
    collect_prediction_metrics,
    run_model_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_ROOT / "artifacts" / "connections_difficulty_dataset.csv",
        help="Dataset CSV produced by build_artifacts.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "artifacts" / "models",
        help="Directory for model metrics, predictions, and saved models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["minilm", "modernbert", "gemma"],
        choices=["minilm", "modernbert", "gemma"],
        help="Models to attempt. Optional LoRA models are skipped if deps/access are missing.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--lora-epochs",
        type=int,
        default=20,
        help="Maximum epochs for optional LoRA models.",
    )
    parser.add_argument(
        "--lora-batch-size",
        type=int,
        default=8,
        help="Batch size for optional LoRA models.",
    )
    parser.add_argument(
        "--lora-learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for optional LoRA models.",
    )
    parser.add_argument(
        "--lora-max-length",
        type=int,
        default=512,
        help="Tokenizer max length for optional LoRA models.",
    )
    parser.add_argument(
        "--build-dataset-if-missing",
        action="store_true",
        help="Create the dataset artifact first if it is missing.",
    )
    parser.add_argument(
        "--from-predictions",
        action="store_true",
        help="Rebuild model_comparison.* from saved prediction CSVs without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_predictions:
        comparison = collect_prediction_metrics(
            args.output_dir,
            dataset_path=args.dataset,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(comparison.to_string(index=False))
        return
    if args.build_dataset_if_missing and not args.dataset.exists():
        write_artifacts(DEFAULT_ROOT, artifacts_dir=args.dataset.parent)
    comparison = run_model_comparison(
        args.dataset,
        args.output_dir,
        models=args.models,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        lora_epochs=args.lora_epochs,
        lora_batch_size=args.lora_batch_size,
        lora_learning_rate=args.lora_learning_rate,
        lora_max_length=args.lora_max_length,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
