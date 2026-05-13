"""Command-line interface for ``python -m connections_sampler``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import (
    GameMode,
    SamplingParameters,
    sample_basic_game,
    sample_game,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a uniquely solvable Connections-style game."
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["easy", "basic", "advanced"],
        default="basic",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/category-templates-new.json",
    )
    parser.add_argument("--seed", default=None)
    parser.add_argument("--num-categories", type=int, default=None)
    parser.add_argument("--items-per-category", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    with open(os.path.abspath(args.config), "r", encoding="utf-8") as file:
        config = json.load(file)

    parameters = SamplingParameters(
        seed=args.seed,
        max_attempts=max(1, args.max_attempts),
        total_timeout_seconds=max(0.1, args.timeout),
    )
    parsed_mode = GameMode.parse(args.mode)
    if args.num_categories is not None or args.items_per_category is not None:
        if parsed_mode is not GameMode.BASIC:
            raise SystemExit("P x Q dimensions are available only in Basic mode.")
        result = sample_basic_game(
            config,
            num_categories=args.num_categories or 4,
            items_per_category=args.items_per_category or 4,
            parameters=parameters,
        )
    else:
        result = sample_game(config, mode=parsed_mode, parameters=parameters)

    if not result.game:
        json.dump(result.to_dict(), sys.stderr, ensure_ascii=False, indent=2)
        sys.stderr.write("\n")
        raise SystemExit(1)

    json.dump(result.game.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
