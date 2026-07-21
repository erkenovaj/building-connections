"""Generate difficulty-estimation tables and visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from difficulty_estimation.workflow.analysis import generate_analysis
from difficulty_estimation.workflow.data import DEFAULT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--strict-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_analysis(
        args.root,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        strict_counts=args.strict_counts,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

