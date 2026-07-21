"""Build joint config and supervised difficulty dataset artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from difficulty_estimation.workflow.data import DEFAULT_ROOT, write_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Difficulty-estimation data root.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Output directory for CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional first-N puzzle limit for smoke tests.",
    )
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help="Assert expected full-corpus headline counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_artifacts(
        args.root,
        artifacts_dir=args.artifacts_dir,
        limit=args.limit,
        strict_counts=args.strict_counts,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

