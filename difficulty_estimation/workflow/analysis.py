"""Reproducible statistical analysis and visualization outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Callable

_cache_root = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "connections_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "connections_xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr

from .data import (
    DEFAULT_ROOT,
    build_dataset,
    build_joint_config,
    build_summary,
    validate_headline_counts,
)


def _numeric_pair(frame: pd.DataFrame, x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    subset = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    return subset[x_column].to_numpy(dtype=float), subset[y_column].to_numpy(dtype=float)


def _safe_corr(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], object],
) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    result = fn(x, y)
    return float(getattr(result, "statistic", result[0]))


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], object],
    *,
    iterations: int,
    seed: int = 13,
) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample_index = rng.integers(0, len(x), size=len(x))
        value = _safe_corr(x[sample_index], y[sample_index], fn)
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def build_correlation_table(
    dataset: pd.DataFrame,
    *,
    bootstrap_iterations: int = 1000,
) -> pd.DataFrame:
    metrics = [
        "user_difficulty_calibrated",
        "user_difficulty_raw",
        "failure_rate",
        "success_rate",
        "averageMistakes",
        "averageCategories",
        "illogicalGuesses",
        "avgSkill",
        "win_percentile",
        "mistakenGuessesCount",
        "oneAwayGuessesCount",
        "mistaken_guesses_per_completed",
        "one_away_guesses_per_completed",
        "top_mistaken_guess_share",
    ]
    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        if metric not in dataset.columns:
            continue
        x, y = _numeric_pair(dataset, "expert_difficulty", metric)
        rows.append(
            {
                "metric": metric,
                "n": len(x),
                "pearson": _safe_corr(x, y, pearsonr),
                "pearson_ci_low": _bootstrap_ci(
                    x, y, pearsonr, iterations=bootstrap_iterations
                )[0],
                "pearson_ci_high": _bootstrap_ci(
                    x, y, pearsonr, iterations=bootstrap_iterations
                )[1],
                "spearman": _safe_corr(x, y, spearmanr),
                "spearman_ci_low": _bootstrap_ci(
                    x, y, spearmanr, iterations=bootstrap_iterations
                )[0],
                "spearman_ci_high": _bootstrap_ci(
                    x, y, spearmanr, iterations=bootstrap_iterations
                )[1],
                "kendall": _safe_corr(x, y, kendalltau),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def build_feature_correlation_table(
    dataset: pd.DataFrame,
    *,
    bootstrap_iterations: int = 1000,
) -> pd.DataFrame:
    feature_columns = [
        "num_image_items",
        "num_repeated_category_titles",
        "num_distinct_repeated_category_titles",
        "num_repeated_items_global",
        "num_items_with_multiple_category_titles",
        "avg_item_occurrences_global",
        "max_item_occurrences_global",
        "avg_item_category_count_global",
        "max_item_category_count_global",
        "num_fill_blank_categories",
        "num_exact_repeated_category_member_sets",
        "avg_item_len",
        "max_item_len",
        "avg_category_title_len",
        "max_category_title_len",
    ]
    rows: list[dict[str, float | int | str]] = []
    for feature in feature_columns:
        if feature not in dataset.columns:
            continue
        x, y = _numeric_pair(dataset, "expert_difficulty", feature)
        rows.append(
            {
                "feature": feature,
                "n": len(x),
                "pearson": _safe_corr(x, y, pearsonr),
                "pearson_ci_low": _bootstrap_ci(
                    x, y, pearsonr, iterations=bootstrap_iterations
                )[0],
                "pearson_ci_high": _bootstrap_ci(
                    x, y, pearsonr, iterations=bootstrap_iterations
                )[1],
                "spearman": _safe_corr(x, y, spearmanr),
                "spearman_ci_low": _bootstrap_ci(
                    x, y, spearmanr, iterations=bootstrap_iterations
                )[0],
                "spearman_ci_high": _bootstrap_ci(
                    x, y, spearmanr, iterations=bootstrap_iterations
                )[1],
                "kendall": _safe_corr(x, y, kendalltau),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def build_category_reuse_table(
    categories: pd.DataFrame,
    items: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    category_labels = categories.merge(
        dataset[
            [
                "date",
                "expert_difficulty",
                "user_difficulty_calibrated",
                "failure_rate",
                "averageMistakes",
            ]
        ],
        on="date",
        how="left",
    )
    item_sizes = (
        items.groupby("category_norm")["item_norm"]
        .nunique()
        .rename("merged_unique_items")
    )
    table = (
        category_labels.groupby("category_norm")
        .agg(
            category_title=("category_title", "first"),
            occurrence_count=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            distinct_member_sets=("item_norms", "nunique"),
            fill_blank_occurrences=("is_fill_blank", "sum"),
            mean_expert_difficulty=("expert_difficulty", "mean"),
            mean_user_difficulty_calibrated=("user_difficulty_calibrated", "mean"),
            mean_failure_rate=("failure_rate", "mean"),
            mean_average_mistakes=("averageMistakes", "mean"),
        )
        .join(item_sizes)
        .reset_index()
    )
    return table.sort_values(
        ["occurrence_count", "merged_unique_items", "category_title"],
        ascending=[False, False, True],
    )


def build_item_reuse_table(items: pd.DataFrame) -> pd.DataFrame:
    table = (
        items.groupby("item_norm")
        .agg(
            item_text=("item_text", "first"),
            occurrence_count=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            distinct_category_titles=("category_norm", "nunique"),
            distinct_puzzles=("date", "nunique"),
            image_occurrences=("is_image", "sum"),
            categories=("category_title", lambda values: " | ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return table.sort_values(
        ["occurrence_count", "distinct_category_titles", "item_text"],
        ascending=[False, False, True],
    )


def _save_histogram(
    values: pd.Series,
    *,
    path: Path,
    title: str,
    xlabel: str,
    bins: int | list[int] = 30,
) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(values.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_scatter(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    subset = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    plt.figure(figsize=(7, 5))
    sns.regplot(data=subset, x=x, y=y, scatter_kws={"alpha": 0.45, "s": 22})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_boxplot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    subset = frame[[x, y]].dropna()
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=subset, x=x, y=y)
    sns.stripplot(data=subset, x=x, y=y, color="black", alpha=0.25, size=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_analysis(
    root: Path | str = DEFAULT_ROOT,
    *,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    bootstrap_iterations: int = 1000,
    strict_counts: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    """Generate dataset artifacts, statistics tables, and PNG visualizations."""
    root = Path(root)
    artifacts_path = Path(artifacts_dir) if artifacts_dir else root / "artifacts"
    reports_path = Path(reports_dir) if reports_dir else root / "reports"
    table_path = reports_path / "tables"
    figure_path = reports_path / "figures"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    dataset, tables = build_dataset(root, limit=limit)
    summary = build_summary(tables, dataset)
    if strict_counts and limit is None:
        validate_headline_counts(summary)

    joint_config = build_joint_config(tables.items)
    dataset.to_csv(artifacts_path / "connections_difficulty_dataset.csv", index=False)
    with (artifacts_path / "joint_config.json").open("w", encoding="utf-8") as file:
        json.dump(joint_config, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (artifacts_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    category_table = build_category_reuse_table(
        tables.categories, tables.items, dataset
    )
    item_table = build_item_reuse_table(tables.items)
    correlations = build_correlation_table(
        dataset, bootstrap_iterations=bootstrap_iterations
    )
    feature_correlations = build_feature_correlation_table(
        dataset, bootstrap_iterations=bootstrap_iterations
    )

    category_table.to_csv(table_path / "category_reuse.csv", index=False)
    item_table.to_csv(table_path / "item_reuse.csv", index=False)
    correlations.to_csv(table_path / "expert_user_correlations.csv", index=False)
    feature_correlations.to_csv(
        table_path / "expert_feature_correlations.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(table_path / "headline_summary.csv", index=False)

    category_size = (
        tables.items.groupby("category_norm")["item_norm"].nunique().rename("size")
    )
    _save_histogram(
        category_size,
        path=figure_path / "category_size_distribution.png",
        title="Merged Category Size Distribution",
        xlabel="Unique items in merged category title",
        bins=30,
    )
    _save_histogram(
        category_table["occurrence_count"],
        path=figure_path / "category_title_reuse_histogram.png",
        title="Category Title Reuse",
        xlabel="Occurrences per category title",
        bins=list(range(1, int(category_table["occurrence_count"].max()) + 2)),
    )
    _save_histogram(
        item_table["occurrence_count"],
        path=figure_path / "item_reuse_histogram.png",
        title="Item Reuse Across Puzzles",
        xlabel="Occurrences per item string",
        bins=40,
    )
    _save_scatter(
        dataset,
        x="user_difficulty_calibrated",
        y="expert_difficulty",
        path=figure_path / "expert_vs_user_difficulty.png",
        title="Expert vs User-Calibrated Difficulty",
        xlabel="User-calibrated difficulty",
        ylabel="Expert difficulty",
    )
    _save_boxplot(
        dataset,
        x="num_fill_blank_categories",
        y="expert_difficulty",
        path=figure_path / "expert_by_fill_blank_count.png",
        title="Expert Difficulty by Fill-in-the-Blank Category Count",
        xlabel="Fill-in-the-blank categories",
        ylabel="Expert difficulty",
    )
    _save_scatter(
        dataset,
        x="num_repeated_items_global",
        y="expert_difficulty",
        path=figure_path / "expert_vs_reused_items.png",
        title="Expert Difficulty vs Globally Reused Items",
        xlabel="Items reused elsewhere in history",
        ylabel="Expert difficulty",
    )

    return {
        "summary": summary,
        "artifact_dir": str(artifacts_path),
        "reports_dir": str(reports_path),
        "tables": {
            "category_reuse": str(table_path / "category_reuse.csv"),
            "item_reuse": str(table_path / "item_reuse.csv"),
            "expert_user_correlations": str(
                table_path / "expert_user_correlations.csv"
            ),
            "expert_feature_correlations": str(
                table_path / "expert_feature_correlations.csv"
            ),
        },
    }
