"""Data ingestion and artifact builders for Connections difficulty estimation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
CATEGORY_COLORS = ("yellow", "green", "blue", "purple")
EXPERT_MIN = 0.5
EXPERT_MAX = 4.6
EXPECTED_HEADLINE_COUNTS = {
    "puzzle_rows": 743,
    "expert_labels": 741,
    "stats_rows": 743,
    "item_occurrences": 11888,
    "image_item_occurrences": 97,
    "unique_category_titles": 2802,
    "repeated_category_titles": 144,
    "unique_item_strings": 6562,
}


def normalize_text(value: object) -> str:
    """Normalize item/category strings for cross-puzzle deduplication."""
    return re.sub(r"\s+", " ", str(value or "").strip().upper())


def _safe_float(value: object) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _safe_int(value: object) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _card_text(card: dict[str, Any]) -> str:
    """Return the canonical visible card text, including image-only puzzles."""
    return str(
        card.get("content")
        or card.get("image_alt_text")
        or card.get("image_url")
        or ""
    ).strip()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _pipe(values: Iterable[object]) -> str:
    return "|".join(str(value) for value in values)


def _comma(values: Iterable[object]) -> str:
    return ", ".join(str(value) for value in values)


def _render_solved_game(categories: pd.DataFrame, board_text: str) -> str:
    lines = ["Board order:", board_text, "", "Solved categories:"]
    for _, category in categories.sort_values("category_index").iterrows():
        lines.append(
            f"{int(category['category_index']) + 1}. "
            f"{category['category_title']}: {category['items']}"
        )
    return "\n".join(lines)


def _render_prompt_text(categories: pd.DataFrame, board_text: str) -> str:
    return (
        "Estimate the difficulty of this solved Connections puzzle.\n\n"
        f"{_render_solved_game(categories, board_text)}\n\n"
        "Return JSON with keys difficulty and explanation."
    )


def load_puzzle_tables(
    root: Path | str = DEFAULT_ROOT,
    *,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load puzzle JSONs into puzzle, category, and item tables."""
    root = Path(root)
    paths = sorted((root / "puzzle").glob("*-puzzle.json"))
    if limit is not None:
        paths = paths[:limit]

    puzzle_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []

    for source_path in paths:
        payload = _read_json(source_path)
        date = str(payload["print_date"])
        categories = payload.get("categories") or []
        board_cards: list[tuple[int, str]] = []
        image_count = 0

        for category_index, category in enumerate(categories):
            title = str(category.get("title") or "").strip()
            cards = category.get("cards") or []
            item_texts: list[str] = []
            item_norms: list[str] = []

            for item_index, card in enumerate(cards):
                text = _card_text(card)
                text_norm = normalize_text(text)
                position = _safe_int(card.get("position"))
                is_image = bool(card.get("image_alt_text") and not card.get("content"))
                if is_image:
                    image_count += 1
                if position is not None:
                    board_cards.append((position, text))
                item_texts.append(text)
                item_norms.append(text_norm)
                item_rows.append(
                    {
                        "date": date,
                        "source_file": source_path.name,
                        "json_id": payload.get("id"),
                        "editor": payload.get("editor"),
                        "category_index": category_index,
                        "category_color": CATEGORY_COLORS[category_index]
                        if category_index < len(CATEGORY_COLORS)
                        else f"category_{category_index + 1}",
                        "category_title": title,
                        "category_norm": normalize_text(title),
                        "item_index": item_index,
                        "item_text": text,
                        "item_norm": text_norm,
                        "position": position,
                        "is_image": is_image,
                        "image_url": card.get("image_url") or "",
                    }
                )

            category_rows.append(
                {
                    "date": date,
                    "source_file": source_path.name,
                    "json_id": payload.get("id"),
                    "editor": payload.get("editor"),
                    "category_index": category_index,
                    "category_color": CATEGORY_COLORS[category_index]
                    if category_index < len(CATEGORY_COLORS)
                    else f"category_{category_index + 1}",
                    "category_title": title,
                    "category_norm": normalize_text(title),
                    "category_size": len(item_texts),
                    "items": _comma(item_texts),
                    "item_norms": _pipe(item_norms),
                    "is_fill_blank": "___" in title,
                }
            )

        board_order = [text for _, text in sorted(board_cards, key=lambda pair: pair[0])]
        board_text = "; ".join(
            f"{position}: {text}" for position, text in sorted(board_cards)
        )
        puzzle_rows.append(
            {
                "date": date,
                "source_file": source_path.name,
                "json_id": payload.get("id"),
                "status": payload.get("status"),
                "editor": payload.get("editor"),
                "n_categories": len(categories),
                "n_items": sum(len(category.get("cards") or []) for category in categories),
                "num_image_items": image_count,
                "board_items": _pipe(board_order),
                "board_text": board_text,
                "category_titles": _pipe(
                    category.get("title") or "" for category in categories
                ),
            }
        )

    puzzles = pd.DataFrame(puzzle_rows)
    categories = pd.DataFrame(category_rows)
    items = pd.DataFrame(item_rows)

    if not puzzles.empty:
        solved_texts: list[str] = []
        prompt_texts: list[str] = []
        for _, puzzle in puzzles.sort_values("date").iterrows():
            category_subset = categories[categories["date"] == puzzle["date"]]
            solved_text = _render_solved_game(category_subset, puzzle["board_text"])
            prompt_text = _render_prompt_text(category_subset, puzzle["board_text"])
            solved_texts.append(solved_text)
            prompt_texts.append(prompt_text)
        puzzles = puzzles.sort_values("date").reset_index(drop=True)
        puzzles["solved_game_text"] = solved_texts
        puzzles["prompt_text"] = prompt_texts

    return puzzles, categories, items


def load_expert_ratings(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Load expert scalar ratings and normalize dates to ISO strings."""
    root = Path(root)
    path = root / "connections_rater_difficulty.csv"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "puzzle_number",
                "expert_difficulty",
                "expert_verbatim",
                "expert_url",
            ]
        )

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            date = datetime.strptime(row["date"], "%d/%m/%Y").strftime("%Y-%m-%d")
            rows.append(
                {
                    "date": date,
                    "puzzle_number": _safe_int(row.get("puzzle_number")),
                    "expert_difficulty": _safe_float(row.get("difficulty")),
                    "expert_verbatim": row.get("verbatim") or "",
                    "expert_url": row.get("url") or "",
                }
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_stats_table(
    root: Path | str = DEFAULT_ROOT,
    *,
    stats_dir_name: str = "stats",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load active user stats and flatten the fields used for calibration."""
    root = Path(root)
    paths = sorted((root / stats_dir_name).glob("*-stats.json"))
    if limit is not None:
        paths = paths[:limit]

    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        date = path.name[:10]
        complete = _safe_float(payload.get("complete"))
        success = _safe_float(payload.get("success"))
        failure = _safe_float(payload.get("failure"))
        total_players = _safe_float(payload.get("totalPlayers"))
        mistaken_guesses_count = _safe_float(payload.get("mistakenGuessesCount"))
        one_away_guesses_count = _safe_float(payload.get("oneAwayGuessesCount"))
        mistaken_guesses = payload.get("mistakenGuesses") or {}
        top_mistaken_count = max(mistaken_guesses.values()) if mistaken_guesses else 0
        category_percentiles = payload.get("categoryPercentiles") or {}
        categories_found = payload.get("categoriesFound") or {}
        found = payload.get("found") or {}
        mistakes = payload.get("mistakes") or {}

        row: dict[str, Any] = {
            "date": date,
            "success": success,
            "failure": failure,
            "complete": complete,
            "incomplete": _safe_float(payload.get("incomplete")),
            "totalPlayers": total_players,
            "success_rate": success / complete if complete else math.nan,
            "failure_rate": failure / complete if complete else math.nan,
            "averageMistakes": _safe_float(payload.get("averageMistakes")),
            "averageCategories": _safe_float(payload.get("averageCategories")),
            "illogicalGuesses": _safe_float(payload.get("illogicalGuesses")),
            "avgSkill": _safe_float(payload.get("avgSkill")),
            "mistakenGuessesCount": mistaken_guesses_count,
            "oneAwayGuessesCount": one_away_guesses_count,
            "mistaken_guesses_per_completed": mistaken_guesses_count / complete
            if complete
            else math.nan,
            "one_away_guesses_per_completed": one_away_guesses_count / complete
            if complete
            else math.nan,
            "top_mistaken_guess_count": top_mistaken_count,
            "top_mistaken_guess_share": top_mistaken_count / mistaken_guesses_count
            if mistaken_guesses_count
            else math.nan,
        }
        for color in CATEGORY_COLORS:
            row[f"{color}_percentile"] = _safe_float(category_percentiles.get(color))
        row["win_percentile"] = _safe_float(category_percentiles.get("win"))
        for key in ("0", "1", "2", "3", "4"):
            row[f"mistakes_{key}"] = _safe_float(mistakes.get(key))
        for key in ("0", "1", "2"):
            row[f"categories_found_{key}"] = _safe_float(categories_found.get(key))
        for key in ("0", "1", "2", "3"):
            row[f"found_{key}"] = _safe_float(found.get(key))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@dataclass(frozen=True)
class CorpusTables:
    puzzles: pd.DataFrame
    categories: pd.DataFrame
    items: pd.DataFrame
    expert: pd.DataFrame
    stats: pd.DataFrame


def load_corpus_tables(
    root: Path | str = DEFAULT_ROOT,
    *,
    limit: int | None = None,
) -> CorpusTables:
    puzzles, categories, items = load_puzzle_tables(root, limit=limit)
    expert = load_expert_ratings(root)
    stats = load_stats_table(root, limit=limit)
    return CorpusTables(puzzles, categories, items, expert, stats)


def compute_global_features(
    puzzles: pd.DataFrame,
    categories: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    """Compute puzzle-level lexical and reuse features from all history."""
    category_counts = categories["category_norm"].value_counts()
    item_counts = items["item_norm"].value_counts()
    item_category_counts = (
        items[["item_norm", "category_norm"]]
        .drop_duplicates()
        .groupby("item_norm")["category_norm"]
        .nunique()
    )

    member_sets = categories.assign(
        category_member_signature=categories["item_norms"].map(
            lambda value: _pipe(sorted(str(value).split("|")))
        )
    )
    member_set_counts = (
        member_sets.groupby(["category_norm", "category_member_signature"])
        .size()
        .rename("member_set_occurrences")
    )

    rows: list[dict[str, Any]] = []
    for _, puzzle in puzzles.iterrows():
        date = puzzle["date"]
        category_subset = categories[categories["date"] == date]
        item_subset = items[items["date"] == date]
        repeated_category_mask = category_subset["category_norm"].map(category_counts) > 1
        repeated_item_counts = item_subset["item_norm"].map(item_counts)
        item_multi_category_counts = item_subset["item_norm"].map(item_category_counts)
        category_member_repeats = []
        for _, category in category_subset.iterrows():
            signature = _pipe(sorted(str(category["item_norms"]).split("|")))
            category_member_repeats.append(
                member_set_counts.get((category["category_norm"], signature), 0)
            )

        rows.append(
            {
                "date": date,
                "num_repeated_category_titles": int(repeated_category_mask.sum()),
                "num_distinct_repeated_category_titles": int(
                    category_subset.loc[repeated_category_mask, "category_norm"].nunique()
                ),
                "num_repeated_items_global": int((repeated_item_counts > 1).sum()),
                "num_items_with_multiple_category_titles": int(
                    (item_multi_category_counts > 1).sum()
                ),
                "avg_item_occurrences_global": float(repeated_item_counts.mean()),
                "max_item_occurrences_global": int(repeated_item_counts.max()),
                "avg_item_category_count_global": float(item_multi_category_counts.mean()),
                "max_item_category_count_global": int(item_multi_category_counts.max()),
                "num_fill_blank_categories": int(category_subset["is_fill_blank"].sum()),
                "avg_item_len": float(item_subset["item_text"].str.len().mean()),
                "max_item_len": int(item_subset["item_text"].str.len().max()),
                "avg_category_title_len": float(
                    category_subset["category_title"].str.len().mean()
                ),
                "max_category_title_len": int(
                    category_subset["category_title"].str.len().max()
                ),
                "num_exact_repeated_category_member_sets": int(
                    sum(count > 1 for count in category_member_repeats)
                ),
            }
        )
    return pd.DataFrame(rows)


def _add_category_columns(dataset: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    for category_index in range(4):
        subset = categories[categories["category_index"] == category_index][
            ["date", "category_title", "items", "is_fill_blank"]
        ].rename(
            columns={
                "category_title": f"category_{category_index + 1}_title",
                "items": f"category_{category_index + 1}_items",
                "is_fill_blank": f"category_{category_index + 1}_is_fill_blank",
            }
        )
        result = result.merge(subset, on="date", how="left")
    return result


def assign_temporal_splits(dataset: pd.DataFrame) -> pd.Series:
    """Assign deterministic 70/15/15 splits over expert-labeled rows."""
    split = pd.Series("unlabeled", index=dataset.index, dtype="object")
    labeled = dataset[dataset["expert_difficulty"].notna()].sort_values("date")
    n_labeled = len(labeled)
    train_end = round(n_labeled * 0.70)
    validation_end = round(n_labeled * 0.85)
    train_index = labeled.index[:train_end]
    validation_index = labeled.index[train_end:validation_end]
    test_index = labeled.index[validation_end:]
    split.loc[train_index] = "train"
    split.loc[validation_index] = "validation"
    split.loc[test_index] = "test"
    return split


def _add_user_difficulty_scores(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    result["user_difficulty_raw"] = (
        pd.to_numeric(result["averageMistakes"], errors="coerce")
        + pd.to_numeric(result["failure_rate"], errors="coerce")
    )

    common = result[
        result["expert_difficulty"].notna()
        & result["user_difficulty_raw"].notna()
    ]
    if len(common) > 1 and common["user_difficulty_raw"].std(ddof=0):
        raw_mean = common["user_difficulty_raw"].mean()
        raw_std = common["user_difficulty_raw"].std(ddof=0)
        expert_mean = common["expert_difficulty"].mean()
        expert_std = common["expert_difficulty"].std(ddof=0)
        calibrated = (
            (result["user_difficulty_raw"] - raw_mean)
            / raw_std
            * expert_std
            + expert_mean
        )
        result["user_difficulty_calibrated"] = calibrated.clip(EXPERT_MIN, EXPERT_MAX)
    else:
        result["user_difficulty_calibrated"] = np.nan
    return result


def build_dataset(
    root: Path | str = DEFAULT_ROOT,
    *,
    limit: int | None = None,
) -> tuple[pd.DataFrame, CorpusTables]:
    """Build the one-row-per-puzzle modeling dataset."""
    tables = load_corpus_tables(root, limit=limit)
    features = compute_global_features(tables.puzzles, tables.categories, tables.items)
    dataset = (
        tables.puzzles.merge(tables.expert, on="date", how="left")
        .merge(tables.stats, on="date", how="left")
        .merge(features, on="date", how="left")
    )
    dataset = _add_category_columns(dataset, tables.categories)
    dataset = _add_user_difficulty_scores(dataset)
    dataset["split"] = assign_temporal_splits(dataset)
    dataset = dataset.sort_values("date").reset_index(drop=True)

    ordered_columns = [
        "date",
        "split",
        "json_id",
        "puzzle_number",
        "source_file",
        "editor",
        "status",
        "n_categories",
        "n_items",
        "expert_difficulty",
        "user_difficulty_calibrated",
        "user_difficulty_raw",
        "success_rate",
        "failure_rate",
        "averageMistakes",
        "averageCategories",
        "illogicalGuesses",
        "avgSkill",
        "mistakenGuessesCount",
        "oneAwayGuessesCount",
        "mistaken_guesses_per_completed",
        "one_away_guesses_per_completed",
        "top_mistaken_guess_share",
        "win_percentile",
        "yellow_percentile",
        "green_percentile",
        "blue_percentile",
        "purple_percentile",
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
        "board_items",
        "board_text",
        "category_titles",
        "category_1_title",
        "category_1_items",
        "category_1_is_fill_blank",
        "category_2_title",
        "category_2_items",
        "category_2_is_fill_blank",
        "category_3_title",
        "category_3_items",
        "category_3_is_fill_blank",
        "category_4_title",
        "category_4_items",
        "category_4_is_fill_blank",
        "solved_game_text",
        "prompt_text",
        "expert_url",
        "expert_verbatim",
    ]
    existing_columns = [column for column in ordered_columns if column in dataset.columns]
    remaining_columns = [
        column for column in dataset.columns if column not in set(existing_columns)
    ]
    return dataset[existing_columns + remaining_columns], tables


def build_joint_config(items: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a sampler-compatible item/tag config from all historical puzzles."""
    if items.empty:
        return []
    sorted_items = items.sort_values(["date", "position", "category_index", "item_index"])
    entries: list[dict[str, Any]] = []
    for item_norm, group in sorted_items.groupby("item_norm", sort=True):
        canonical_name = group.iloc[0]["item_text"]
        tag_rows = (
            group[["category_norm", "category_title"]]
            .drop_duplicates("category_norm")
            .sort_values("category_norm")
        )
        tags = tag_rows["category_title"].tolist()
        entries.append({"name": canonical_name, "tags": tags})
    return entries


def build_summary(tables: CorpusTables, dataset: pd.DataFrame) -> dict[str, Any]:
    category_counts = tables.categories["category_norm"].value_counts()
    item_counts = tables.items["item_norm"].value_counts()
    item_category_counts = (
        tables.items[["item_norm", "category_norm"]]
        .drop_duplicates()
        .groupby("item_norm")["category_norm"]
        .nunique()
    )
    split_counts = dataset["split"].value_counts().to_dict()
    return {
        "puzzle_rows": int(len(tables.puzzles)),
        "category_occurrences": int(len(tables.categories)),
        "item_occurrences": int(len(tables.items)),
        "expert_labels": int(dataset["expert_difficulty"].notna().sum()),
        "stats_rows": int(len(tables.stats)),
        "stats_joined_rows": int(dataset["success_rate"].notna().sum()),
        "image_item_occurrences": int(tables.items["is_image"].sum()),
        "unique_category_titles": int(category_counts.size),
        "repeated_category_titles": int((category_counts > 1).sum()),
        "category_occurrences_in_repeated_titles": int(category_counts[category_counts > 1].sum()),
        "unique_item_strings": int(item_counts.size),
        "repeated_item_strings": int((item_counts > 1).sum()),
        "item_occurrences_of_repeated_strings": int(item_counts[item_counts > 1].sum()),
        "unique_items_in_non_unique_categories": int(
            tables.items[
                tables.items["category_norm"].isin(category_counts[category_counts > 1].index)
            ]["item_norm"].nunique()
        ),
        "item_occurrences_in_non_unique_categories": int(
            len(
                tables.items[
                    tables.items["category_norm"].isin(
                        category_counts[category_counts > 1].index
                    )
                ]
            )
        ),
        "items_under_multiple_category_titles": int((item_category_counts > 1).sum()),
        "item_occurrences_under_multiple_category_titles": int(
            item_counts[item_category_counts[item_category_counts > 1].index].sum()
        ),
        "date_min": str(tables.puzzles["date"].min()) if not tables.puzzles.empty else None,
        "date_max": str(tables.puzzles["date"].max()) if not tables.puzzles.empty else None,
        "missing_expert_dates": dataset.loc[
            dataset["expert_difficulty"].isna(), "date"
        ].tolist(),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
    }


def validate_headline_counts(summary: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in EXPECTED_HEADLINE_COUNTS.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Headline count mismatch: {json.dumps(mismatches, indent=2)}")


def write_artifacts(
    root: Path | str = DEFAULT_ROOT,
    *,
    artifacts_dir: Path | str | None = None,
    limit: int | None = None,
    strict_counts: bool = False,
) -> dict[str, Any]:
    """Write joint config, dataset CSV, and summary JSON."""
    root = Path(root)
    artifacts_path = Path(artifacts_dir) if artifacts_dir else root / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)

    dataset, tables = build_dataset(root, limit=limit)
    joint_config = build_joint_config(tables.items)
    summary = build_summary(tables, dataset)
    if strict_counts and limit is None:
        validate_headline_counts(summary)

    dataset.to_csv(artifacts_path / "connections_difficulty_dataset.csv", index=False)
    with (artifacts_path / "joint_config.json").open("w", encoding="utf-8") as file:
        json.dump(joint_config, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (artifacts_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary
