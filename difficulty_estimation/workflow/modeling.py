"""Training and evaluation helpers for difficulty-estimation models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


MODEL_RANDOM_SEED = 13
MIN_PAIR_GAP = 0.15
FEATURE_COLUMNS = [
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
USER_OUTCOME_COLUMNS = [
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
    "user_difficulty_raw",
    "user_difficulty_calibrated",
]


def load_supervised_dataset(dataset_path: Path | str) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    return dataset[dataset["expert_difficulty"].notna()].reset_index(drop=True)


def split_supervised_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = dataset[dataset["split"] == "train"].copy()
    validation = dataset[dataset["split"] == "validation"].copy()
    test = dataset[dataset["split"] == "test"].copy()
    return train, validation, test


def sample_pairwise_indices(
    labels: Iterable[float],
    *,
    min_gap: float = MIN_PAIR_GAP,
    max_pairs: int | None = None,
    seed: int = MODEL_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(list(labels), dtype=float)
    pairs: list[tuple[int, int]] = []
    targets: list[float] = []
    for i in range(len(labels_array)):
        for j in range(i + 1, len(labels_array)):
            difference = labels_array[i] - labels_array[j]
            if abs(difference) < min_gap:
                continue
            pairs.append((i, j))
            targets.append(1.0 if difference > 0 else 0.0)
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[index] for index in chosen]
        targets = [targets[index] for index in chosen]
    return np.asarray(pairs, dtype=np.int64), np.asarray(targets, dtype=np.float32)


def pairwise_ranking_accuracy(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    min_gap: float = MIN_PAIR_GAP,
) -> float:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    pairs, targets = sample_pairwise_indices(true, min_gap=min_gap)
    if len(pairs) == 0:
        return float("nan")
    predicted_targets = (pred[pairs[:, 0]] > pred[pairs[:, 1]]).astype(float)
    return float((predicted_targets == targets).mean())


def pairwise_regression_loss(
    predictions: Any,
    labels: Any,
    *,
    huber_delta: float = 0.5,
    pairwise_weight: float = 0.25,
    min_gap: float = MIN_PAIR_GAP,
) -> Any:
    """Torch loss: Huber on scores plus pairwise BCE on score differences."""
    import torch
    import torch.nn.functional as functional

    predictions = predictions.view(-1).float()
    labels = labels.view(-1).float()
    huber = functional.huber_loss(predictions, labels, delta=huber_delta)
    if len(labels) < 2 or pairwise_weight <= 0:
        return huber

    i_indices: list[int] = []
    j_indices: list[int] = []
    targets: list[float] = []
    label_values = labels.detach().cpu().numpy()
    for i in range(len(label_values)):
        for j in range(i + 1, len(label_values)):
            difference = label_values[i] - label_values[j]
            if abs(difference) < min_gap:
                continue
            i_indices.append(i)
            j_indices.append(j)
            targets.append(1.0 if difference > 0 else 0.0)
    if not targets:
        return huber

    device = predictions.device
    i_tensor = torch.tensor(i_indices, device=device)
    j_tensor = torch.tensor(j_indices, device=device)
    target_tensor = torch.tensor(targets, dtype=predictions.dtype, device=device)
    logits = predictions[i_tensor] - predictions[j_tensor]
    pairwise = functional.binary_cross_entropy_with_logits(logits, target_tensor)
    return huber + pairwise_weight * pairwise


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    min_gap: float = MIN_PAIR_GAP,
) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    error = pred - true
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson": float(pearsonr(true, pred).statistic)
        if len(true) > 2 and np.std(pred) > 0
        else float("nan"),
        "spearman": float(spearmanr(true, pred).statistic)
        if len(true) > 2 and np.std(pred) > 0
        else float("nan"),
        "kendall": float(kendalltau(true, pred).statistic)
        if len(true) > 2 and np.std(pred) > 0
        else float("nan"),
        "pairwise_accuracy": pairwise_ranking_accuracy(
            true, pred, min_gap=min_gap
        ),
    }
    return metrics


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    *,
    iterations: int = 1000,
    seed: int = MODEL_RANDOM_SEED,
) -> tuple[float, float]:
    if len(y_true) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample_index = rng.integers(0, len(y_true), size=len(y_true))
        value = regression_metrics(y_true[sample_index], y_pred[sample_index])[
            metric_name
        ]
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _metric_row(
    *,
    model_name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bootstrap_iterations: int,
    status: str = "ok",
    reason: str = "",
) -> dict[str, Any]:
    metrics = regression_metrics(y_true, y_pred)
    row: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "status": status,
        "reason": reason,
        "n": int(len(y_true)),
        **metrics,
    }
    for metric_name in ("mae", "rmse", "pearson", "spearman", "pairwise_accuracy"):
        low, high = _bootstrap_metric_ci(
            y_true,
            y_pred,
            metric_name,
            iterations=bootstrap_iterations,
        )
        row[f"{metric_name}_ci_low"] = low
        row[f"{metric_name}_ci_high"] = high
    return row


def _skipped_row(model_name: str, reason: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "split": "all",
        "status": "skipped",
        "reason": reason,
        "n": 0,
        "mae": np.nan,
        "rmse": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
        "kendall": np.nan,
        "pairwise_accuracy": np.nan,
    }


def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in FEATURE_COLUMNS if column in frame.columns]
    return frame[available].apply(pd.to_numeric, errors="coerce")


def train_minilm_baseline(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    bootstrap_iterations: int = 1000,
    seed: int = MODEL_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Train MiniLM embeddings plus structured features with sklearn regressors."""
    from joblib import dump
    from scipy import sparse
    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_supervised_dataset(dataset_path)
    train, validation, test = split_supervised_dataset(dataset)
    texts = dataset["prompt_text"].fillna("").tolist()
    try:
        encoder = SentenceTransformer(model_id)
        embeddings = encoder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        reason = f"MiniLM encoder unavailable: {exc}"
        return [_skipped_row("minilm_features", reason)]

    numeric = _numeric_features(dataset)
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_values = numeric_pipeline.fit_transform(numeric)
    feature_matrix = np.hstack([embeddings, numeric_values])
    row_to_position = {row_id: position for position, row_id in enumerate(dataset.index)}

    def select(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        positions = [row_to_position[index] for index in frame.index]
        return feature_matrix[positions], frame["expert_difficulty"].to_numpy(dtype=float)

    x_train, y_train = select(train)
    x_validation, y_validation = select(validation)
    x_test, y_test = select(test)

    candidates = {
        "minilm_ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "minilm_hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=120,
            learning_rate=0.04,
            l2_regularization=0.05,
        ),
        "minilm_random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            min_samples_leaf=4,
            n_jobs=-1,
        ),
    }
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    best_model_name = ""
    best_model = None
    best_validation_mae = float("inf")

    for model_name, model in candidates.items():
        model.fit(x_train, y_train)
        for split_name, frame, x_split, y_split in (
            ("validation", validation, x_validation, y_validation),
            ("test", test, x_test, y_test),
        ):
            y_pred = model.predict(x_split)
            rows.append(
                _metric_row(
                    model_name=model_name,
                    split=split_name,
                    y_true=y_split,
                    y_pred=y_pred,
                    bootstrap_iterations=bootstrap_iterations,
                )
            )
            prediction_frame = frame[["date", "split", "expert_difficulty"]].copy()
            prediction_frame["model"] = model_name
            prediction_frame["prediction"] = y_pred
            predictions.append(prediction_frame)
        validation_mae = rows[-2]["mae"]
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_model_name = model_name
            best_model = model

    if best_model is not None:
        dump(
            {
                "model_name": best_model_name,
                "model": best_model,
                "numeric_pipeline": numeric_pipeline,
                "feature_columns": list(numeric.columns),
                "encoder_model_id": model_id,
            },
            output_path / "minilm_best_model.joblib",
        )
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            output_path / "minilm_predictions.csv", index=False
        )
    return rows


class _TextRegressionDataset:
    def __init__(self, tokenizer: Any, frame: pd.DataFrame, *, max_length: int) -> None:
        self.labels = frame["expert_difficulty"].to_numpy(dtype=np.float32)
        self.encodings = tokenizer(
            frame["prompt_text"].fillna("").tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


def _target_modules_for_model(model_id: str) -> list[str] | None:
    lower = model_id.lower()
    if "gemma" in lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "modernbert" in lower:
        return ["Wqkv", "Wo", "Wi"]
    return None


def train_lora_regressor(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    model_id: str,
    model_label: str,
    bootstrap_iterations: int = 1000,
    seed: int = MODEL_RANDOM_SEED,
    max_length: int = 512,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
) -> list[dict[str, Any]]:
    """Fine-tune a LoRA sequence regressor when optional deps/model access exist."""
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:
        return [_skipped_row(model_label, f"LoRA dependencies unavailable: {exc}")]

    output_path = Path(output_dir) / model_label
    output_path.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    dataset = load_supervised_dataset(dataset_path)
    train, validation, test = split_supervised_dataset(dataset)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1,
            problem_type="regression",
        )
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=_target_modules_for_model(model_id),
        )
        model = get_peft_model(model, lora_config)
    except Exception as exc:
        return [_skipped_row(model_label, f"Model/tokenizer unavailable: {exc}")]

    train_dataset = _TextRegressionDataset(tokenizer, train, max_length=max_length)
    validation_dataset = _TextRegressionDataset(tokenizer, validation, max_length=max_length)
    test_dataset = _TextRegressionDataset(tokenizer, test, max_length=max_length)

    class PairwiseTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.view(-1)
            loss = pairwise_regression_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss

    try:
        training_arguments = TrainingArguments(
            output_dir=str(output_path),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_mae",
            greater_is_better=False,
            report_to=[],
            seed=seed,
        )
    except TypeError:
        training_arguments = TrainingArguments(
            output_dir=str(output_path),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_mae",
            greater_is_better=False,
            report_to=[],
            seed=seed,
        )

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions = np.asarray(eval_prediction.predictions).reshape(-1)
        labels = np.asarray(eval_prediction.label_ids).reshape(-1)
        return regression_metrics(labels, predictions)

    trainer_kwargs = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }
    try:
        trainer = PairwiseTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        try:
            trainer = PairwiseTrainer(**trainer_kwargs, tokenizer=tokenizer)
        except TypeError:
            trainer = PairwiseTrainer(**trainer_kwargs)

    try:
        trainer.train()
        trainer.save_model(str(output_path / "best_adapter"))
        rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        for split_name, frame, split_dataset in (
            ("validation", validation, validation_dataset),
            ("test", test, test_dataset),
        ):
            predictions = trainer.predict(split_dataset).predictions.reshape(-1)
            labels = frame["expert_difficulty"].to_numpy(dtype=float)
            rows.append(
                _metric_row(
                    model_name=model_label,
                    split=split_name,
                    y_true=labels,
                    y_pred=predictions,
                    bootstrap_iterations=bootstrap_iterations,
                )
            )
            prediction_frame = frame[["date", "split", "expert_difficulty"]].copy()
            prediction_frame["model"] = model_label
            prediction_frame["prediction"] = predictions
            prediction_frames.append(prediction_frame)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_path / "predictions.csv", index=False
        )
        return rows
    except Exception as exc:
        return [_skipped_row(model_label, f"Training failed: {exc}")]


def run_model_comparison(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    models: Iterable[str] = ("minilm", "modernbert", "gemma"),
    bootstrap_iterations: int = 1000,
    seed: int = MODEL_RANDOM_SEED,
    lora_epochs: int = 20,
    lora_batch_size: int = 8,
    lora_learning_rate: float = 2e-4,
    lora_max_length: int = 512,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected = {model.lower() for model in models}

    if "minilm" in selected:
        rows.extend(
            train_minilm_baseline(
                dataset_path,
                output_path,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        )
    if "modernbert" in selected:
        rows.extend(
            train_lora_regressor(
                dataset_path,
                output_path,
                model_id="answerdotai/ModernBERT-base",
                model_label="modernbert_base_lora",
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                epochs=lora_epochs,
                batch_size=lora_batch_size,
                learning_rate=lora_learning_rate,
                max_length=lora_max_length,
            )
        )
    if "gemma" in selected:
        rows.extend(
            train_lora_regressor(
                dataset_path,
                output_path,
                model_id="google/gemma-3-270m-it",
                model_label="gemma_3_270m_it_lora",
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                epochs=lora_epochs,
                batch_size=lora_batch_size,
                learning_rate=lora_learning_rate,
                max_length=lora_max_length,
            )
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    with (output_path / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
        file.write("\n")
    return comparison


def collect_prediction_metrics(
    output_dir: Path | str,
    *,
    dataset_path: Path | str | None = None,
    bootstrap_iterations: int = 1000,
) -> pd.DataFrame:
    """Rebuild a combined comparison table from saved prediction CSVs."""
    output_path = Path(output_dir)
    prediction_paths = []
    minilm_path = output_path / "minilm_predictions.csv"
    if minilm_path.exists():
        prediction_paths.append(minilm_path)
    prediction_paths.extend(sorted(output_path.glob("*/predictions.csv")))

    dataset = None
    if dataset_path is not None and Path(dataset_path).exists():
        dataset = pd.read_csv(dataset_path)[
            ["date", "user_difficulty_calibrated"]
        ].drop_duplicates("date")

    rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        frame = pd.read_csv(prediction_path)
        required = {"model", "split", "expert_difficulty", "prediction"}
        if not required.issubset(frame.columns):
            continue
        if dataset is not None and "user_difficulty_calibrated" not in frame.columns:
            frame = frame.merge(dataset, on="date", how="left")
        for (model_name, split), group in frame.groupby(["model", "split"]):
            y_true = group["expert_difficulty"].to_numpy(dtype=float)
            y_pred = group["prediction"].to_numpy(dtype=float)
            rows.append(
                _metric_row(
                    model_name=model_name,
                    split=split,
                    y_true=y_true,
                    y_pred=y_pred,
                    bootstrap_iterations=bootstrap_iterations,
                )
            )
            if "user_difficulty_calibrated" in group.columns:
                user_group = group.dropna(subset=["user_difficulty_calibrated"])
                if not user_group.empty:
                    user_rows.append(
                        _metric_row(
                            model_name=model_name,
                            split=split,
                            y_true=user_group[
                                "user_difficulty_calibrated"
                            ].to_numpy(dtype=float),
                            y_pred=user_group["prediction"].to_numpy(dtype=float),
                            bootstrap_iterations=bootstrap_iterations,
                        )
                    )

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        split_order = {"validation": 0, "test": 1, "all": 2}
        comparison["_split_order"] = comparison["split"].map(split_order).fillna(99)
        comparison = (
            comparison.sort_values(["model", "_split_order"])
            .drop(columns=["_split_order"])
            .reset_index(drop=True)
        )
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    with (output_path / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison.to_dict(orient="records"), file, indent=2)
        file.write("\n")
    user_comparison = pd.DataFrame(user_rows)
    if not user_comparison.empty:
        user_comparison["_split_order"] = user_comparison["split"].map(split_order).fillna(99)
        user_comparison = (
            user_comparison.sort_values(["model", "_split_order"])
            .drop(columns=["_split_order"])
            .reset_index(drop=True)
        )
    user_comparison.to_csv(output_path / "model_comparison_vs_user.csv", index=False)
    with (output_path / "model_comparison_vs_user.json").open("w", encoding="utf-8") as file:
        json.dump(user_comparison.to_dict(orient="records"), file, indent=2)
        file.write("\n")
    return comparison
