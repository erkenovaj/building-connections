# Connections Difficulty Estimation

This folder contains the historical puzzle data plus a self-contained workflow
for building a difficulty-estimation dataset, analyzing it, and comparing model
families.

## Build Artifacts

```bash
.venv/bin/python -m difficulty_estimation.build_artifacts --strict-counts
```

Outputs:

- `artifacts/joint_config.json`
- `artifacts/connections_difficulty_dataset.csv`
- `artifacts/summary.json`

The dataset joins expert ratings by normalized date, not by puzzle JSON `id`.
Image cards use `image_alt_text` as the item text and keep `image_url` metadata
in the item table used internally.

## Analyze

```bash
.venv/bin/python -m difficulty_estimation.analyze --strict-counts
```

Outputs CSV tables in `reports/tables/` and PNG figures in `reports/figures/`.

## Train Models

```bash
.venv/bin/python -m difficulty_estimation.train_models --models minilm
```

The default model command attempts MiniLM plus the optional LoRA models:

```bash
.venv/bin/python -m difficulty_estimation.train_models
```

ModernBERT and Gemma are skipped with a recorded reason when optional
dependencies, model downloads, or gated Hugging Face access are unavailable.
The app deployment dependency files are intentionally not changed; ML-only
packages are listed in `requirements-ml.txt`.

Model labels are expert difficulty scores. The user-derived score is used
for analysis and secondary evaluation, not as a model input feature:

```text
user_difficulty_raw = averageMistakes + failure_rate
user_difficulty_calibrated =
  zscore(user_difficulty_raw on expert-labeled rows)
  * std(expert_difficulty)
  + mean(expert_difficulty)
```

Rebuild
comparison tables from saved predictions with:

```bash
.venv/bin/python -m difficulty_estimation.train_models --from-predictions
```

This writes both `model_comparison.csv` against expert scores and
`model_comparison_vs_user.csv` against the calibrated user-difficulty score.
