# Difficulty Estimation Instructions

These instructions apply only to `difficulty_estimation/` and its descendants.
Symmetry and invariance experiments belong in `invariance_audit/`; do not add
their code, results, or scientific claims here.

## Environment

- Run commands from the repository root.
- Use `uv` and one local virtual environment at the project root. Do not create
  another environment inside `difficulty_estimation/`.
- Create and sync the environment with:

  ```bash
  uv venv .venv --python 3.12
  uv sync
  uv pip install --python .venv/bin/python -r difficulty_estimation/requirements-ml.txt
  ```

- Invoke project commands with `.venv/bin/python`.

## Data and workflow

- Treat `puzzle/`, `guesses/`, `stats/`,
  `stats-arch-2026-01-28/`, and `connections_rater_difficulty.csv` as source
  data. Do not rewrite or reformat them.
- `stats/` is active data; `stats-arch-2026-01-28/` is an archive and must not
  be counted as additional observations.
- Join games, ratings, and logs by normalized date, following
  `workflow/data.py`.
- Rebuild and validate the full corpus with:

  ```bash
  .venv/bin/python -m difficulty_estimation.build_artifacts --strict-counts
  ```

- Heavy model downloads and GPU training run on the separate GPU server.
  Locally, prepare code and configs and run bounded CPU smoke tests.
