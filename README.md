# Connections, Reasoning & Evals

A configurable Connections-style game for combinatorial reasoning, multiplayer rooms, and model evaluation. Games are generated server-side with an OR-Tools CP-SAT sampler and are accepted only when the intended solution is unique.

## Game Modes

- **Easy:** Four groups of four. Selected category templates must be mutually disjoint.
- **Basic:** Four groups of four. Category intersections and red herrings are allowed, but the final board must have exactly one complete solution.
- **Advanced:** Three groups of four plus four decoys. Every unselected category must have fewer than four terms on the board, so the decoys cannot form another group alone or together with solution terms.

The game supports solo play and multiplayer rooms. Room games use deterministic round seeds so every player receives the same sampled board, and scores are stored only in the room-scoped leaderboard.

## Features

- Solo and multiplayer room modes
- Easy, Basic, and Advanced sampling priors
- Custom item/tag JSON configs
- Timer, scoring, hints, definitions, and dictionary
- Room-scoped leaderboard and CSV export
- OllamaFreeAPI model playtesting with structured guesses and reasoning output
- The original dog interface artifact, integrated into the current lab-style UI

## Python Setup

Create the ignored repository-local virtual environment and install dependencies:

```bash
uv venv .venv --python 3.12
uv sync
```

The main Python dependencies are:

- `ortools` for CP-SAT sampling and uniqueness verification
- `ollamafreeapi` for model playtesting

## Reusable Sampler API

The sampler is a standalone package under `connections_sampler/`.

```python
import json

from connections_sampler import SamplingParameters, sample_game

with open("configs/category-templates-new.json", encoding="utf-8") as file:
    config = json.load(file)

result = sample_game(
    config,
    mode="basic",
    parameters=SamplingParameters(seed="experiment-42"),
)

if result.ok:
    game = result.game.to_dict()
else:
    print(result.status, result.reason, result.diagnostics)
```

Generic Basic-style `P x Q` games use the same model:

```python
from connections_sampler import sample_basic_game

result = sample_basic_game(
    config,
    num_categories=6,
    items_per_category=3,
    parameters=SamplingParameters(seed=123),
)
```

The generic interface changes the logical board dimensions. The current webpage remains a 16-tile game.

### Result Contract

Sampling returns a `SamplingResult` with:

- `status`: `success`, `infeasible`, `timeout`, or `invalid_config`
- `game`: a `SampledGame` on success
- `reason`: a human-readable failure explanation
- `diagnostics`: seed, attempts, rejected ambiguous candidates, and solver status

The baseline method is `randomized_feasible`. It uses seeded randomized CP-SAT search and one worker for reproducibility. It is intentionally not described as uniform sampling.

## Config Format

The preferred config is an array of item entries:

```json
[
  {
    "name": "Instrumental Convergence",
    "description": "A tendency for many goals to imply similar useful subgoals.",
    "tags": ["Technical Failure Modes"]
  }
]
```

Each item may have multiple tags. The library also accepts category-oriented entries with `name` and `members`.

## Sampler HTTP API

Vercel serves the sampler at:

```text
POST /api/game/sample
```

Example body:

```json
{
  "mode": "advanced",
  "seed": "room-abc-round-1",
  "config": [
    {
      "name": "Term",
      "description": "Definition",
      "tags": ["Category"]
    }
  ]
}
```

Old links and saved rooms that use `normal` are read as Basic mode. New code should use `basic`. External callers may also send `numCategories` and `itemsPerCategory` in Basic mode.

## Running Locally

Install JavaScript dependencies:

```bash
npm install
```

Run the full Vercel application:

```bash
npm start
```

Vercel normally prints a URL such as:

```text
http://localhost:3000
```

The full Vercel server is required for sampling, rooms, leaderboards, and LLM endpoints. A plain static file server can render the interface but cannot start a game because CP-SAT sampling is server-side.

For room APIs, add Turso credentials to `.env.local`:

```bash
TURSO_DATABASE_URL=...
TURSO_AUTH_TOKEN=...
```

The optional model default is:

```bash
LLM_MODEL=llama3.2:3b
```

## Command-Line Sampling

```bash
.venv/bin/python -m connections_sampler --mode basic --seed 42
.venv/bin/python -m connections_sampler --mode advanced --seed room-1
.venv/bin/python -m connections_sampler \
  --mode basic \
  --num-categories 5 \
  --items-per-category 3
```

## Deploying To Vercel

1. Create a Turso database.
2. Configure `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` in Vercel.
3. Optionally configure `LLM_MODEL`.
4. Deploy the repository.

Vercel installs Python dependencies from `pyproject.toml` or `requirements.txt`. The sampler runs as a Python Function, so no persistent proxy process is required.

## Project Structure

```text
ai-connections-vercel/
├── api/
│   ├── game/sample.py          # Server-side CP-SAT sampling endpoint
│   ├── llm/                    # OllamaFreeAPI model endpoints
│   └── room/                   # Room results and room leaderboard
├── connections_sampler/
│   ├── __init__.py             # Public library API
│   ├── __main__.py             # Command-line interface
│   └── sampler.py              # Types, config handling, CP-SAT model, validation
├── configs/
├── css/
├── images/
├── js/
├── tests/
│   ├── game-rules.test.js
│   └── test_sampler.py
├── index.html
├── pyproject.toml
└── requirements.txt
```

## Testing

Python sampler tests:

```bash
PYTHONPYCACHEPREFIX=/private/tmp \
  .venv/bin/python -m unittest discover -s tests -p 'test_sampler.py'
```

JavaScript game-rule tests:

```bash
node --test tests/game-rules.test.js
```

Syntax checks:

```bash
PYTHONPYCACHEPREFIX=/private/tmp \
  .venv/bin/python -m py_compile connections_sampler/*.py api/game/sample.py
node --check js/game.js
```
