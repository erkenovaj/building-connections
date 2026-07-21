# Invariance Audit Instructions

These instructions apply to `invariance_audit/` and its descendants.

## Environment

- Run commands from the repository root.
- Use `uv` with the single repository-local `.venv`; do not create an
  environment inside this folder.
- Set up the environment with:

  ```bash
  uv venv .venv --python 3.12
  uv sync
  ```

- Invoke Python with `.venv/bin/python`.

## Experimental boundary

- Follow `DESIGN.md`. This is a solution-invariance experiment, not a
  difficulty-estimation experiment.
- The only permitted source from `difficulty_estimation/` is
  `puzzle/*-puzzle.json`. Do not load expert ratings, derived difficulty
  features, player statistics, guess logs, or trained difficulty models.
- Extract only a game ID, the 16 terms in canonical position order, and the
  four exact solution groups. Never include the solution in model input.
- Treat each stored game as one latent instance. Every representation in an
  orbit must contain exactly the same terms and solution membership; only the
  registered permutation or rendering may differ.
- Use fixed prompts, decoding settings, model revisions, permutations,
  viewports, fonts, and screenshot settings within an experiment.
- Canonicalize group and member order before evaluation. Category-title text is
  not part of the correctness criterion.

## Local game engine

- Reuse the existing game contract and renderer in `js/game-rules.js` and
  `js/game.js`. Add a narrow fixture-loading path; do not route stored games
  through the random sampler.
- Validate every fixture with `validatePuzzleStructure` before rendering.
- Screenshots must come from the real local webpage's `#game-board` after fonts
  and layout settle. Hide or crop dynamic UI such as timers, scores, hints,
  selections, and modals.
- Re-render each permutation from terms. Do not rotate or rearrange screenshot
  pixels after capture.

## Execution boundary

- Heavy inference runs on the separate GPU server. Locally, implement adapters,
  validation, parsing, deterministic manifests, and small rendering smoke tests.
- Jobs must write incrementally and resume from a manifest without resampling
  permutations.
