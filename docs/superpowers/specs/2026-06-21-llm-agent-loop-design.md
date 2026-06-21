# LLM Agent Loop — Design

Date: 2026-06-21
Status: Draft for review

## Goal

Add an interactive LLM agent that plays a solo Connections game one group at a
time: guess a group of 4 → get binary feedback (correct / wrong) → revise →
repeat, under the same 4-mistake budget a human plays with. The agent is a
first-class play mode alongside Solo and Room, not a hidden button inside a
running game.

Today the repo already has a one-shot benchmark (`/api/llm/solve` +
`runLlmBenchmark`) that asks the model for all groups at once and scores them.
This feature is the interactive upgrade of that benchmark.

## Decisions (locked with user)

- **Feedback granularity:** binary correct / wrong only. The game gives a human
  no "one away" hint (`submitGuess` confirms this), so the agent gets the same.
- **Wrong guess:** any miss = one mistake, no retry. A valid-but-wrong group and
  an unusable/unparseable model output both cost one mistake and the loop moves
  on. Stop when `mistakes >= CONFIG.MAX_MISTAKES (4)` or all groups solved.
- **Scope:** solo only. Room / multiplayer leaderboard is out of scope.
- **UI surface:** panel trace only, no board-tile animation.
- **Modes:** agent inherits the selected Mode (Easy / Basic / Advanced) and
  custom category templates from the welcome screen, exactly like Solo. No
  agent-specific board logic — it consumes whatever `currentPuzzle` is loaded.

## Architecture

Browser is the game-master and holds ground truth
(`currentPuzzle.categories`). The backend never sees the answer key, so the
model cannot cheat. Each turn:

1. Browser sends the **remaining** board terms (full board minus already-solved
   groups) plus the history of prior wrong guesses to the backend.
2. Backend prompts the model for **exactly one** next group of 4 and returns it.
3. Browser scores that single guess against `currentPuzzle.categories`, gives
   binary feedback, updates mistakes/solved, appends a trace row, loops.

The backend stays stateless; all game state lives in the browser.

## Backend — `api/llm/guess.py` (new)

POST endpoint, mirrors the shape of `solve.py`. Reuses `_ollamafree` helpers
(`read_json`, `write_json`, `get_client`, `default_model`, `extract_json`).

Request:

```json
{
  "remaining": ["TERM", "..."], // current board terms not yet solved
  "numCategories": 4, // total groups in this puzzle (for context)
  "history": [
    // prior wrong guesses this game
    { "items": ["A", "B", "C", "D"] }
  ],
  "model": "llama3.2:3b",
  "temperature": 0.2
}
```

Prompt asks for ONE group of exactly 4 terms drawn only from `remaining`, must
not repeat any group listed in `history`, strict JSON:

```json
{
  "guess": { "label": "short label", "items": ["A", "B", "C", "D"] },
  "notes": "one sentence"
}
```

Response:

```json
{ "model": "...", "guess": { "label": "...", "items": [...] }, "notes": "...", "raw": "..." }
```

Normalization: add `normalize_one_guess(parsed, remaining)` to `_ollamafree.py`
(or reuse `normalize_guesses` with `max_guesses=1` against the `remaining` set).
It validates 4 distinct terms all present in `remaining`. If it can't produce a
valid 4-term guess, return `{ "guess": null, ... }` — the browser treats null as
a missed turn (one mistake, no retry).

## Frontend

### `js/api.js`

Add:

```js
export async function runLlmGuess({
  remaining,
  numCategories,
  history,
  model,
}) {
  // POST /api/llm/guess, returns { guess, notes, raw, model }
}
```

### `js/game.js`

**Extract a shared single-guess matcher** from `scoreModelGuesses`. The inner
match loop (lines ~1507–1514) becomes:

```js
matchGuessToCategory(items, solvedDifficulties) {
  const guessSet = new Set(items);
  for (const [difficulty, category] of Object.entries(this.currentPuzzle.categories)) {
    if (solvedDifficulties.has(difficulty)) continue;
    if (this.setsAreEqual(new Set(category.members), guessSet)) {
      return { difficulty, category };
    }
  }
  return null;
}
```

`scoreModelGuesses` is refactored to call this helper (behavior unchanged) so
the one-shot benchmark and the agent loop share one matcher.

**New `runLlmAgent()` loop:**

- Guard: `this.currentPuzzle && !this.isLlmRunning`.
- State: `solvedDifficulties = new Set()`, `mistakes = 0`, `score = 0`,
  `history = []`, `elapsed = 0`.
- Compute `remaining` = board terms whose category difficulty is not in
  `solvedDifficulties`.
- Loop while `mistakes < CONFIG.MAX_MISTAKES` and
  `solvedDifficulties.size < numCategories`:
  1. `setLlmStatus('solving · turn N', 'running')`, render an in-flight trace row.
  2. `const { guess } = await api.runLlmGuess({ remaining, numCategories, history, model })`.
  3. If `guess` is null → mistake++, push wrong-row "unusable output", continue.
  4. `const match = this.matchGuessToCategory(guess.items, solvedDifficulties)`.
  5. Correct → add difficulty to solved, recompute `remaining`, score via the
     same `rawPoints` formula as `scoreModelGuesses`, push ✓ row.
  6. Wrong → mistakes++, push guess.items to `history`, score -= secondsPerGuess,
     push ✗ row.
  7. Re-render trace + mistakes dots + metric cards after each turn.
- On exit: set final status (`Solved` / `Out of mistakes`), fill metric cards
  (model score, groups n/4, vs human via `updateLlmHumanScore`).

Scoring reuses the existing constants/formula in `scoreModelGuesses`
(`secondsPerGuess = 8`, `rawPoints = 12000/8 - elapsed`, min 100 per correct,
-8 per miss) so agent and one-shot scores are comparable.

### Welcome screen — `index.html`

Add a third play card in the right column, below Room:

- Header "Agent (model plays)" + "new" badge.
- Model `<select id="agent-model-input">` (populated by the existing
  `loadLlmModels`, reused; default `llama3.2:3b`).
- Button `#welcome-agent` "Start agent game".

On click: read selected model + current Mode + templates (same path Solo uses),
start the solo game, set `this.agentMode = true` and `this.agentModel = model`,
open `#llm-bench-panel`, then call `runLlmAgent()` automatically.

The in-game panel keeps the existing "Run" (one-shot) button and adds an
"Agent" button so a run can also be triggered mid-game on the current board.

### Panel trace UI

Reuse `#llm-bench-panel`, `#llm-trace`, status pill, and metric grid. Per-turn
trace rows: ✓ green (correct, shows matched category name) / ✗ red (wrong) /
muted in-flight ("thinking…"). Mistakes shown as the same 4-dot row, `n/4`.
Final win/loss + score in the metric cards. (Visual matches the mockups shown in
chat.)

## Files touched

- `api/llm/guess.py` — new endpoint.
- `api/llm/_ollamafree.py` — add `normalize_one_guess` (or reuse path).
- `js/api.js` — add `runLlmGuess`.
- `js/game.js` — extract `matchGuessToCategory`, add `runLlmAgent`, welcome-card
  wiring, panel "Agent" button handler.
- `index.html` — welcome Agent card, panel "Agent" button.

## Out of scope

- Room / multiplayer agent participation and leaderboard entries.
- Board-tile animation during agent play.
- Retries on bad output, self-consistency sampling, embeddings, beam search.
- One-away / partial feedback.

## Open questions

None blocking. Ready for implementation plan.
