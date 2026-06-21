# LLM Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a solo "Agent" play mode where a chosen LLM solves the current Connections board one group at a time, with binary correct/wrong feedback under the 4-mistake budget, shown as a live panel trace.

**Architecture:** Browser stays game-master (holds `currentPuzzle.categories`); a new stateless `POST /api/llm/guess` returns exactly one next group from the remaining terms. A browser loop (`runLlmAgent`) sends remaining terms + prior wrong guesses, scores each returned group locally via the existing pure `evaluateGuess`, updates mistakes/score, and re-renders the trace until solved or out of mistakes.

**Tech Stack:** Vanilla JS ES modules (frontend), Python `BaseHTTPRequestHandler` Vercel serverless (backend), OllamaFreeAPI client, `node --test` + `unittest` for tests.

---

## File Structure

- `js/game-rules.js` — add pure `remainingTerms(puzzle, solvedDifficulties)`. Pure rules live here, already imported by tests.
- `api/llm/_ollamafree.py` — add pure `normalize_one_guess(parsed, remaining)` alongside existing `normalize_guesses`.
- `api/llm/guess.py` — new stateless endpoint, mirrors `solve.py`.
- `js/api.js` — add `runLlmGuess(...)` client.
- `js/game.js` — import the two pure helpers; add `runLlmAgent`, `renderAgentTrace`, `startAgentGame`; new state flags; panel + welcome event wiring; `loadLlmModels` mirrors options into the welcome select; `startSolo` kickoff hook.
- `index.html` — welcome "Agent" card + panel "Agent" button.
- `tests/game-rules.test.js` — tests for `remainingTerms`.
- `tests/test_llm_guess.py` — new tests for `normalize_one_guess`.

---

## Task 1: Pure `remainingTerms` helper

**Files:**

- Modify: `js/game-rules.js`
- Test: `tests/game-rules.test.js`

- [ ] **Step 1: Write the failing test**

Append to `tests/game-rules.test.js`. The file already imports from `../js/game-rules.js` (line 7-13); add `remainingTerms` to that import list, then add this block at the end of the file:

```javascript
describe("remainingTerms (agent loop)", () => {
  it("returns the full board when nothing is solved", () => {
    const out = remainingTerms(validPuzzle, new Set());
    assert.strictEqual(out.length, 16);
  });

  it("drops members of solved categories", () => {
    const out = remainingTerms(validPuzzle, new Set(["easy"]));
    assert.strictEqual(out.length, 12);
    assert.ok(!out.includes("A1"));
    assert.ok(out.includes("B1"));
  });

  it("returns empty when every category is solved", () => {
    const out = remainingTerms(
      validPuzzle,
      new Set(["easy", "medium", "hard", "harder"]),
    );
    assert.strictEqual(out.length, 0);
  });

  it("tolerates unknown difficulty keys", () => {
    const out = remainingTerms(validPuzzle, new Set(["nope"]));
    assert.strictEqual(out.length, 16);
  });
});
```

The import line becomes:

```javascript
import {
  evaluateGuess,
  getGameStatus,
  validatePuzzleStructure,
  CONFIG,
  setsAreEqual,
  remainingTerms,
} from "../js/game-rules.js";
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/game-rules.test.js`
Expected: FAIL — `remainingTerms is not a function` (or import undefined).

- [ ] **Step 3: Write minimal implementation**

Add to `js/game-rules.js` after `evaluateGuess` (after line 42):

```javascript
/**
 * Compute board terms not belonging to any already-solved category.
 * @param {Object} puzzle - Puzzle object with board and categories
 * @param {Set<string>} solvedDifficulties - Difficulty keys already solved
 * @returns {Array<string>} - Remaining board terms
 */
export function remainingTerms(puzzle, solvedDifficulties) {
  const solved = new Set();
  for (const difficulty of solvedDifficulties) {
    const category = puzzle.categories[difficulty];
    if (!category || !Array.isArray(category.members)) continue;
    for (const member of category.members) {
      solved.add(member);
    }
  }
  return puzzle.board.filter((term) => !solved.has(term));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/game-rules.test.js`
Expected: PASS — all `remainingTerms` cases plus the existing suite.

- [ ] **Step 5: Commit**

```bash
git add js/game-rules.js tests/game-rules.test.js
git commit -m "feat: add remainingTerms helper for agent loop"
```

---

## Task 2: Pure `normalize_one_guess` backend helper

**Files:**

- Modify: `api/llm/_ollamafree.py`
- Test: `tests/test_llm_guess.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_guess.py`:

```python
"""Tests for single-guess normalization used by the agent loop endpoint."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "api", "llm"))

from _ollamafree import normalize_one_guess


class NormalizeOneGuessTest(unittest.TestCase):
    def test_valid_guess(self):
        parsed = {"guess": {"label": "Planets", "items": ["MARS", "VENUS", "SATURN", "MERCURY"]}}
        remaining = ["MARS", "VENUS", "SATURN", "MERCURY", "BRIDGE", "POKER"]
        result = normalize_one_guess(parsed, remaining)
        self.assertEqual(result, {"label": "Planets", "items": ["MARS", "VENUS", "SATURN", "MERCURY"]})

    def test_term_not_remaining_rejected(self):
        parsed = {"guess": {"items": ["MARS", "VENUS", "SATURN", "PLUTO"]}}
        remaining = ["MARS", "VENUS", "SATURN", "MERCURY"]
        self.assertIsNone(normalize_one_guess(parsed, remaining))

    def test_fewer_than_four_returns_none(self):
        parsed = {"guess": {"items": ["MARS", "VENUS"]}}
        self.assertIsNone(normalize_one_guess(parsed, ["MARS", "VENUS", "SATURN", "MERCURY"]))

    def test_missing_guess_returns_none(self):
        self.assertIsNone(normalize_one_guess({"notes": "hi"}, ["A", "B", "C", "D"]))

    def test_flat_items_shape_accepted(self):
        parsed = {"items": ["A", "B", "C", "D"], "label": "Letters"}
        result = normalize_one_guess(parsed, ["A", "B", "C", "D", "E"])
        self.assertEqual(result, {"label": "Letters", "items": ["A", "B", "C", "D"]})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp \
  .venv/bin/python -m unittest discover -s tests -p 'test_llm_guess.py'
```

Expected: FAIL — `ImportError: cannot import name 'normalize_one_guess'`.

- [ ] **Step 3: Write minimal implementation**

Add to `api/llm/_ollamafree.py` after `normalize_guesses` (after line 81):

```python
def normalize_one_guess(parsed, remaining):
    remaining_set = set(remaining)
    guess = None
    if isinstance(parsed, dict):
        candidate = parsed.get("guess")
        if isinstance(candidate, dict):
            guess = candidate
        elif isinstance(parsed.get("items"), list):
            guess = parsed
    if not isinstance(guess, dict):
        return None

    raw_items = guess.get("items")
    if not isinstance(raw_items, list):
        raw_items = guess.get("words")
    if not isinstance(raw_items, list):
        return None

    items = []
    for item in raw_items:
        term = str(item or "").strip()
        if term and term in remaining_set and term not in items:
            items.append(term)
        if len(items) == 4:
            break
    if len(items) != 4:
        return None

    label = str(guess.get("label") or guess.get("category") or "Guess")[:80]
    return {"label": label, "items": items}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp \
  .venv/bin/python -m unittest discover -s tests -p 'test_llm_guess.py'
```

Expected: PASS — 5 tests OK.

- [ ] **Step 5: Commit**

```bash
git add api/llm/_ollamafree.py tests/test_llm_guess.py
git commit -m "feat: add normalize_one_guess for single-group agent guesses"
```

---

## Task 3: `POST /api/llm/guess` endpoint

**Files:**

- Create: `api/llm/guess.py`

- [ ] **Step 1: Write the endpoint**

Create `api/llm/guess.py`:

```python
from http.server import BaseHTTPRequestHandler
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from _ollamafree import (
    default_model,
    extract_json,
    get_client,
    normalize_one_guess,
    read_json,
    write_json,
)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = read_json(self)
        except Exception:
            write_json(self, 400, {"error": "Invalid JSON request body."})
            return

        remaining = (
            [str(item) for item in body.get("remaining", [])]
            if isinstance(body.get("remaining"), list)
            else []
        )
        if len(remaining) < 4:
            write_json(self, 400, {"error": "At least 4 remaining terms are required."})
            return

        try:
            num_categories = max(1, min(4, int(body.get("numCategories") or 4)))
        except Exception:
            num_categories = 4

        raw_history = body.get("history") if isinstance(body.get("history"), list) else []
        tried = []
        for entry in raw_history:
            if isinstance(entry, dict) and isinstance(entry.get("items"), list):
                tried.append([str(term) for term in entry["items"]])

        model = str(body.get("model") or default_model()).strip() or default_model()

        prompt = "\n".join(
            [
                "You are playing a Connections-style puzzle.",
                "Find ONE group of exactly 4 terms that share a theme.",
                "Use only exact terms from the remaining list. Do not invent terms.",
                "Do not repeat any group already tried; those were wrong.",
                "Return strict JSON only with this schema:",
                '{"guess":{"label":"short label","items":["term","term","term","term"]},"notes":"one short sentence"}',
                "",
                f"Remaining terms: {json.dumps(remaining)}",
                f"Already tried (wrong): {json.dumps(tried)}",
                f"Groups still to find: {num_categories}",
            ]
        )

        try:
            client = get_client()
            response = client.chat(
                prompt=prompt,
                model=model,
                temperature=float(body.get("temperature") or 0.2),
                num_predict=400,
            )
            parsed = extract_json(response)
            guess = normalize_one_guess(parsed, remaining) if parsed else None
            write_json(
                self,
                200,
                {
                    "model": model,
                    "guess": guess,
                    "notes": str(parsed.get("notes", ""))[:300] if isinstance(parsed, dict) else "",
                    "raw": str(response),
                },
            )
        except Exception as error:
            write_json(
                self,
                500,
                {
                    "error": "Could not run OllamaFreeAPI model solver.",
                    "details": str(error),
                },
            )
```

- [ ] **Step 2: Verify it compiles**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp .venv/bin/python -m py_compile api/llm/guess.py
```

Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add api/llm/guess.py
git commit -m "feat: add /api/llm/guess single-group endpoint"
```

---

## Task 4: `runLlmGuess` API client

**Files:**

- Modify: `js/api.js`

- [ ] **Step 1: Add the client function**

Add to `js/api.js` after `runLlmSolver` (after line 66):

```javascript
export async function runLlmGuess({
  remaining,
  numCategories,
  history,
  model,
}) {
  const res = await fetch(`${getBase()}/api/llm/guess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ remaining, numCategories, history, model }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

- [ ] **Step 2: Verify syntax**

Run: `node --check js/api.js`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add js/api.js
git commit -m "feat: add runLlmGuess API client"
```

---

## Task 5: Agent loop in `game.js`

**Files:**

- Modify: `js/game.js`

- [ ] **Step 1: Import the pure helpers**

Change the imports near the top of `js/game.js` (line 5 area). After:

```javascript
import * as api from "./api.js";
```

add:

```javascript
import { evaluateGuess, remainingTerms } from "./game-rules.js";
```

- [ ] **Step 2: Add state flags**

In the constructor/init where `this.llmBenchmark = null;` and `this.isLlmRunning = false;` are set (lines 329-330), add right after:

```javascript
this.agentModel = null;
this.pendingAgentRun = false;
```

- [ ] **Step 3: Add `renderAgentTrace` and `runLlmAgent` methods**

Add both methods inside the class, right before `runLlmBenchmark()` (before line 1579):

```javascript
    renderAgentTrace(rows, mistakes, thinking) {
        const trace = document.getElementById('llm-trace');
        if (!trace) return;
        const lines = rows.map((row, index) => {
            const label = this.escapeHtml(row.label || `Guess ${index + 1}`);
            const verdict = row.correct ? `Matched ${this.escapeHtml(row.matchedName)}` : 'Miss';
            const items = this.escapeHtml((row.items || []).join(', '));
            return `
                <div class="llm-guess ${row.correct ? 'correct' : 'incorrect'}">
                    <div class="llm-guess-title">${label}<span>${verdict}</span></div>
                    <div>${items}</div>
                </div>
            `;
        });
        if (thinking) {
            lines.push('<div class="llm-guess"><div class="llm-guess-title">thinking…</div></div>');
        }
        const header = `<p class="mono-label">Mistakes ${mistakes}/${CONFIG.MAX_MISTAKES}</p>`;
        trace.innerHTML = header + (lines.join('') || '<p>Agent is starting…</p>');
    }

    async runLlmAgent() {
        if (!this.currentPuzzle || this.isLlmRunning) return;
        const panel = document.getElementById('llm-bench-panel');
        if (panel) panel.open = true;
        const agentBtn = document.getElementById('llm-agent-btn');
        const runBtn = document.getElementById('llm-run-btn');
        const modelInput = document.getElementById('llm-model-input');
        const model = this.agentModel
            || (modelInput && modelInput.value && modelInput.value.trim())
            || 'llama3.2:3b';

        this.isLlmRunning = true;
        if (agentBtn) agentBtn.disabled = true;
        if (runBtn) runBtn.disabled = true;

        const numCategories = this.currentPuzzle.numCategories;
        const solvedDifficulties = new Set();
        const history = [];
        const rows = [];
        let mistakes = 0;
        let solved = 0;
        let score = 0;
        let elapsed = 0;
        const secondsPerGuess = 8;

        try {
            while (mistakes < CONFIG.MAX_MISTAKES && solved < numCategories) {
                this.setLlmStatus(`solving · turn ${rows.length + 1}`, 'running');
                this.renderAgentTrace(rows, mistakes, true);

                const remaining = remainingTerms(this.currentPuzzle, solvedDifficulties);
                let guess = null;
                try {
                    const res = await api.runLlmGuess({ remaining, numCategories, history, model });
                    guess = res && res.guess ? res.guess : null;
                } catch (_) {
                    guess = null;
                }
                elapsed += secondsPerGuess;

                if (!guess || !Array.isArray(guess.items) || guess.items.length !== 4) {
                    mistakes++;
                    score = Math.max(0, Math.round(score - secondsPerGuess));
                    rows.push({ label: 'Unusable output', items: (guess && Array.isArray(guess.items)) ? guess.items : [], correct: false, matchedName: null });
                    this.renderAgentTrace(rows, mistakes, false);
                    continue;
                }

                const verdict = evaluateGuess(this.currentPuzzle, guess.items);
                if (verdict.correct) {
                    solvedDifficulties.add(verdict.difficulty);
                    solved++;
                    const rawPoints = 12000 / secondsPerGuess - elapsed;
                    score = Math.max(0, Math.round(score + Math.max(100, Math.round(rawPoints))));
                    rows.push({ label: guess.label || verdict.matchedCategory.name, items: guess.items, correct: true, matchedName: verdict.matchedCategory.name });
                } else {
                    mistakes++;
                    history.push({ items: guess.items });
                    score = Math.max(0, Math.round(score - secondsPerGuess));
                    rows.push({ label: guess.label || 'Guess', items: guess.items, correct: false, matchedName: null });
                }
                this.renderAgentTrace(rows, mistakes, false);
            }

            const won = solved === numCategories && mistakes < CONFIG.MAX_MISTAKES;
            this.llmBenchmark = { score, solved, mistakes, won, rows };
            this.setText('llm-score-value', String(score));
            this.setText('llm-match-value', `${solved}/${numCategories}`);
            this.updateLlmHumanScore();
            this.setLlmStatus(won ? 'Solved' : 'Out of mistakes', won ? 'ready' : 'error');
        } finally {
            this.isLlmRunning = false;
            if (agentBtn) agentBtn.disabled = false;
            if (runBtn) runBtn.disabled = false;
            this.agentModel = null;
        }
    }
```

- [ ] **Step 4: Verify syntax**

Run: `node --check js/game.js`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add js/game.js
git commit -m "feat: add interactive runLlmAgent loop"
```

---

## Task 6: Welcome card, panel button, and wiring

**Files:**

- Modify: `index.html`
- Modify: `js/game.js`

- [ ] **Step 1: Add the welcome Agent card**

In `index.html`, inside `.welcome-right`, after the room panel `</div>` that closes at line 76 (before the closing `</div>` of `.welcome-right` at line 77), insert:

```html
<div class="welcome-actions-panel agent-panel">
  <h3 class="welcome-panel-title">Agent (model plays)</h3>
  <div class="welcome-agent-controls">
    <label for="agent-model-input">Model</label>
    <select id="agent-model-input">
      <option value="llama3.2:3b">llama3.2:3b</option>
    </select>
    <button type="button" class="welcome-btn welcome-agent" id="welcome-agent">
      Start agent game
    </button>
  </div>
</div>
```

- [ ] **Step 2: Add the panel Agent button**

In `index.html`, in the `.llm-run-row` (line 175-181), after the `llm-run-btn` button (line 180) add:

```html
<button type="button" id="llm-agent-btn" class="llm-run-btn">Agent</button>
```

- [ ] **Step 3: Wire the buttons in `bindEvents`**

In `js/game.js` `bindEvents()`, after the `llm-refresh-btn` binding (line 790) add:

```javascript
const llmAgentBtn = document.getElementById("llm-agent-btn");
if (llmAgentBtn)
  llmAgentBtn.addEventListener("click", () => this.runLlmAgent());
const welcomeAgentBtn = document.getElementById("welcome-agent");
if (welcomeAgentBtn)
  welcomeAgentBtn.addEventListener("click", () => this.startAgentGame());
```

- [ ] **Step 4: Add `startAgentGame` and hook `startSolo`**

In `js/game.js`, add this method immediately before `startSolo()` (before line 634):

```javascript
    startAgentGame() {
        const select = document.getElementById('agent-model-input');
        this.agentModel = (select && select.value && select.value.trim()) || 'llama3.2:3b';
        this.pendingAgentRun = true;
        this.startSolo();
    }
```

Then change the `requestAnimationFrame` block at the end of `startSolo()` (lines 662-668) from:

```javascript
requestAnimationFrame(async () => {
  try {
    await this.startNewGame();
  } finally {
    this.hideLoading();
  }
});
```

to:

```javascript
requestAnimationFrame(async () => {
  try {
    await this.startNewGame();
  } finally {
    this.hideLoading();
    if (this.pendingAgentRun) {
      this.pendingAgentRun = false;
      this.runLlmAgent();
    }
  }
});
```

- [ ] **Step 5: Mirror model options into the welcome select**

In `js/game.js` `loadLlmModels()`, immediately before `if (models.length > 0) {` (line 1471) add:

```javascript
const agentInput = document.getElementById("agent-model-input");
if (agentInput) {
  agentInput.innerHTML = input.innerHTML;
  agentInput.value = input.value;
}
```

- [ ] **Step 6: Verify syntax**

Run: `node --check js/game.js`
Expected: no output, exit 0.

- [ ] **Step 7: Manual smoke test**

Run: `npm start` (vercel dev), open the app.

- Welcome screen shows the "Agent (model plays)" card with a model select and "Start agent game".
- Pick a Mode (e.g. Basic), click "Start agent game" → game screen loads, LLM panel opens, trace fills turn-by-turn with ✓/✗ rows, "Mistakes n/4" updates, ends with "Solved" or "Out of mistakes" and a model score.
- The in-game panel "Agent" button re-runs on the current board.

Expected: agent plays the board end-to-end without touching the answer key on the backend.

- [ ] **Step 8: Commit**

```bash
git add index.html js/game.js
git commit -m "feat: add Agent play mode entry on welcome screen and panel"
```

---

## Self-Review Notes

- **Spec coverage:** binary feedback (Task 5 `evaluateGuess`), any-miss-no-retry incl. unusable output as a mistake (Task 5 loop), stop at `MAX_MISTAKES`/all-solved (Task 5 `while`), solo-only entry inheriting Mode+templates (Task 6 via `startSolo`), panel trace UI reusing `.llm-guess` (Task 5 `renderAgentTrace`), stateless backend never sees the key (Task 3 only gets `remaining`). Room/animation/retries/one-away explicitly out of scope — no tasks, as intended.
- **Type consistency:** `runLlmGuess` returns `{ guess: {label, items} | null }` (Task 3/4); consumed in Task 5. `remainingTerms(puzzle, Set)` (Task 1) matches the call in Task 5. `evaluateGuess` returns `{correct, matchedCategory, difficulty}` (existing) — used in Task 5. `normalize_one_guess(parsed, remaining)` (Task 2) used in Task 3.
- **No placeholders:** every code step shows full content.
