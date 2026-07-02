# Agentic GRPO: Tool-Loop Episodes + Terminal Reward — Design

Date: 2026-07-02
Status: Draft for review

## Goal

Migrate GRPO training from the contextual-bandit setup (one prompt → one
guess → dense per-guess reward) to full agentic episodes: the policy plays a
whole Connections board multi-turn — guess, get environment feedback, revise —
and is trained on a single terminal reward for the episode. This matches how
the deployed agent (`api/llm/guess.py` loop) actually plays and lets the model
learn recovery behaviour (using wrong-guess feedback) instead of only
first-guess accuracy.

## Context

- Bandit GRPO (`train/grpo_connections.py`) works but optimizes a proxy: best
  single guess from a fresh board. Win rate over a full episode is the real
  objective.
- `connections_gym/env.py` already drives full episodes (`reset`/`step`,
  history, mistakes, win/loss) — the training loop just never uses it
  multi-turn.
- TRL ≥ 1.7 `rollout_func` lets a custom function produce
  `{"prompt_ids", "completion_ids", "logprobs"}` per sample; extra returned
  fields (our `env_mask`) propagate to the reward kwargs and can mask
  env-injected tokens out of the loss. This resolves the main technical risk
  of multi-turn GRPO (not penalizing/reinforcing tokens the model didn't
  produce).
- Этап 0 blocker (gen-configs probe showed valid_rate 0.0) is fixed: the cause
  was `max_new_tokens=256` truncating Qwen3 mid-`<think>` (commit `0a90d49`).
  Gen configs are usable for training.
- Mentor feedback incorporated: raise effective batch via gradient
  accumulation, cosine LR schedule, SFT before RL (STaR), board difficulty
  via empirical win-rate ordering (see below); Saturn-1.5B comparison
  deferred to backlog.

## Decisions (locked with user)

- **Episode reward:** terminal only,
  `R = 1.0·[won] − 0.25·mistakes − 0.02·tool_calls`. No per-step shaping;
  the dense `connections_gym/reward.py` stays for the bandit script and
  probes but does not feed the agentic trainer.
- **SFT-before-RL = STaR / rejection sampling:** sample base Qwen3-1.7B
  episodes on 2-category boards, keep only won trajectories (with `<think>`
  intact), SFT a LoRA on them, then run GRPO on top of that adapter.
- **Curriculum:** 2 → 3 → 4 categories via the existing `--num-categories`
  knob; advance when win-rate probe clears a threshold.
- **Difficulty ordering (mentor's estimator, empirical variant):** the win@k
  probe records per-board win rate; each GRPO stage trains on the easiest
  seeds from the probed pool. Difficulty is thus measured directly for the
  model being trained — no external dataset or learned estimator. This is
  board _selection_, not in-run ordering (the TRL sampler shuffles rows).
  A learned estimator on real NYT games stays in backlog.
- **Gating:** never start a GRPO stage blind — run the win@k probe first; if
  win@k ≈ 0 there is nothing for GRPO to amplify.
- **Hardware:** Kaggle T4 (fp16), LoRA r=16 all-linear as today; MPS + 0.6B
  for local smoke tests.

## Architecture

Three new/changed units, each independently testable:

### 1. Rollout driver — `train/rollout.py` (new)

`play_episode(model, tokenizer, env, seed, gen_kwargs) -> Episode`.

Loop per turn: render prompt via `env` observation → `model.generate` →
`parse_guess` (`connections_gym/prompt.py`) → `env.step(items)` (its dense
per-step reward is ignored; only state/stats are used) → append
feedback message ("correct" / "wrong, N mistakes left") to the running
conversation → repeat until `done`. Unparseable output counts as a mistake
(same as deployment) and the loop continues.

`Episode` carries: full token id sequence (prompt + interleaved model/env
tokens), `env_mask` (1 = model-generated token, 0 = injected env/feedback
token), per-token logprobs from generation, and episode stats (`won`,
`mistakes`, `tool_calls` = number of guesses).

The driver is pure orchestration — no training deps — so it serves the win@k
probe, STaR sampling, and the GRPO `rollout_func` alike.

### 2. Terminal reward — `connections_gym/episode_reward.py` (new)

`episode_reward(won: bool, mistakes: int, tool_calls: int) -> float` with the
formula above as module constants. Trivial, but isolated so the trainer,
probe, and tests share one definition.

### 3. Agentic GRPO trainer — `train/grpo_agentic.py` (new)

Same skeleton as `train/grpo_connections.py` (LoRA config,
`CompactLogCallback`, CLI) but:

- passes a `rollout_func` that calls the rollout driver for each prompt ×
  `num_generations`, returning `prompt_ids` / `completion_ids` / `logprobs` /
  `env_mask` + episode stats;
- reward function reads episode stats from kwargs and applies
  `episode_reward` — no completion parsing in the trainer;
- `env_mask` zeroes env tokens out of the policy-gradient loss;
- defaults per mentor feedback: `--lr-scheduler cosine`,
  `--grad-accum` (gradient_accumulation_steps) exposed with a default sized
  for T4 (effective batch = per-device × accum ≥ 16);
- `--init-lora` flag to start from the STaR SFT adapter.

Bandit script stays untouched as baseline.

## Stages

0. **Gen-configs bug** — done (`0a90d49`), probe reports `truncated_rate`.
1. **Rollout driver** (`train/rollout.py`) + unit tests with a scripted fake
   model (no GPU).
2. **Terminal reward** (`connections_gym/episode_reward.py`) + table-driven
   tests.
3. **Multi-turn win@k probe** (`train/probe_wink.py`): N episodes per board ×
   M boards on base model, reports win@k (unbiased estimator, reuse
   `pass_at_k` from `train/probe_passk.py`), mean mistakes, truncation rate.
   Gate for every later stage. `--out-boards` writes per-board win-rate
   JSONL consumed by the difficulty ordering.
4. **STaR SFT** (`train/star_sft.py`): sample base Qwen3-1.7B episodes on
   2-cat boards via the driver, filter to won trajectories, SFT LoRA
   (TRL `SFTTrainer`, loss masked to model tokens via `env_mask`).
5. **Agentic GRPO** (`train/grpo_agentic.py`): `rollout_func` + `env_mask` +
   terminal reward, starting from the STaR adapter.
6. **Difficulty ordering** (`train/difficulty.py`):
   `load_seed_order(path) -> list[int]` sorts probed seeds easiest-first
   (win rate desc, mean mistakes asc, seed asc); `train/grpo_agentic.py
--seed-order` builds the training pool from the easiest slice.
7. **Curriculum**: 2 → 3 → 4 categories; advance on win-rate threshold,
   re-probe at each size; per-stage pool selected via difficulty ordering.

## Error handling

- Unparseable / invalid guess: one mistake, episode continues (mirrors env
  and deployment; no special reward term — it already costs via `mistakes`).
- Generation truncation (`max_new_tokens` hit): treated as unparseable;
  probe and trainer both report truncation rate so budget exhaustion is
  visible (lesson from этап 0).
- Board sampling failure: existing `ConnectionsEnv.reset` raise; probes skip
  no boards silently.

## Testing

- Unit: rollout driver with fake model (scripted outputs: win path, mistake
  path, unparseable path, truncation) asserts token/mask alignment and stats;
  episode reward table; probe estimator reuse already covered.
- Smoke (local, MPS, Qwen3-0.6B): 1 episode end-to-end via driver; probe on
  2 boards × 4 samples.
- Integration (Kaggle T4): win@k probe on 2-cat boards gates этап 4; STaR SFT
  then re-probe (expect win@k up); GRPO short run — reward EMA rises, no
  zero-grad camping, `env_mask` sanity (no loss on env tokens).

## Backlog (out of scope)

- Saturn-1.5B base-model comparison probe.
- Learned difficulty estimator trained on real NYT games (empirical
  win-rate ordering covers curriculum needs for now).
