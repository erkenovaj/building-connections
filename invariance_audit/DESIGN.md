# Real-Game Representation Invariance Audit

## Objective

Test whether a model gives the same correct solution to the same Connections
game when only the order or rendering of its 16 terms changes.

This experiment is independent of difficulty estimation. It does not use
expert ratings, player outcomes, mistake logs, difficulty models, or difficulty
features. The stored real games contribute only:

- the 16 visible terms;
- their canonical board positions;
- the exact four groups of four terms.

Each stored game is one latent instance. Its semantic solution is fixed. A
permutation changes the representation, not the task.

## Scientific questions

1. Does exact solution accuracy change when the 16 terms are permuted?
2. When the canonical board is solved correctly, how often does a permutation
   make the model fail or return a different partition?
3. Are matched JSON and rendered-webpage inputs equally stable?
4. Which permutation structures expose the largest positional sensitivity?
5. Does Qwen3-VL-4B-Instruct improve invariance as well as accuracy relative to
   the InternVL3-2B pilot on the same games?

The last comparison is descriptive: model family and parameter count both
change, so it is not a causal model-scale experiment.

## Corpus and separation from difficulty estimation

The source archive contains 743 real four-by-four games with stored solutions.
Seven contain native image cards; exclude them from the initial word-tile study,
leaving 736 games.

Create a dedicated audit manifest under `invariance_audit/` by reading only
`difficulty_estimation/puzzle/*-puzzle.json`. The extractor must not import or
join any other file from `difficulty_estimation/`. Each manifest record contains
only private evaluation data:

```json
{
  "game_id": "2025-01-01",
  "canonical_board": [
    "EARTHWORM", "CLOG", "SLUG", "WATERFALL",
    "GLOWSTICK", "RAINDROP", "CANAL", "FIREFLY",
    "WINDMILL", "RADIUM", "SKYDIVE", "SALAMANDER",
    "AURORA", "EEL", "TULIP", "GATECRASH"
  ],
  "solution": [
    ["EARTHWORM", "EEL", "SALAMANDER", "SLUG"],
    ["AURORA", "FIREFLY", "GLOWSTICK", "RADIUM"],
    ["CANAL", "CLOG", "TULIP", "WINDMILL"],
    ["GATECRASH", "RAINDROP", "SKYDIVE", "WATERFALL"]
  ]
}
```

The model never receives `game_id`, category titles, or `solution`. Category
titles are excluded from scoring because many valid descriptions can name the
same grouping.

Before use, assert that every game has 16 unique terms, four disjoint groups of
four, and that the group union equals the board. The stored partition is the
evaluation ground truth.

## Local game-engine adapter

The webpage already renders the structure needed by the audit:

- `js/game.js` stores a game in `currentPuzzle`;
- `createGameBoard()` renders `currentPuzzle.board` as 16 `.concept-card`
  elements inside `#game-board`;
- `js/game-rules.js` validates the board and category membership.

Add a fixture adapter that converts one manifest game to the existing engine
shape:

```json
{
  "board": ["... 16 permuted terms ..."],
  "categories": {
    "easy": {"name": "group_0", "members": ["... four terms ..."]},
    "medium": {"name": "group_1", "members": ["... four terms ..."]},
    "hard": {"name": "group_2", "members": ["... four terms ..."]},
    "harder": {"name": "group_3", "members": ["... four terms ..."]}
  },
  "numCategories": 4,
  "itemsPerCategory": 4,
  "mode": "basic"
}
```

The engine keys `easy`, `medium`, `hard`, and `harder` are compatibility IDs
only; the experiment must not treat them as difficulty labels. The fixture path
must bypass `/api/game/sample`, because resampling would change the task.

Provide a deterministic evaluation URL or test hook that loads a fixture and a
registered permutation, calls `validatePuzzleStructure`, assigns
`currentPuzzle`, and invokes the normal board renderer. It should expose no
solution, hint, or category information in the visible page.

## Input conditions

Every permutation is evaluated in two matched representations.

### JSON board

Submit a JSON object containing only the ordered 16-term array:

```json
{
  "board": ["TERM_1", "TERM_2", "...", "TERM_16"]
}
```

The surrounding instruction and output schema are identical for every game and
permutation.

### Rendered webpage board

Load the same ordered array through the fixture adapter and capture the actual
local webpage's `#game-board`. Use a fixed viewport, device scale, font set,
zoom, color scheme, and image format. Wait for fonts and layout to settle before
capture.

Crop to the board, or use a dedicated evaluation mode that hides dynamic UI.
Timers, scores, controls, hints, selections, animations, and modals must not
vary between screenshots. Generate each image by re-rendering the permuted
board; do not transform screenshot pixels.

The prompt text is fixed and the screenshot is the only visual input that
changes.

## Transformations

The primary transformation is a permutation in `S_16`, acting on board order.
Because the answer is expressed as sets of term strings, the semantic answer is
unchanged. This is an invariance test.

Use registered transformation families so positional effects can be diagnosed:

1. canonical stored order;
2. grid symmetries that move upright tiles (reflections and quarter-turn cell
   mappings, never raw image rotation);
3. row and column permutations;
4. uniform seeded random permutations;
5. fixed-budget worst-case search only after the confirmatory experiment.

The same `permutation_id` and ordered terms must be used for the JSON and
screenshot pair. Store permutations in the manifest and reuse them across
models. Never draw them inside an inference job.

Equivariance is not part of the primary condition. If a later experiment asks
the model to answer with positions or tile IDs, the correct output must be
transformed by the same permutation and evaluated as a separately registered
equivariance task.

## Prompt, decoding, and correctness

Use one fixed instruction:

```text
Partition the 16 terms into exactly four groups of four related terms.
Return JSON only in this form:
{"groups":[["...","...","...","..."], ...]}
Use every displayed term exactly once.
```

For the screenshot condition, replace “terms” with “displayed terms” if needed,
but freeze that template before the pilot. Do not provide category titles,
feedback, examples drawn from the corpus, dates, or solution hints.

Primary generation is greedy and deterministic. Pin the complete decoding
configuration and model/processor revision. Stochastic reasoning traces are
excluded: otherwise generation sampling becomes another source of variance.

Normalize case and surrounding whitespace, but do not perform fuzzy semantic
repairs. A response is correct exactly when it:

- parses into four groups of four;
- contains every board term exactly once and no other term;
- equals the stored partition after sorting members within groups and sorting
  the groups.

Group order and member order are serialization choices and never affect
correctness. Record malformed output separately from a valid but wrong
partition. At most one prespecified mechanical JSON extraction pass may be
reported alongside strict parse rate; it must not change terms or grouping.

## Experimental stages

### Stage 0: local validation

- Extract the label-free manifest and verify all structural invariants.
- Validate fixture conversion with `validatePuzzleStructure`.
- Confirm that fixture loading bypasses the sampler.
- Test semantic canonicalization across all group/member orderings.
- For several games, compare DOM term order, screenshot term order, and JSON
  term order for every registered permutation.
- Inspect screenshots for clipping and verify byte-identical rerenders under
  the same environment.

### Stage 1: InternVL3-2B pilot

Select 64 word-card games using a seed-fixed hash of `game_id`, without any
difficulty labels or gameplay statistics. Evaluate:

- canonical order plus seven registered permutations;
- JSON and screenshot input for every permutation;
- one deterministic full-partition generation per input.

This produces 1,024 inputs. Use the pilot only to validate prompts, parsing,
rendering, throughput, and effect-size estimation.

Freeze the protocol only when:

- fixture and permutation validation is 100%;
- rerendering is deterministic;
- strict parsing succeeds often enough that semantic behavior, rather than
  formatting failure, is measurable;
- no term is clipped or unreadable;
- JSON and screenshot pairs contain identical ordered terms.

### Stage 2: Qwen3-VL-4B-Instruct confirmatory run

Use the frozen protocol on all 736 word-card games with the canonical order and
15 registered permutations in each representation. This yields 23,552 inputs.

Report primary confirmatory estimates on the 672 games not used during pilot
development. Run Qwen on the 64 pilot games as a separately labeled common-set
comparison with InternVL3-2B.

If pilot power analysis requires a different number of permutations, change it
before Stage 2 and freeze a new manifest. Never add transformations after
looking at confirmatory results.

### Stage 3: secondary studies

Only after the primary analysis:

- search for worst-case permutations with a fixed evaluation budget;
- test direct full-partition output against the existing interactive game loop;
- test the seven native image-card games once their original visual assets are
  reproducibly available;
- add an explicitly positional-output equivariance condition;
- add same-family model sizes if a causal scale claim is desired.

## Metrics

For each game, model, and representation, report:

- mean exact solution accuracy across the orbit;
- all-permutations-correct rate;
- semantic partition consistency across permutations;
- correctness flip rate relative to canonical order;
- failure rate conditional on the canonical order being correct;
- canonical-to-worst permutation accuracy loss;
- strict parse failure and structurally invalid answer rates;
- paired JSON-versus-screenshot disagreement for the same permutation.

Aggregate accuracy alone is insufficient. A model can have the same mean
accuracy under two conditions while failing on different permutations.

Optionally score one fixed, lexicographically canonical solution serialization
under teacher forcing and report within-game log-likelihood range. Keep this a
secondary diagnostic because one string does not represent the full unordered
answer probability.

## Statistical analysis

The game is the independent unit; permutations are repeated measurements.

- Use game-clustered paired estimates.
- Bootstrap games with at least 2,000 replicates for final 95% confidence
  intervals.
- Compare matched JSON and screenshot outcomes for the same game/permutation.
- Use a logistic mixed-effects model with a game random intercept for model,
  representation, permutation-family, and prespecified interactions.
- Report effect sizes and confidence intervals, not only p-values.
- Keep pilot and confirmatory results separate.

No difficulty variable belongs in these models.

## Reproducibility and outputs

Record one row per game, model, representation, and permutation:

```text
game_id, model_id, model_revision, representation,
permutation_id, ordered_terms_hash, prompt_hash, screenshot_hash,
raw_output, parse_status, canonical_partition, exact_correct,
generation_config, runtime_metadata
```

Persist the extracted game manifest, permutation manifest, screenshots, raw
responses, parsed responses, model/processor revisions, chat templates,
dependency lock, GPU/CUDA metadata, and Git commit. Jobs must write
incrementally and resume without regenerating permutations or screenshots.

The final result should answer one narrow question: **when the latent game and
its unique solution are fixed, how much does a model's solution depend on the
ordering and representation of the same 16 terms?**
