"""Public types and CP-SAT sampling logic for Connections-style games."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from itertools import combinations
import random
import secrets
import time
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from ortools.sat.python import cp_model


class GameMode(str, Enum):
    EASY = "easy"
    BASIC = "basic"
    ADVANCED = "advanced"

    @classmethod
    def parse(cls, value: str) -> "GameMode":
        normalized = str(value or "").strip().lower()
        if normalized == "normal":
            normalized = cls.BASIC.value
        return cls(normalized)


class SamplingMethod(str, Enum):
    RANDOMIZED_FEASIBLE = "randomized_feasible"


@dataclass(frozen=True)
class SamplingParameters:
    """Controls randomized CP-SAT search without changing mode semantics."""

    seed: Optional[object] = None
    max_attempts: int = 40
    per_solve_timeout_seconds: float = 1.5
    total_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class SampledGroup:
    name: str
    members: Tuple[str, ...]
    key: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "members": list(self.members),
            "difficulty": self.key,
        }


@dataclass(frozen=True)
class SampledGame:
    board: Tuple[str, ...]
    groups: Tuple[SampledGroup, ...]
    mode: GameMode
    items_per_category: int
    sampling_method: SamplingMethod = SamplingMethod.RANDOMIZED_FEASIBLE

    def to_dict(self) -> Dict[str, object]:
        return {
            "board": list(self.board),
            "categories": {group.key: group.to_dict() for group in self.groups},
            "groups": [group.to_dict() for group in self.groups],
            "numCategories": len(self.groups),
            "itemsPerCategory": self.items_per_category,
            "mode": self.mode.value,
            "samplingMethod": self.sampling_method.value,
        }


@dataclass(frozen=True)
class SamplingResult:
    status: str
    game: Optional[SampledGame] = None
    reason: Optional[str] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.game is not None and self.status == "success"

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "status": self.status,
            "game": self.game.to_dict() if self.game else None,
            "diagnostics": dict(self.diagnostics),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class UniquenessReport:
    is_unique: bool
    solution_count: int
    intended_solution_found: bool
    status: str


@dataclass(frozen=True)
class ConfigIndex:
    items: Tuple[str, ...]
    item_to_categories: Mapping[str, FrozenSet[str]]
    category_to_members: Mapping[str, Tuple[str, ...]]


def build_config_index(config: Sequence[Mapping[str, object]]) -> ConfigIndex:
    """Normalize item/tag or category/member config entries."""
    item_order: List[str] = []
    item_categories: Dict[str, set[str]] = {}
    category_members: Dict[str, List[str]] = {}

    def add_item(item_name: object) -> str | None:
        name = str(item_name).strip() if item_name is not None else ""
        if not name:
            return None
        if name not in item_categories:
            item_categories[name] = set()
            item_order.append(name)
        return name

    for entry in config:
        if not isinstance(entry, Mapping):
            continue

        raw_members = entry.get("members")
        if isinstance(raw_members, (list, tuple)):
            category = str(entry.get("name") or "").strip()
            if not category:
                continue
            members = category_members.setdefault(category, [])
            for raw_member in raw_members:
                item = add_item(raw_member)
                if item is None:
                    continue
                item_categories[item].add(category)
                if item not in members:
                    members.append(item)
            continue

        item = add_item(entry.get("name"))
        if item is None:
            continue
        raw_tags = entry.get("tags")
        tags = raw_tags if isinstance(raw_tags, (list, tuple)) else ()
        for raw_tag in tags:
            category = str(raw_tag).strip() if raw_tag is not None else ""
            if not category:
                continue
            item_categories[item].add(category)
            members = category_members.setdefault(category, [])
            if item not in members:
                members.append(item)

    return ConfigIndex(
        items=tuple(item_order),
        item_to_categories={
            item: frozenset(categories)
            for item, categories in item_categories.items()
        },
        category_to_members={
            category: tuple(members)
            for category, members in category_members.items()
        },
    )


_GROUP_KEYS = ("easy", "medium", "hard", "harder")
GroupSignature = Tuple[str, Tuple[str, ...]]
SolutionSignature = Tuple[GroupSignature, ...]


def _canonical_solution(groups: Iterable[GroupSignature]) -> SolutionSignature:
    return tuple(sorted((name, tuple(sorted(members))) for name, members in groups))


class _SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        variables: Sequence[cp_model.IntVar],
        signatures: Sequence[GroupSignature],
        limit: int = 2,
    ) -> None:
        super().__init__()
        self._variables = variables
        self._signatures = signatures
        self._limit = limit
        self.solutions: List[SolutionSignature] = []

    def on_solution_callback(self) -> None:
        chosen = [
            signature
            for variable, signature in zip(self._variables, self._signatures)
            if self.value(variable)
        ]
        self.solutions.append(_canonical_solution(chosen))
        if len(self.solutions) >= self._limit:
            self.stop_search()


def verify_unique_solution(
    index: ConfigIndex,
    *,
    board: Sequence[str],
    intended_groups: Sequence[Tuple[str, Sequence[str]]],
    num_categories: int,
    items_per_category: int,
    mode: GameMode,
    timeout_seconds: float = 2.0,
) -> UniquenessReport:
    """Enumerate at most two CP-SAT solutions for a fixed board."""
    board_items = tuple(dict.fromkeys(str(item) for item in board))
    board_set = set(board_items)
    if len(board_items) != len(board):
        return UniquenessReport(False, 0, False, "duplicate_board_items")

    signatures: List[GroupSignature] = []
    for category, members in index.category_to_members.items():
        present = sorted(board_set.intersection(members))
        if len(present) < items_per_category:
            continue
        signatures.extend(
            (category, tuple(group))
            for group in combinations(present, items_per_category)
        )

    intended = _canonical_solution(
        (name, tuple(str(member) for member in members))
        for name, members in intended_groups
    )
    if not signatures:
        return UniquenessReport(False, 0, False, "no_candidate_groups")

    model = cp_model.CpModel()
    variables = [
        model.new_bool_var(f"group_{idx}")
        for idx in range(len(signatures))
    ]

    for category in index.category_to_members:
        model.add(
            sum(
                variable
                for variable, signature in zip(variables, signatures)
                if signature[0] == category
            )
            <= 1
        )

    for item in board_items:
        covering = [
            variable
            for variable, (_, members) in zip(variables, signatures)
            if item in members
        ]
        if mode is GameMode.ADVANCED:
            model.add(sum(covering) <= 1)
        else:
            model.add(sum(covering) == 1)

    model.add(sum(variables) == num_categories)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = max(0.05, timeout_seconds)
    solver.parameters.num_search_workers = 1
    collector = _SolutionCollector(variables, signatures)
    status = solver.solve(model, collector)
    intended_found = intended in collector.solutions
    return UniquenessReport(
        is_unique=(
            status == cp_model.OPTIMAL
            and len(collector.solutions) == 1
            and intended_found
        ),
        solution_count=len(collector.solutions),
        intended_solution_found=intended_found,
        status=solver.status_name(status).lower(),
    )


def completed_categories(
    index: ConfigIndex,
    board: Sequence[str],
    items_per_category: int,
) -> Dict[str, Tuple[str, ...]]:
    """Return every category with enough board terms to form a group."""
    board_set = set(board)
    return {
        category: tuple(sorted(board_set.intersection(members)))
        for category, members in index.category_to_members.items()
        if len(board_set.intersection(members)) >= items_per_category
    }


@dataclass(frozen=True)
class _Candidate:
    board: Tuple[str, ...]
    groups: Tuple[Tuple[str, Tuple[str, ...]], ...]
    selected_categories: frozenset[str]

    @property
    def signature(self) -> Tuple[frozenset[str], frozenset[str]]:
        return self.selected_categories, frozenset(self.board)


def _seed_to_int(seed: object | None) -> int:
    if seed is None:
        return secrets.randbits(31)
    if isinstance(seed, int):
        return seed & 0x7FFFFFFF
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _group_key(index: int) -> str:
    return _GROUP_KEYS[index] if index < len(_GROUP_KEYS) else f"group_{index + 1}"


def _add_exclusion(
    model: cp_model.CpModel,
    selected_vars: Mapping[str, cp_model.IntVar],
    board_vars: Mapping[str, cp_model.IntVar],
    signature: Tuple[frozenset[str], frozenset[str]],
) -> None:
    selected_categories, board_items = signature
    matching_literals = [
        variable if category in selected_categories else variable.negated()
        for category, variable in selected_vars.items()
    ]
    matching_literals.extend(
        variable if item in board_items else variable.negated()
        for item, variable in board_vars.items()
    )
    model.add(sum(matching_literals) <= len(matching_literals) - 1)


def _solve_candidate(
    index: ConfigIndex,
    *,
    mode: GameMode,
    num_categories: int,
    items_per_category: int,
    board_size: int,
    rng: random.Random,
    excluded: Sequence[Tuple[frozenset[str], frozenset[str]]],
    timeout_seconds: float,
) -> Tuple[Optional[_Candidate], str]:
    eligible_categories = {
        category: members
        for category, members in index.category_to_members.items()
        if len(members) >= items_per_category
    }
    if len(eligible_categories) < num_categories:
        return None, "infeasible"

    model = cp_model.CpModel()
    selected = {
        category: model.new_bool_var(f"selected_{idx}")
        for idx, category in enumerate(eligible_categories)
    }
    board = {
        item: model.new_bool_var(f"board_{idx}")
        for idx, item in enumerate(index.items)
    }
    model.add(sum(selected.values()) == num_categories)
    model.add(sum(board.values()) == board_size)

    group_membership: Dict[Tuple[str, str], cp_model.IntVar] = {}
    item_group_vars: Dict[str, List[cp_model.IntVar]] = {
        item: [] for item in index.items
    }

    for category, members in eligible_categories.items():
        category_count = sum(board[item] for item in members)
        model.add(category_count == items_per_category).only_enforce_if(
            selected[category]
        )
        if mode is GameMode.ADVANCED:
            model.add(category_count <= items_per_category - 1).only_enforce_if(
                selected[category].negated()
            )

        category_group_vars: List[cp_model.IntVar] = []
        for item in members:
            variable = model.new_bool_var(
                f"in_group_{len(group_membership)}"
            )
            group_membership[(category, item)] = variable
            category_group_vars.append(variable)
            item_group_vars[item].append(variable)
            model.add(variable <= selected[category])
            model.add(variable <= board[item])
            model.add(variable >= selected[category] + board[item] - 1)
        model.add(
            sum(category_group_vars)
            == items_per_category * selected[category]
        )

    for variables in item_group_vars.values():
        if variables:
            model.add(sum(variables) <= 1)

    if mode is GameMode.EASY:
        categories = list(eligible_categories)
        member_sets = {
            category: set(eligible_categories[category])
            for category in categories
        }
        for left_index, left in enumerate(categories):
            for right in categories[left_index + 1 :]:
                if member_sets[left].intersection(member_sets[right]):
                    model.add(selected[left] + selected[right] <= 1)

    for signature in excluded:
        _add_exclusion(model, selected, board, signature)

    objective_terms = [
        rng.randint(-1000, 1000) * variable
        for variable in (*selected.values(), *board.values())
    ]
    model.maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, timeout_seconds)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = rng.randrange(1, 2**31)
    solver.parameters.randomize_search = True
    solver.parameters.search_branching = cp_model.RANDOMIZED_SEARCH
    status = solver.solve(model)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return None, solver.status_name(status).lower()

    chosen_categories = [
        category
        for category, variable in selected.items()
        if solver.value(variable)
    ]
    rng.shuffle(chosen_categories)
    board_items = [
        item for item, variable in board.items() if solver.value(variable)
    ]
    groups = tuple(
        (
            category,
            tuple(
                item
                for item in eligible_categories[category]
                if solver.value(board[item])
            ),
        )
        for category in chosen_categories
    )
    return (
        _Candidate(
            board=tuple(board_items),
            groups=groups,
            selected_categories=frozenset(chosen_categories),
        ),
        solver.status_name(status).lower(),
    )


def _sample(
    config: Sequence[Mapping[str, object]],
    *,
    mode: GameMode,
    num_categories: int,
    items_per_category: int,
    board_size: int,
    parameters: SamplingParameters,
) -> SamplingResult:
    if num_categories < 1 or items_per_category < 1:
        return SamplingResult(
            status="invalid_config",
            reason="num_categories and items_per_category must be positive",
        )
    index = build_config_index(config)
    if len(index.items) < board_size:
        return SamplingResult(
            status="infeasible",
            reason=f"config has {len(index.items)} items but board needs {board_size}",
        )

    master_seed = _seed_to_int(parameters.seed)
    rng = random.Random(master_seed)
    excluded: List[Tuple[frozenset[str], frozenset[str]]] = []
    started = time.monotonic()
    last_solver_status = "not_started"
    ambiguous_candidates = 0

    for attempt in range(1, max(1, parameters.max_attempts) + 1):
        elapsed = time.monotonic() - started
        remaining = parameters.total_timeout_seconds - elapsed
        if remaining <= 0:
            break
        timeout = min(parameters.per_solve_timeout_seconds, remaining)
        candidate, last_solver_status = _solve_candidate(
            index,
            mode=mode,
            num_categories=num_categories,
            items_per_category=items_per_category,
            board_size=board_size,
            rng=rng,
            excluded=excluded,
            timeout_seconds=timeout,
        )
        if candidate is None:
            if not excluded and last_solver_status == "infeasible":
                return SamplingResult(
                    status="infeasible",
                    reason="no game satisfies the mode constraints",
                    diagnostics={
                        "attempts": attempt,
                        "seed": master_seed,
                        "solverStatus": last_solver_status,
                    },
                )
            break

        elapsed = time.monotonic() - started
        remaining = parameters.total_timeout_seconds - elapsed
        if remaining <= 0:
            break
        report = verify_unique_solution(
            index,
            board=candidate.board,
            intended_groups=candidate.groups,
            num_categories=num_categories,
            items_per_category=items_per_category,
            mode=mode,
            timeout_seconds=min(parameters.per_solve_timeout_seconds, remaining),
        )
        if report.is_unique:
            shuffled_board = list(candidate.board)
            rng.shuffle(shuffled_board)
            groups = tuple(
                SampledGroup(
                    name=category,
                    members=members,
                    key=_group_key(index),
                )
                for index, (category, members) in enumerate(candidate.groups)
            )
            return SamplingResult(
                status="success",
                game=SampledGame(
                    board=tuple(shuffled_board),
                    groups=groups,
                    mode=mode,
                    items_per_category=items_per_category,
                ),
                diagnostics={
                    "attempts": attempt,
                    "ambiguousCandidatesRejected": ambiguous_candidates,
                    "seed": master_seed,
                    "solverStatus": last_solver_status,
                    "uniquenessStatus": report.status,
                },
            )

        ambiguous_candidates += 1
        excluded.append(candidate.signature)

    timed_out = (
        time.monotonic() - started >= parameters.total_timeout_seconds
        or last_solver_status == "unknown"
    )
    return SamplingResult(
        status="timeout" if timed_out else "infeasible",
        reason="could not find a uniquely solvable game within the search budget",
        diagnostics={
            "attempts": len(excluded),
            "ambiguousCandidatesRejected": ambiguous_candidates,
            "seed": master_seed,
            "solverStatus": last_solver_status,
        },
    )


def sample_game(
    config: Sequence[Mapping[str, object]],
    *,
    mode: GameMode | str = GameMode.BASIC,
    parameters: Optional[SamplingParameters] = None,
) -> SamplingResult:
    """Sample one of the three webpage modes."""
    parsed_mode = mode if isinstance(mode, GameMode) else GameMode.parse(mode)
    params = parameters or SamplingParameters()
    if parsed_mode is GameMode.ADVANCED:
        return _sample(
            config,
            mode=parsed_mode,
            num_categories=3,
            items_per_category=4,
            board_size=16,
            parameters=params,
        )
    return _sample(
        config,
        mode=parsed_mode,
        num_categories=4,
        items_per_category=4,
        board_size=16,
        parameters=params,
    )


def sample_basic_game(
    config: Sequence[Mapping[str, object]],
    *,
    num_categories: int = 4,
    items_per_category: int = 4,
    parameters: Optional[SamplingParameters] = None,
) -> SamplingResult:
    """Sample a generic uniquely solvable Basic game with P groups of Q items."""
    return _sample(
        config,
        mode=GameMode.BASIC,
        num_categories=num_categories,
        items_per_category=items_per_category,
        board_size=num_categories * items_per_category,
        parameters=parameters or SamplingParameters(),
    )


def sample_easy_game(
    config: Sequence[Mapping[str, object]],
    *,
    parameters: Optional[SamplingParameters] = None,
) -> SamplingResult:
    return sample_game(config, mode=GameMode.EASY, parameters=parameters)


def sample_advanced_game(
    config: Sequence[Mapping[str, object]],
    *,
    parameters: Optional[SamplingParameters] = None,
) -> SamplingResult:
    return sample_game(config, mode=GameMode.ADVANCED, parameters=parameters)
