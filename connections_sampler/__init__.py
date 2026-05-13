"""Reusable CP-SAT sampling API for Connections-style games."""

from .sampler import (
    ConfigIndex,
    GameMode,
    SampledGame,
    SampledGroup,
    SamplingMethod,
    SamplingParameters,
    SamplingResult,
    UniquenessReport,
    build_config_index,
    completed_categories,
    sample_advanced_game,
    sample_basic_game,
    sample_easy_game,
    sample_game,
    verify_unique_solution,
)

__all__ = [
    "ConfigIndex",
    "GameMode",
    "SampledGame",
    "SampledGroup",
    "SamplingMethod",
    "SamplingParameters",
    "SamplingResult",
    "UniquenessReport",
    "build_config_index",
    "completed_categories",
    "sample_advanced_game",
    "sample_basic_game",
    "sample_easy_game",
    "sample_game",
    "verify_unique_solution",
]
