"""Unified random seed configuration for the Kaggle MAP project."""

import os
import random
from typing import Final

import torch
from loguru import logger

from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

DEFAULT_RANDOM_SEED: Final[int] = 42
_ENV_VAR: Final[str] = "KAGGLE_MAP_RANDOM_SEED"
_MAX_SEED: Final[int] = 2**32 - 1

_ACTIVE_SEED: dict[str, int] = {"value": -1}


def _parse_seed(value: str) -> int:
    assert value, "Seed string cannot be empty"
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - should fail during integration tests
        msg = f"Invalid seed value: {value!r}"
        raise AssertionError(msg) from exc
    assert 0 <= parsed <= _MAX_SEED, f"Seed must be between 0 and {_MAX_SEED}, got {parsed}"
    return parsed


def _determine_seed(override: int | None) -> int:
    if override is not None:
        return _validate_seed(override)

    env_value = os.environ.get(_ENV_VAR)
    if env_value is not None:
        return _parse_seed(env_value)

    return DEFAULT_RANDOM_SEED


def _validate_seed(seed: int) -> int:
    assert isinstance(seed, int), f"Seed must be int, got {type(seed).__name__}"
    assert 0 <= seed <= _MAX_SEED, f"Seed must be between 0 and {_MAX_SEED}, got {seed}"
    return seed


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_random_seed(*, override: int | None = None) -> int:
    """Resolve and apply the global random seed."""
    active_seed = _ACTIVE_SEED["value"]
    if active_seed != -1:
        if override is None:
            return active_seed
        seed = _validate_seed(override)
        if seed == active_seed:
            return active_seed
        logger.debug("Reconfiguring global random seed: {} -> {}", active_seed, seed)
        _apply_seed(seed)
        _ACTIVE_SEED["value"] = seed
        return seed

    seed = _determine_seed(override)
    logger.debug("Configuring global random seed: {}", seed)
    _apply_seed(seed)
    _ACTIVE_SEED["value"] = seed
    return seed


def get_active_seed() -> int:
    """Return the currently active seed, configuring it on first use."""
    if _ACTIVE_SEED["value"] == -1:
        return configure_random_seed()
    return _ACTIVE_SEED["value"]


__all__ = [
    "DEFAULT_RANDOM_SEED",
    "configure_random_seed",
    "get_active_seed",
]
