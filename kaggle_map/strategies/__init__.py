"""Dynamic strategy registry for automatic discovery of prediction strategies."""

import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from loguru import logger

from .base import Strategy


def _should_skip_file(filename: str) -> bool:
    return filename.startswith("_") or filename in ("base.py", "__init__.py")


def _import_strategy_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(f"kaggle_map.strategies.{module_name}")
    except ImportError as e:
        logger.warning(f"Failed to import strategy module {module_name}: {e}")
        return None


def _find_strategy_class(module: ModuleType) -> type[Strategy] | None:
    for _name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Strategy) and obj is not Strategy:
            return obj
    return None


def _discover_strategies() -> dict[str, type[Strategy]]:
    """Scan strategies directory and import all Strategy classes.

    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    strategies = {}
    strategies_dir = Path(__file__).parent

    logger.debug(f"Scanning for strategies in {strategies_dir}")

    for py_file in strategies_dir.glob("*.py"):
        if _should_skip_file(py_file.name):
            continue

        module_name = py_file.stem
        logger.debug(f"Checking module: {module_name}")

        module = _import_strategy_module(module_name)
        if not module:
            continue

        strategy_cls = _find_strategy_class(module)
        if strategy_cls is not None:
            strategies[module_name] = strategy_cls
            logger.debug(f"Found strategy '{module_name}': {strategy_cls.__name__}")

    logger.info(f"Discovered {len(strategies)} strategies: {list(strategies.keys())}")
    return strategies


def get_strategy(name: str) -> type[Strategy]:
    """Get strategy class by name.

    Args:
        name: Strategy name (e.g., 'baseline', 'probabilistic')

    Returns:
        Strategy class

    Raises:
        ValueError: If strategy name is not found
    """
    strategies = get_all_strategies()
    if name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return strategies[name]


def list_strategies() -> list[str]:
    """List all available strategy names.

    Returns:
        List of strategy names
    """
    return list(get_all_strategies().keys())


@lru_cache(maxsize=1)
def get_all_strategies() -> dict[str, type[Strategy]]:
    """Get all discovered strategies.

    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    return _discover_strategies()


def refresh_strategies() -> None:
    """Force refresh of strategy cache (useful for testing)."""
    get_all_strategies.cache_clear()
