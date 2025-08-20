"""Dynamic strategy registry for automatic discovery of prediction strategies."""

import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from loguru import logger

from .base import Strategy


def _should_skip_file(filename: str) -> bool:
    should_skip = filename.startswith("_") or filename in ("base.py", "__init__.py")
    if should_skip:
        logger.debug(
            f"Skipping file '{filename}': {'private/internal file' if filename.startswith('_') else 'base/init file'}"
        )
    return should_skip


def _import_strategy_module(module_name: str) -> ModuleType | None:
    full_module_name = f"kaggle_map.strategies.{module_name}"
    logger.debug(f"Attempting to import module: {full_module_name}")

    try:
        module = importlib.import_module(full_module_name)
        logger.debug(f"Successfully imported module: {full_module_name}")
        return module
    except ImportError as e:
        logger.warning(f"Failed to import strategy module '{module_name}': {e}")
        logger.debug(
            f"Import error details for '{full_module_name}': {type(e).__name__}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error importing module '{module_name}': {type(e).__name__}: {e}"
        )
        return None


def _log_class_analysis(name: str, obj: type, _module_name: str) -> str:
    """Log analysis of a class and return skip reason if not a valid strategy."""
    if obj is Strategy:
        reason = "is the base Strategy class"
    elif not issubclass(obj, Strategy):
        reason = "does not inherit from Strategy"
    else:
        # This is a valid Strategy subclass
        logger.debug(
            f"Class '{name}' qualifies as Strategy (inherits from Strategy base class)"
        )
        return ""

    logger.debug(f"Skipping '{name}': {reason}")
    return reason


def _find_strategy_class(module: ModuleType) -> type[Strategy] | None:
    module_name = module.__name__
    logger.debug(f"Scanning module '{module_name}' for Strategy classes")

    all_classes = []
    strategy_classes = []

    for name, obj in inspect.getmembers(module, inspect.isclass):
        all_classes.append(name)
        logger.debug(f"Found class '{name}' in module '{module_name}'")

        skip_reason = _log_class_analysis(name, obj, module_name)
        if not skip_reason:  # Valid strategy class found
            strategy_classes.append(name)
            logger.info(f"Selected strategy class '{name}' from module '{module_name}'")
            return obj

    logger.debug(
        f"Module '{module_name}' contains {len(all_classes)} classes: {all_classes}"
    )

    if not strategy_classes:
        logger.warning(
            f"No Strategy classes found in module '{module_name}'. Available classes: {all_classes}"
        )

    return None


def _log_discovery_summary(
    processed_files: int,
    skipped_files: int,
    failed_imports: int,
    successful_strategies: int,
) -> None:
    """Log summary of strategy discovery process."""
    logger.info(
        f"Strategy discovery complete. "
        f"Files processed: {processed_files}, skipped: {skipped_files}, "
        f"import failures: {failed_imports}, successful strategies: {successful_strategies}"
    )


def _log_final_strategies(strategies: dict[str, type[Strategy]]) -> None:
    """Log final list of discovered strategies."""
    if strategies:
        strategy_details = [
            f"'{name}' ({cls.__name__})" for name, cls in strategies.items()
        ]
        logger.info(
            f"Registered {len(strategies)} strategies: {', '.join(strategy_details)}"
        )
    else:
        logger.warning(
            "No strategies were discovered! This may indicate a configuration problem."
        )


def _discover_strategies() -> dict[str, type[Strategy]]:
    """Scan strategies directory and import all Strategy classes.

    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    strategies = {}
    strategies_dir = Path(__file__).parent

    logger.info(f"Starting strategy discovery in directory: {strategies_dir}")

    # Find all Python files and packages in the strategies directory
    py_files = list(strategies_dir.glob("*.py"))
    packages = [
        d
        for d in strategies_dir.glob("*")
        if d.is_dir() and (d / "__init__.py").exists()
    ]

    logger.debug(f"Found {len(py_files)} Python files: {[f.name for f in py_files]}")
    logger.debug(f"Found {len(packages)} packages: {[p.name for p in packages]}")

    processed_files = 0
    skipped_files = 0
    failed_imports = 0
    successful_strategies = 0

    # Process both Python files and packages
    all_modules = [(f, f.stem, str(f)) for f in py_files] + [
        (p, p.name, str(p)) for p in packages
    ]

    for module_path, module_name, file_path in all_modules:
        logger.debug(f"Processing module: {module_name} (path: {file_path})")

        if _should_skip_file(module_path.name):
            skipped_files += 1
            continue

        processed_files += 1
        logger.debug(f"Processing potential strategy module: {module_name}")

        module = _import_strategy_module(module_name)
        if not module:
            failed_imports += 1
            logger.debug(
                f"Failed to import module '{module_name}', moving to next file"
            )
            continue

        strategy_cls = _find_strategy_class(module)
        if strategy_cls is not None:
            strategies[module_name] = strategy_cls
            successful_strategies += 1
            logger.info(
                f"Registered strategy '{module_name}': {strategy_cls.__name__} from {file_path}"
            )
        else:
            logger.debug(f"No valid Strategy class found in module '{module_name}'")

    _log_discovery_summary(
        processed_files, skipped_files, failed_imports, successful_strategies
    )
    _log_final_strategies(strategies)

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
    logger.debug(f"Looking up strategy: '{name}'")

    strategies = get_all_strategies()

    if name in strategies:
        strategy_cls = strategies[name]
        logger.debug(f"Strategy lookup successful: '{name}' -> {strategy_cls.__name__}")
        return strategy_cls

    # Strategy not found - log details for debugging
    available_names = list(strategies.keys())
    logger.warning(
        f"Strategy lookup failed for '{name}'. Available strategies: {available_names}"
    )

    # Provide suggestions for common typos or similar names
    suggestions = [
        available
        for available in available_names
        if name.lower() in available.lower() or available.lower() in name.lower()
    ]

    if suggestions:
        logger.info(f"Did you mean one of these similar strategies? {suggestions}")

    available = ", ".join(available_names)
    msg = f"Unknown strategy '{name}'. Available: {available}"
    raise ValueError(msg)


def list_strategies() -> list[str]:
    """List all available strategy names.

    Returns:
        List of strategy names
    """
    logger.debug("Requesting list of all available strategies")
    strategies = get_all_strategies()
    strategy_names = list(strategies.keys())
    logger.debug(f"Returning {len(strategy_names)} strategy names: {strategy_names}")
    return strategy_names


@lru_cache(maxsize=1)
def get_all_strategies() -> dict[str, type[Strategy]]:
    """Get all discovered strategies.

    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    logger.debug("Retrieving all strategies (may trigger discovery if not cached)")
    strategies = _discover_strategies()
    logger.debug(f"Returning {len(strategies)} strategies from cache/discovery")
    return strategies


def refresh_strategies() -> None:
    """Force refresh of strategy cache (useful for testing)."""
    logger.info("Clearing strategy cache - next access will trigger fresh discovery")
    get_all_strategies.cache_clear()
    logger.debug("Strategy cache cleared successfully")
