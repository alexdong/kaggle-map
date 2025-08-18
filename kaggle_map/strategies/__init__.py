"""Dynamic strategy registry for automatic discovery of prediction strategies."""

import importlib
import inspect
from pathlib import Path
from typing import Type

from loguru import logger

from .base import Strategy


def _discover_strategies() -> dict[str, type[Strategy]]:
    """Scan strategies directory and import all Strategy classes.
    
    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    strategies = {}
    strategies_dir = Path(__file__).parent
    
    logger.debug(f"Scanning for strategies in {strategies_dir}")
    
    for py_file in strategies_dir.glob("*.py"):
        # Skip internal files and base class
        if py_file.name.startswith("_") or py_file.name in ("base.py", "__init__.py"):
            continue
            
        module_name = py_file.stem
        logger.debug(f"Checking module: {module_name}")
        
        try:
            module = importlib.import_module(f"kaggle_map.strategies.{module_name}")
            
            # Find Strategy classes in module
            for _name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, Strategy) and
                    obj is not Strategy):
                    # Use module name as strategy key (e.g., "baseline", "probabilistic")
                    strategies[module_name] = obj
                    logger.debug(f"Found strategy '{module_name}': {obj.__name__}")
                    break  # Only take first Strategy class per module
                    
        except ImportError as e:
            logger.warning(f"Failed to import strategy module {module_name}: {e}")
            continue
            
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


def get_all_strategies() -> dict[str, type[Strategy]]:
    """Get all discovered strategies.
    
    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    # Cache strategies on first call
    if not hasattr(get_all_strategies, "_cached_strategies"):
        get_all_strategies._cached_strategies = _discover_strategies()
    return get_all_strategies._cached_strategies


def refresh_strategies() -> None:
    """Force refresh of strategy cache (useful for testing)."""
    if hasattr(get_all_strategies, "_cached_strategies"):
        delattr(get_all_strategies, "_cached_strategies")
