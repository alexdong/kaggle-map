import json
from datetime import UTC, datetime
from pathlib import Path

import optuna
import torch
from loguru import logger

# Single consolidated database location
STORAGE_URL = "sqlite:///kaggle_map/optimise/optuna.db"


def create_study(study_name: str, direction: str = "maximize", storage_url: str = STORAGE_URL) -> optuna.Study:
    logger.info(f"Creating/loading study: {study_name}")

    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction=direction,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=1,
        ),
    )


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def track_gpu_memory(trial: optuna.Trial) -> None:
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        trial.set_user_attr("peak_gpu_memory_gb", peak_memory)
        logger.info(f"Trial {trial.number} peak GPU memory: {peak_memory:.2f}GB")


def handle_oom_error(trial: optuna.Trial, error: Exception) -> float:
    if torch.cuda.is_available():
        logger.error(f"Trial {trial.number} OOM: {error}")
        logger.error(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        logger.error(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        torch.cuda.empty_cache()

    trial.set_user_attr("oom_error", value=True)
    trial.set_user_attr("oom_details", value=str(error))

    return 0.0


def save_best_config(study: optuna.Study, strategy_name: str) -> Path:
    best_config_path = Path(f"models/{strategy_name}_best_config.json")
    best_config_path.parent.mkdir(parents=True, exist_ok=True)

    with best_config_path.open("w") as f:
        json.dump(
            {
                "study_name": study.study_name,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )

    logger.info(f"Best configuration saved to {best_config_path}")
    return best_config_path


def list_all_studies(storage_url: str = STORAGE_URL) -> list[str]:
    try:
        storage = optuna.storages.RDBStorage(url=storage_url)
        study_summaries = storage.get_all_studies()
        return sorted([s.study_name for s in study_summaries])
    except Exception as e:
        logger.error(f"Failed to list studies: {e}")
        return []
