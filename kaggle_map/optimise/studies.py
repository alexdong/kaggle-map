import json
from datetime import UTC, datetime
from pathlib import Path

import optuna
from loguru import logger

from kaggle_map.utils.file_utils import create_timestamped_filename, ensure_directory
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

# Single consolidated database location
STORAGE_URL = "sqlite:///kaggle_map/optimise/optuna.db"


def create(study_name: str, direction: str = "maximize", storage_url: str = STORAGE_URL) -> optuna.Study:
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


def get_by_name(study_name: str, storage_url: str = STORAGE_URL) -> optuna.Study:
    return optuna.load_study(study_name=study_name, storage=storage_url)


def save(study: optuna.Study) -> Path:
    filepath = Path("models/studies") / create_timestamped_filename(
        f"{study.study_name}",
        extension=".json",
    )
    ensure_directory(filepath)

    with filepath.open("w") as f:
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
    return filepath


def ls(storage_url: str = STORAGE_URL) -> list[str]:
    storage = optuna.storages.RDBStorage(url=storage_url)
    study_summaries = storage.get_all_studies()
    return sorted([s.study_name for s in study_summaries])
