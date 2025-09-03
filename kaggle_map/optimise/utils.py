import json
from datetime import datetime
from pathlib import Path

import numpy as np
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

    trial.set_user_attr("oom_error", True)
    trial.set_user_attr("oom_details", str(error))

    return 0.0


def cleanup_after_trial() -> None:
    # Close wandb if it's running
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    # Clear GPU memory
    clear_gpu_memory()


def build_wandb_run_name(trial: optuna.Trial, hyperparams: dict) -> str:
    trial_num = trial.number
    trial_info = f"trial_{trial_num}"
    key_params = []

    # Include key parameters in run name for easy identification
    if "embedding_model" in hyperparams:
        key_params.append(f"emb_{hyperparams['embedding_model']}")
    if "learning_rate" in hyperparams:
        key_params.append(f"lr_{hyperparams['learning_rate']:.1e}")
    if "batch_size" in hyperparams:
        key_params.append(f"bs_{hyperparams['batch_size']}")
    if "dropout" in hyperparams:
        key_params.append(f"do_{hyperparams['dropout']:.2f}")
    if "architecture_size" in hyperparams:
        key_params.append(f"arch_{hyperparams['architecture_size']}")
    if "num_layers" in hyperparams:
        key_params.append(f"layers_{hyperparams['num_layers']}")

    return f"hypersearch_{trial_info}_{'_'.join(key_params)}"


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
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    logger.info(f"Best configuration saved to {best_config_path}")
    return best_config_path


def get_trial_statistics(study: optuna.Study) -> dict:
    completed_trials = [t for t in study.trials if t.value is not None]

    if not completed_trials:
        return {}

    values = [t.value for t in completed_trials]
    values_array = np.array(values, dtype=np.float64)

    return {
        "n_completed": len(completed_trials),
        "n_total": len(study.trials),
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": min(values),
        "max": max(values),
    }


def generate_study_summary(study: optuna.Study, output_dir: Path = Path("logs")) -> Path | None:
    if len(study.trials) == 0:
        logger.warning(f"Study {study.study_name} has no trials, skipping summary")
        return None

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Generate summary file
    summary_path = output_dir / f"{study.study_name}.md"

    with summary_path.open("w") as f:
        f.write(f"# Optimization Study: {study.study_name}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overview
        f.write("## Study Overview\n\n")
        f.write(f"- **Total Trials**: {len(study.trials)}\n")
        f.write(f"- **Best Value**: {study.best_value:.4f}\n")
        f.write(f"- **Best Trial**: #{study.best_trial.number}\n\n")

        # Best parameters
        f.write("## Best Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for param, value in sorted(study.best_params.items()):
            if isinstance(value, float):
                if value < 0.01:
                    f.write(f"| {param} | {value:.2e} |\n")
                else:
                    f.write(f"| {param} | {value:.4f} |\n")
            else:
                f.write(f"| {param} | {value} |\n")

        # Parameter importance
        f.write("\n## Parameter Importance\n\n")
        try:
            importance = optuna.importance.get_param_importances(study)
            f.write("| Parameter | Importance |\n")
            f.write("|-----------|------------|\n")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {param} | {imp:.3f} |\n")
        except Exception:
            f.write("*Parameter importance analysis unavailable*\n")

        # Trial statistics
        stats = get_trial_statistics(study)
        if stats:
            f.write("\n## Trial Statistics\n\n")
            f.write(f"- **Completed**: {stats['n_completed']}/{stats['n_total']}\n")
            f.write(f"- **Mean**: {stats['mean']:.4f}\n")
            f.write(f"- **Std Dev**: {stats['std']:.4f}\n")
            f.write(f"- **Min**: {stats['min']:.4f}\n")
            f.write(f"- **Max**: {stats['max']:.4f}\n")

        # Top trials
        completed_trials = [t for t in study.trials if t.value is not None]
        if completed_trials:
            f.write("\n## Top 5 Trials\n\n")
            top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]

            for _i, trial in enumerate(top_trials, 1):
                f.write(f"### Trial #{trial.number} (Value: {trial.value:.4f})\n\n")
                f.write("```json\n")
                f.write(json.dumps(trial.params, indent=2))
                f.write("\n```\n\n")

        # OOM analysis if applicable
        oom_trials = [t for t in study.trials if t.user_attrs.get("oom_error", False)]
        if oom_trials:
            f.write("\n## Memory Issues\n\n")
            f.write(f"- **OOM Trials**: {len(oom_trials)} trials encountered out-of-memory errors\n")
            if len(oom_trials) > 0:
                f.write("- **Examples**: ")
                examples = []
                for t in oom_trials[:3]:
                    batch_size = t.params.get("batch_size", "N/A")
                    arch = t.params.get("architecture_size", "N/A")
                    examples.append(f"Trial #{t.number} (bs={batch_size}, arch={arch})")
                f.write(", ".join(examples) + "\n")

    logger.info(f"Generated study summary: {summary_path}")
    return summary_path


def list_all_studies(storage_url: str = STORAGE_URL) -> list[str]:
    try:
        storage = optuna.storages.RDBStorage(url=storage_url)
        study_summaries = storage.get_all_studies()
        return sorted([s.study_name for s in study_summaries])
    except Exception as e:
        logger.error(f"Failed to list studies: {e}")
        return []

