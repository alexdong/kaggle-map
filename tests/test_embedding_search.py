"""Test embedding search functionality and Optuna compatibility."""

import tempfile

import optuna
import pytest

from kaggle_map.strategies.mlp import MLPStrategy


def test_optuna_categorical_distribution_dynamic_values() -> None:
    """Test that demonstrates the Optuna CategoricalDistribution dynamic value space error."""

    # Create a temporary database for the study
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        storage = f"sqlite:///{tmp.name}"

        # Create a study
        study = optuna.create_study(
            study_name="test_categorical",
            storage=storage,
            direction="maximize",
            load_if_exists=False,
        )

        # First trial with one set of choices
        def objective1(trial) -> float:
            # This works fine for the first trial
            trial.suggest_categorical("param", ["A", "B", "C"])
            return 1.0

        study.optimize(objective1, n_trials=1)

        # Second trial trying to use different choices - this will fail
        def objective2(trial) -> float:
            # This will raise ValueError: CategoricalDistribution does not support dynamic value space
            trial.suggest_categorical("param", ["A", "B", "C", "D"])  # Added "D"
            return 2.0

        # This should raise the error we're seeing
        with pytest.raises(ValueError, match="CategoricalDistribution does not support dynamic value space"):
            study.optimize(objective2, n_trials=1)


def test_embedding_search_space_consistency() -> None:
    """Test that embedding search space maintains consistent choices across trials."""
    # Create a mock trial
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        storage = f"sqlite:///{tmp.name}"
        study = optuna.create_study(storage=storage, direction="maximize")

        # Test that get_embedding_search_space exists and returns consistent choices
        def objective(trial) -> float:
            # This should work if the method exists and uses consistent choices
            params = MLPStrategy.get_embedding_search_space(trial)

            # Verify expected parameters are present
            assert "embedding_model" in params
            assert "learning_rate" in params
            assert "batch_size" in params

            return 1.0

        # This should work for multiple trials if choices are consistent
        study.optimize(objective, n_trials=3)

        # Verify all trials used the same categorical choices
        trials = study.trials
        assert len(trials) == 3

        # Check that embedding_model choices are consistent
        for trial in trials:
            assert "embedding_model" in trial.params
