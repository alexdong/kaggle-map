"""Tests for checkpoint save/load helpers."""

from pathlib import Path

import torch

from kaggle_map.core.models import MLPTrainingConfig, default_mlp_training_config
from kaggle_map.mlp.checkpoint import load_checkpoint, save_checkpoint
from kaggle_map.mlp.model import QuestionSpecificMLP
from kaggle_map.utils.device import get_device


def _build_model() -> tuple[QuestionSpecificMLP, MLPTrainingConfig, torch.device]:
    config = default_mlp_training_config()
    predictions = {42: ["True_Correct:NA", "False_Incorrect:Computation"]}
    torch.manual_seed(0)
    model = QuestionSpecificMLP(
        predictions,
        embedding_model=config.embedding_model,
        embedding_strategy=config.embedding_strategy,
        architecture_size=config.architecture_size,
        dropout=config.dropout,
        activation=config.activation,
    )
    device = get_device()
    model = model.to(device)
    return model, config, device


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model, config, device = _build_model()
    target = tmp_path / "mlp_checkpoint.pt"

    save_checkpoint(model, target, config)
    assert target.exists()

    payload = torch.load(target, weights_only=False)
    assert "embedding_dim" not in payload["config"], "embedding_dim should be derived at load time"

    loaded_model, loaded_config = load_checkpoint(target)

    assert isinstance(loaded_model, QuestionSpecificMLP)
    assert loaded_config.embedding_model == config.embedding_model
    assert loaded_config.embedding_strategy == config.embedding_strategy
    assert next(loaded_model.parameters()).device.type == device.type

    original_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    loaded_state = {k: v.detach().cpu() for k, v in loaded_model.state_dict().items()}
    assert original_state.keys() == loaded_state.keys()
    for name, tensor in original_state.items():
        assert torch.equal(tensor, loaded_state[name])
