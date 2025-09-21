"""Tests for Optuna metadata extraction utilities."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from kaggle_map.optimise.optuna_config import (
    attach_optuna_metadata,
    derive_optuna_metadata,
)


class DummyEnum(Enum):
    FIRST = "first"
    SECOND = "second"


def _make_dummy_config() -> type[BaseModel]:
    class DummyConfig(BaseModel):
        width: Annotated[int, Field(ge=16, le=64)]
        rate: Annotated[float, Field(gt=1e-5, lt=1e-2)]
        activation: Annotated[DummyEnum, Field(default=DummyEnum.FIRST)]
        path: Path = Path("datasets/train.csv")

    return DummyConfig


def test_derive_metadata_for_numeric_and_enum_fields() -> None:
    config = _make_dummy_config()
    metadata = derive_optuna_metadata(
        config,
        log_scale_fields={"rate"},
        categorical_field_weights={"activation": [0.8, 0.2]},
    )

    assert set(metadata) == {"width", "rate", "activation"}

    width = metadata["width"]
    assert width == {"distribution": "int", "low": 16, "high": 64}

    rate = metadata["rate"]
    assert rate["distribution"] == "float"
    assert rate["low"] == pytest.approx(1e-5)
    assert rate["high"] == pytest.approx(1e-2)
    assert rate["log"] is True

    activation = metadata["activation"]
    assert activation["distribution"] == "categorical"
    assert activation["choices"] == ["first", "second"]
    assert activation["weights"] == [0.8, 0.2]


def test_attach_metadata_sets_json_schema_extra() -> None:
    config = _make_dummy_config()
    metadata = derive_optuna_metadata(config)
    attach_optuna_metadata(config, metadata)

    field_info = config.model_fields["width"]
    assert field_info.json_schema_extra == {"optuna": metadata["width"]}


def test_log_scale_requires_float_field() -> None:
    config = _make_dummy_config()
    with pytest.raises(AssertionError):
        derive_optuna_metadata(config, log_scale_fields={"width"})


def test_weights_length_must_match_choices() -> None:
    config = _make_dummy_config()
    with pytest.raises(AssertionError):
        derive_optuna_metadata(
            config,
            categorical_field_weights={"activation": [1.0]},
        )


def test_numeric_fields_must_define_bounds() -> None:
    class MissingBounds(BaseModel):
        value: Annotated[int, Field(ge=1)]

    with pytest.raises(AssertionError):
        derive_optuna_metadata(MissingBounds)


def test_attach_metadata_refuses_overwrite() -> None:
    config = _make_dummy_config()
    metadata = derive_optuna_metadata(config)
    attach_optuna_metadata(config, metadata)

    with pytest.raises(AssertionError):
        attach_optuna_metadata(config, metadata)
