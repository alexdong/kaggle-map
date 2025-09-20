"""Optuna metadata derivation utilities for Pydantic models."""

import types
from enum import Enum
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel

OptunaMetadata = dict[str, dict[str, Any]]


def derive_optuna_metadata(
    model_cls: type[BaseModel],
    *,
    log_scale_fields: set[str] | None = None,
    categorical_field_weights: dict[str, list[float]] | None = None,
) -> OptunaMetadata:
    """Derive Optuna distributions from field constraints on ``model_cls``.

    Args:
        model_cls: Pydantic model defining hyperparameters.
        log_scale_fields: Field names that should explore float ranges in log space.
        categorical_field_weights: Optional sampling weights for enum-backed fields.

    Returns:
        Mapping of field names to Optuna distribution metadata.
    """

    assert issubclass(model_cls, BaseModel), "model_cls must inherit from BaseModel"

    log_fields = log_scale_fields or set()
    weights = categorical_field_weights or {}

    hints = get_type_hints(model_cls, include_extras=True)
    metadata: OptunaMetadata = {}

    for field_name, field_info in model_cls.model_fields.items():
        annotation = hints.get(field_name)
        if annotation is None:
            continue

        base_type = _unwrap_annotation(annotation)
        if base_type is None:
            continue

        optuna_payload = _build_metadata_for_field(
            field_name,
            base_type,
            field_info,
            log_fields,
            weights,
        )

        if optuna_payload is not None:
            metadata[field_name] = optuna_payload

    unknown_log = log_fields.difference(metadata)
    assert not unknown_log, f"Log-scale fields not found or unsupported: {sorted(unknown_log)}"

    unknown_weights = set(weights).difference(metadata)
    assert not unknown_weights, f"Weighted categorical fields not found: {sorted(unknown_weights)}"

    return metadata


def attach_optuna_metadata(model_cls: type[BaseModel], metadata: OptunaMetadata) -> None:
    """Attach Optuna metadata to ``json_schema_extra`` for later consumers."""

    for field_name, optuna_payload in metadata.items():
        assert field_name in model_cls.model_fields, f"Unknown field: {field_name}"
        field_info = model_cls.model_fields[field_name]
        assert not field_info.json_schema_extra, (
            f"Field '{field_name}' already defines json_schema_extra; refusing to overwrite"
        )
        field_info.json_schema_extra = {"optuna": dict(optuna_payload)}


def _unwrap_annotation(annotation: Any) -> type[Any] | None:
    origin = get_origin(annotation)

    if origin is Annotated:
        return _unwrap_annotation(get_args(annotation)[0])

    if origin in (types.UnionType, Union):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _unwrap_annotation(args[0])
        return None

    return annotation if isinstance(annotation, type) else None


def _build_metadata_for_field(
    field_name: str,
    base_type: type[Any],
    field_info: Any,
    log_fields: set[str],
    weights: dict[str, list[float]],
) -> dict[str, Any] | None:
    if isinstance(base_type, type) and issubclass(base_type, Enum):
        return _enum_metadata(field_name, base_type, weights)

    if base_type is int:
        return _numeric_metadata(field_name, field_info, base_type, log_fields)

    if base_type is float:
        return _numeric_metadata(field_name, field_info, base_type, log_fields)

    return None


def _enum_metadata(
    field_name: str,
    enum_cls: type[Enum],
    weights: dict[str, list[float]],
) -> dict[str, Any]:
    choices = [member.value for member in enum_cls]
    payload: dict[str, Any] = {"distribution": "categorical", "choices": choices}

    if field_name in weights:
        field_weights = weights[field_name]
        assert len(field_weights) == len(choices), f"Weights for '{field_name}' must match number of choices"
        payload["weights"] = list(field_weights)

    return payload


def _numeric_metadata(
    field_name: str,
    field_info: Any,
    base_type: type[Any],
    log_fields: set[str],
) -> dict[str, Any]:
    bounds = _extract_bounds(field_name, field_info, base_type)
    payload: dict[str, Any] = {"low": bounds[0], "high": bounds[1]}
    payload["distribution"] = "int" if base_type is int else "float"

    if field_name in log_fields:
        assert base_type is float, f"Log scale only valid for float fields: {field_name}"
        payload["log"] = True

    return payload


def _extract_bounds(
    field_name: str,
    field_info: Any,
    base_type: type[Any],
) -> tuple[float, float]:
    lower: float | None = None
    upper: float | None = None

    for constraint in getattr(field_info, "metadata", ()):  # type: ignore[attr-defined]
        if isinstance(constraint, Ge):
            lower = _max_value(lower, constraint.ge)
        elif isinstance(constraint, Gt):
            bound = constraint.gt + 1 if base_type is int else constraint.gt
            lower = _max_value(lower, bound)
        elif isinstance(constraint, Le):
            upper = _min_value(upper, constraint.le)
        elif isinstance(constraint, Lt):
            bound = constraint.lt - 1 if base_type is int else constraint.lt
            upper = _min_value(upper, bound)

    assert lower is not None, f"Field '{field_name}' must define a lower bound"
    assert upper is not None, f"Field '{field_name}' must define an upper bound"
    assert lower < upper, f"Field '{field_name}' has invalid bounds: lower={lower}, upper={upper}"

    return lower, upper


def _max_value(current: float | None, candidate: float) -> float:
    if current is None or candidate > current:
        return candidate
    return current


def _min_value(current: float | None, candidate: float) -> float:
    if current is None or candidate < current:
        return candidate
    return current
