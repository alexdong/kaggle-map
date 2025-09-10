"""CLI utility functions and custom Click types.

This module provides reusable CLI components for the kaggle-map project.
"""

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from enum import Enum


class EnumChoice(click.Choice):
    """Click type for enum choices that returns the enum instance."""

    def __init__(self, enum_type: type["Enum"]) -> None:
        self.enum_type = enum_type
        super().__init__([e.value for e in enum_type])

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> "Enum":
        # First use parent to validate the choice
        converted_value = super().convert(value, param, ctx)
        # Then convert back to enum
        return self.enum_type(converted_value)
