"""Unit tests for simple helper functions in kaggle_map.llm.api."""

import pytest

from kaggle_map.llm import api


def test_validate_prompts_accepts_string() -> None:
    payload = {api.PROMPTS_KEY: "  hello "}
    assert api._validate_prompts(payload) == ["hello"]


def test_validate_prompts_accepts_list() -> None:
    payload = {api.PROMPTS_KEY: [" first ", "second"]}
    assert api._validate_prompts(payload) == ["first", "second"]


def test_validate_prompts_rejects_empty_entries() -> None:
    payload = {api.PROMPTS_KEY: ["   "]}
    with pytest.raises(AssertionError):
        api._validate_prompts(payload)


def test_completion_payload_normalises_whitespace() -> None:
    result = api._completion_payload([" one ", "two"])
    assert result == {api.COMPLETIONS_KEY: ["one", "two"]}


def test_completion_payload_rejects_empty_output() -> None:
    with pytest.raises(AssertionError):
        api._completion_payload(["", " "])
