"""GPT-OSS 20B Modal deployment for LLM-assisted misconception scoring.

Key decisions and conventions
-----------------------------
- Hosting: deploy with `modal deploy kaggle_map.llm.api`; the module registers a FastAPI POST
  endpoint (`/completions`) on Modal, backed by the GPT-OSS 20B Q2_K_L GGUF weights.
- Bundling: the Modal image bakes the repository into the container via `add_local_dir` and mounts
  a persistent volume at `/root/kaggle-map/models` for downloaded weights.
- Scaling: environment variables `MODAL_MIN_CONTAINERS`, `MODAL_MAX_CONTAINERS`, and either
  `MODAL_SCALEDOWN_WINDOW` or the legacy `MODAL_CONTAINER_IDLE_TIMEOUT` tune worker availability.
  Defaults keep one warm worker; `(0, 3)` is cost-friendly, `(1, 3)` keeps latency low.
- Interfaces: this module exposes a single Modal web endpoint; invoke via HTTP or from other Modal
  apps, not a local CLI.
- API contract: submit `{"prompt": "..."}` and receive `{"completion": "..."}`.

Quick reference
---------------
1. Deploy from Makefile
       make deploy-llm
2. Remote curl smoke test (replace `YOUR_URL` with Modal output)
       curl -X POST "https://YOUR_URL/completions" \
           -H "Content-Type: application/json" \
           -d "{\"prompt\": \"1/3 * 2/3 =\"}"
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal
from llama_cpp import Llama
from loguru import logger

from kaggle_map.utils.gguf_model import (
    GGUFModelInferenceConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_model_path,
    load_llm_model,
)
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"
APP_NAME = "kaggle-map-gpt-oss"
VOLUME_NAME = "kaggle-map-gpt-oss-gguf-cache"
MODEL_NAME = GGUFModelName.GPT_OSS_20B
DEFAULT_QUANTIZATION = GGUFModelQuantizationLevel.Q2_K_L
CONTAINER_ROOT = Path("/root/kaggle-map")


MIN_CONTAINERS = int(os.environ.get("MODAL_MIN_CONTAINERS", "1"))
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "8"))
SCALEDOWN_WINDOW_SECONDS = int(
    os.environ.get("MODAL_SCALEDOWN_WINDOW", os.environ.get("MODAL_CONTAINER_IDLE_TIMEOUT", "30"))
)

assert CONTAINER_ROOT.is_absolute(), "CONTAINER_ROOT must be absolute."
assert MIN_CONTAINERS <= MAX_CONTAINERS, "MODAL_MIN_CONTAINERS cannot exceed MODAL_MAX_CONTAINERS."

MODEL_WEIGHTS_PATH = (CONTAINER_ROOT / get_model_path(MODEL_NAME, DEFAULT_QUANTIZATION)).resolve()


def _validate_prompt(payload: dict[str, Any]) -> str:
    assert isinstance(payload, dict), "Request payload must be a dictionary."
    assert PROMPT_KEY in payload, f"Request payload must include '{PROMPT_KEY}'."
    prompt = str(payload[PROMPT_KEY]).strip()
    assert prompt, "Prompt must contain non-whitespace characters."
    return prompt


def _completion_payload(text: str) -> dict[str, str]:
    completion = text.strip()
    assert completion, "Completion must not be empty."
    return {COMPLETION_KEY: completion}


@dataclass(slots=True)
class InferenceArtifacts:
    llm: Llama
    inference_config: GGUFModelInferenceConfig


app = modal.App(name=APP_NAME)

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "git")
    .pip_install(
        "fastapi>=0.111.0",
        "llama-cpp-python>=0.2.90",
        "huggingface-hub>=0.20.0",
        "numpy>=2.0.0",
        "loguru>=0.7.0",
        "pydantic>=2.5.0",
        "rich>=13.0.0",
    )
    .add_local_dir(
        local_path=".",
        remote_path=CONTAINER_ROOT.as_posix(),
        copy=True,
        ignore=[".git", "__pycache__", "logs", "datasets", "models"],
    )
)

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
volume_mount_path = (CONTAINER_ROOT / "models").as_posix()


def _build_inference_artifacts() -> InferenceArtifacts:
    logger.info("Loading GGUF model %s with quantization %s", MODEL_NAME.value, DEFAULT_QUANTIZATION.value)
    inference_config = GGUFModelInferenceConfig.get_default_config(MODEL_NAME)
    llm = load_llm_model(MODEL_NAME)
    return InferenceArtifacts(llm=llm, inference_config=inference_config)


def _generate_from_model(
    *,
    llm: Llama,
    inference_config: GGUFModelInferenceConfig,
    model_name: GGUFModelName,
    prompt: str,
) -> str:
    formatted_prompt = format_chat_prompt(model_name, prompt)
    logger.debug("Prompt length after formatting: %d", len(formatted_prompt))
    response = llm.create_completion(
        prompt=formatted_prompt,
        temperature=inference_config.temperature,
        top_p=inference_config.top_p,
        max_tokens=inference_config.max_tokens,
        stop=inference_config.stop_words,
        repeat_penalty=inference_config.repeat_penalty,
    )
    choices = response.get("choices")  # type: ignore[arg-type]
    assert choices, "LLM completion returned no choices."
    first_choice = choices[0]
    text = str(first_choice.get("text", "")).strip()
    if not text:
        message = first_choice.get("message")
        if isinstance(message, dict):
            text = str(message.get("content", "")).strip()
    assert text, "LLM completion was empty."
    logger.debug("Generated completion length: %d", len(text))
    return text


@app.cls(
    image=image,
    gpu="T4",
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={volume_mount_path: model_volume},
)
class GPTOSSService:
    @modal.enter()
    def enter(self) -> None:
        logger.info("Warming GPT-OSS 20B worker")
        self._artifacts = _build_inference_artifacts()
        logger.success("GPT-OSS 20B ready for requests")

    def _artifacts_or_fail(self) -> InferenceArtifacts:
        artifacts: InferenceArtifacts | None = getattr(self, "_artifacts", None)
        assert artifacts is not None, "Inference artifacts missing."
        return artifacts

    @modal.fastapi_endpoint(method="POST")
    def completions(self, request: dict[str, Any]) -> dict[str, str]:
        prompt = _validate_prompt(request)
        artifacts = self._artifacts_or_fail()
        completion = _generate_from_model(
            llm=artifacts.llm,
            inference_config=artifacts.inference_config,
            model_name=MODEL_NAME,
            prompt=prompt,
        )
        return _completion_payload(completion)
