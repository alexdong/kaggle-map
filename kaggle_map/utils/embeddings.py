from __future__ import annotations

"""Embedding model registry and metadata.

Defines the `EmbeddingModel` enum with a few strong baseline choices and
metadata helpful for configuration (dimensions and recommended max sequence).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str
    dim: int
    recommended_max_seq: int
    notes: str = ""


class EmbeddingModel(Enum):
    """Common sentence-embedding backbones with handy metadata.

    - all-MiniLM-L6-v2: Fast, 384d, great on short texts; default to 256 tokens.
    - all-mpnet-base-v2: Stronger on general STS/retrieval; 768d but slower/larger.
    - gte-small: Modern small, 384d; quick like MiniLM.
    - gte-base-en-v1.5: 768d; similar size/speed to MPNet.
    - bge-small-en-v1.5: Strong small, 512d; still efficient.
    """

    MINI_LM = "sentence-transformers/all-MiniLM-L6-v2"
    MP_NET = "sentence-transformers/all-mpnet-base-v2"
    GTE_SMALL = "thenlper/gte-small"
    GTE_BASE = "Alibaba-NLP/gte-base-en-v1.5"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"

    @property
    def spec(self) -> EmbeddingSpec:
        specs: Dict[EmbeddingModel, EmbeddingSpec] = {
            EmbeddingModel.MINI_LM: EmbeddingSpec(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                dim=384,
                recommended_max_seq=256,
                notes=(
                    "Fast, tiny index; strong on short student explanations. "
                    "Pairs well with token budgeting and Answer one-hot."
                ),
            ),
            EmbeddingModel.MP_NET: EmbeddingSpec(
                model_id="sentence-transformers/all-mpnet-base-v2",
                dim=768,
                recommended_max_seq=512,
                notes=(
                    "Runner-up; often a bit stronger on general retrieval. "
                    "~2–3x slower, 2x vector size vs MiniLM."
                ),
            ),
            EmbeddingModel.GTE_SMALL: EmbeddingSpec(
                model_id="thenlper/gte-small",
                dim=384,
                recommended_max_seq=512,
                notes="Small modern alternative; quick like MiniLM.",
            ),
            EmbeddingModel.GTE_BASE: EmbeddingSpec(
                model_id="Alibaba-NLP/gte-base-en-v1.5",
                dim=768,
                recommended_max_seq=512,
                notes="Base-sized GTE; similar size/speed to MPNet.",
            ),
            EmbeddingModel.BGE_SMALL: EmbeddingSpec(
                model_id="BAAI/bge-small-en-v1.5",
                dim=512,
                recommended_max_seq=512,
                notes="Strong small model; slightly larger vectors than MiniLM.",
            ),
        }
        return specs[self]

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def recommended_max_seq(self) -> int:
        return self.spec.recommended_max_seq

    @staticmethod
    def all() -> List["EmbeddingModel"]:
        return list(EmbeddingModel)

