from dataclasses import dataclass
from enum import Enum

from sentence_transformers import SentenceTransformer

from kaggle_map.utils.device import get_device

"""Embedding model registry and metadata.

Defines the `EmbeddingModel` enum with a few strong baseline choices and
metadata helpful for configuration (dimensions and recommended max sequence).

IMPORTANT: All embeddings in this system use concatenated question + answer embeddings,
resulting in 2x the base model dimensions (e.g., 384-dim model → 768-dim concatenated).
This provides richer representations by separately encoding questions and answers.
"""


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str
    base_dim: int  # Base dimension of the model
    recommended_max_seq: int
    notes: str = ""

    @property
    def dim(self) -> int:
        """Effective embedding dimension after concatenation (2x base_dim)."""
        return 2 * self.base_dim


class EmbeddingModel(Enum):
    MINI_LM = "sentence-transformers/all-MiniLM-L6-v2"
    MP_NET = "sentence-transformers/all-mpnet-base-v2"
    GTE_SMALL = "thenlper/gte-small"
    GTE_BASE = "Alibaba-NLP/gte-base-en-v1.5"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    # Additional models for embedding search
    E5_BASE = "intfloat/e5-base-v2"
    INSTRUCTOR_BASE = "hkunlp/instructor-base"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    CONTRIEVER = "facebook/contriever"
    SENTENCE_T5_BASE = "sentence-transformers/sentence-t5-base"
    MINI_LM_L12 = "sentence-transformers/all-MiniLM-L12-v2"

    @property
    def spec(self) -> EmbeddingSpec:
        specs: dict[EmbeddingModel, EmbeddingSpec] = {
            EmbeddingModel.MINI_LM: EmbeddingSpec(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                base_dim=384,
                recommended_max_seq=256,
                notes=(
                    "Fast, tiny index; strong on short student explanations. "
                    "Pairs well with token budgeting and Answer one-hot."
                ),
            ),
            EmbeddingModel.MP_NET: EmbeddingSpec(
                model_id="sentence-transformers/all-mpnet-base-v2",
                base_dim=768,
                recommended_max_seq=512,
                notes=("Runner-up; often a bit stronger on general retrieval. ~2-3x slower, 2x vector size vs MiniLM."),
            ),
            EmbeddingModel.GTE_SMALL: EmbeddingSpec(
                model_id="thenlper/gte-small",
                base_dim=384,
                recommended_max_seq=512,
                notes="Small modern alternative; quick like MiniLM.",
            ),
            EmbeddingModel.GTE_BASE: EmbeddingSpec(
                model_id="Alibaba-NLP/gte-base-en-v1.5",
                base_dim=768,
                recommended_max_seq=512,
                notes="Base-sized GTE; similar size/speed to MPNet.",
            ),
            EmbeddingModel.BGE_SMALL: EmbeddingSpec(
                model_id="BAAI/bge-small-en-v1.5",
                base_dim=512,
                recommended_max_seq=512,
                notes="Strong small model; slightly larger vectors than MiniLM.",
            ),
            EmbeddingModel.E5_BASE: EmbeddingSpec(
                model_id="intfloat/e5-base-v2",
                base_dim=768,
                recommended_max_seq=512,
                notes="Strong balanced model with instruction following capabilities.",
            ),
            EmbeddingModel.INSTRUCTOR_BASE: EmbeddingSpec(
                model_id="hkunlp/instructor-base",
                base_dim=768,
                recommended_max_seq=512,
                notes="Task-specific instruction embeddings for better domain adaptation.",
            ),
            EmbeddingModel.BGE_BASE: EmbeddingSpec(
                model_id="BAAI/bge-base-en-v1.5",
                base_dim=768,
                recommended_max_seq=512,
                notes="Modern efficient architecture with strong performance.",
            ),
            EmbeddingModel.CONTRIEVER: EmbeddingSpec(
                model_id="facebook/contriever",
                base_dim=768,
                recommended_max_seq=512,
                notes="Facebook's unsupervised dense retrieval model.",
            ),
            EmbeddingModel.SENTENCE_T5_BASE: EmbeddingSpec(
                model_id="sentence-transformers/sentence-t5-base",
                base_dim=768,
                recommended_max_seq=512,
                notes="T5-based sentence embeddings with strong generalization.",
            ),
            EmbeddingModel.MINI_LM_L12: EmbeddingSpec(
                model_id="sentence-transformers/all-MiniLM-L12-v2",
                base_dim=384,
                recommended_max_seq=512,
                notes="Deeper MiniLM variant with 12 layers for better representations.",
            ),
        }
        return specs[self]

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def base_dim(self) -> int:
        """Base dimension of the embedding model."""
        return self.spec.base_dim

    @property
    def dim(self) -> int:
        """Effective embedding dimension after concatenation (2x base_dim)."""
        return self.spec.dim

    @property
    def recommended_max_seq(self) -> int:
        return self.spec.recommended_max_seq

    @staticmethod
    def all() -> list["EmbeddingModel"]:
        return list(EmbeddingModel)


def get_tokenizer(
    model: EmbeddingModel = EmbeddingModel.MINI_LM,
    device: str | None = None,
) -> "SentenceTransformer":
    # If device not specified, use the get_device utility
    if device is None:
        device = str(get_device())

    # Load model to CPU first to avoid meta tensor issues in parallel processes
    st_model = SentenceTransformer(model.model_id, device="cpu")

    # Move to target device if not CPU
    if device != "cpu":
        st_model = st_model.to(device)

    return st_model
