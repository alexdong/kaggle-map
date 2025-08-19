from __future__ import annotations

from kaggle_map.utils.embeddings import EmbeddingModel


def test_embedding_enum_has_expected_models():
    ids = {m.model_id for m in EmbeddingModel.all()}
    assert {
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "thenlper/gte-small",
        "Alibaba-NLP/gte-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
    }.issubset(ids)


def test_embedding_metadata_dims_and_seq():
    assert EmbeddingModel.MINI_LM.dim == 384
    assert EmbeddingModel.MINI_LM.recommended_max_seq == 256
    assert EmbeddingModel.MP_NET.dim == 768
    assert EmbeddingModel.BGE_SMALL.dim == 512

