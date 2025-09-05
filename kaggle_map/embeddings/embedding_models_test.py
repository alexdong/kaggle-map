import pytest
import numpy as np
from unittest.mock import Mock, patch

from kaggle_map.embeddings.embedding_models import EmbeddingModel

def test_embedding_metadata_dims_and_seq():
    # Dimensions are 2x base_dim due to concatenation
    assert EmbeddingModel.MINI_LM.dim == 768  # 2 * 384
    assert EmbeddingModel.MINI_LM.recommended_max_seq == 256
    assert EmbeddingModel.MP_NET.dim == 1536  # 2 * 768
    assert EmbeddingModel.BGE_SMALL.dim == 1024  # 2 * 512