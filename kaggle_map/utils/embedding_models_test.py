import pytest
import numpy as np
from unittest.mock import Mock, patch

from kaggle_map.utils.embedding_models import EmbeddingModel

def test_embedding_metadata_dims_and_seq():
    assert EmbeddingModel.MINI_LM.dim == 384
    assert EmbeddingModel.MINI_LM.recommended_max_seq == 256
    assert EmbeddingModel.MP_NET.dim == 768
    assert EmbeddingModel.BGE_SMALL.dim == 512