"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def sample_entities():
    """Load sample entities for testing."""
    entities_path = Path(__file__).parent.parent / "sample_entities.json"
    if entities_path.exists():
        with open(entities_path) as f:
            return json.load(f)
    return [
        {
            "id": "Q1",
            "name": "Entity 1",
            "description": "Test entity 1",
            "aliases": ["E1", "Entity One"],
        },
        {
            "id": "Q2",
            "name": "Entity 2",
            "description": "Test entity 2",
            "aliases": ["E2", "Entity Two"],
        },
    ]


@pytest.fixture(scope="session")
def sample_text():
    """Sample text for testing."""
    return "Apple Inc. is a technology company founded by Steve Jobs in California."


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def simple_config():
    """Basic configuration for testing."""
    return {
        "retriever_model": "microsoft/deberta-v3-small",
        "reader_model": "microsoft/deberta-v3-small",  # Use small model for testing
        "top_k_candidates": 10,
        "consensus_threshold": 0.6,
        "max_seq_length": 128,  # Changed from max_length to match SystemConfig
        "batch_size": 2,
    }


@pytest.fixture
def temp_entity_file(tmp_path, sample_entities):
    """Create a temporary entity file."""
    entity_file = tmp_path / "entities.json"
    with open(entity_file, "w") as f:
        json.dump(sample_entities, f)
    return entity_file


@pytest.fixture
def mock_faiss_index(monkeypatch):
    """Mock FAISS index for testing without actual embeddings."""

    class MockIndex:
        def __init__(self, dimension):
            self.dimension = dimension
            self.ntotal = 0

        def add(self, vectors):
            self.ntotal += len(vectors)

        def search(self, queries, k):
            # Return dummy results
            import numpy as np

            batch_size = len(queries)
            distances = np.random.rand(batch_size, k).astype(np.float32)
            indices = np.random.randint(0, 100, (batch_size, k)).astype(np.int64)
            return distances, indices

    def mock_index_flat_ip(dimension):
        return MockIndex(dimension)

    try:
        import faiss

        monkeypatch.setattr(faiss, "IndexFlatIP", mock_index_flat_ip)
    except ImportError:
        pass

    return MockIndex
