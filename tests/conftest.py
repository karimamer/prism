"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# =============================================================================
# Pytest Configuration Hooks
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_model: Tests requiring model downloads")
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test location and naming."""
    for item in items:
        # Auto-mark based on directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Auto-mark slow tests
        if "slow" in item.nodeid.lower() or "e2e" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Auto-mark tests that require models
        if any(keyword in item.nodeid.lower() for keyword in ["atg", "relik", "spel", "unirel"]):
            item.add_marker(pytest.mark.requires_model)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="run end-to-end tests that download models",
    )


# =============================================================================
# Session-scoped Fixtures (Shared Across All Tests)
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_knowledge_base():
    """Standard knowledge base for testing."""
    return {
        "Q312": {
            "id": "Q312",
            "name": "Apple Inc.",
            "description": "American multinational technology company",
            "aliases": ["Apple", "Apple Computer", "Apple Computer Inc."],
            "type": "ORG",
        },
        "Q19837": {
            "id": "Q19837",
            "name": "Steve Jobs",
            "description": "American entrepreneur and co-founder of Apple",
            "aliases": ["Steven Paul Jobs", "Jobs", "Steven Jobs"],
            "type": "PER",
        },
        "Q99": {
            "id": "Q99",
            "name": "California",
            "description": "State of the United States of America",
            "aliases": ["CA", "Calif.", "State of California"],
            "type": "LOC",
        },
        "Q2283": {
            "id": "Q2283",
            "name": "Microsoft",
            "description": "American multinational technology corporation",
            "aliases": ["Microsoft Corporation", "MSFT"],
            "type": "ORG",
        },
        "Q8074": {
            "id": "Q8074",
            "name": "Bill Gates",
            "description": "American business magnate and philanthropist",
            "aliases": ["William Henry Gates III", "William Gates"],
            "type": "PER",
        },
    }


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
def sample_texts():
    """Standard test texts covering various scenarios."""
    return {
        "simple": "Apple is a company.",
        "multiple_entities": "Apple was founded by Steve Jobs in California.",
        "ambiguous": "Apple could be a fruit or a company.",
        "long": "Apple Inc. is a technology company. " * 50,
        "special_chars": "Apple™ Inc. (2023) @AppleSupport #Apple",
        "unicode": "Apple Inc. (アップル) is based in California.",
        "rtl": "Apple في كاليفورنيا",
        "emoji": "Apple 😀 makes great products 🎉",
        "entity_at_start": "Apple makes iPhones",
        "entity_at_end": "I love Apple",
        "entity_only": "Apple",
        "overlapping": "New York City is in New York State",
        "nested": "The Apple Store in New York City",
    }


@pytest.fixture(scope="session")
def sample_text():
    """Sample text for testing (backward compatibility)."""
    return "Apple Inc. is a technology company founded by Steve Jobs in California."


@pytest.fixture(scope="session")
def simple_config():
    """Basic configuration for testing."""
    return {
        "retriever_model": "microsoft/deberta-v3-small",
        "reader_model": "microsoft/deberta-v3-small",
        "top_k_candidates": 10,
        "consensus_threshold": 0.6,
        "max_seq_length": 128,
        "batch_size": 2,
    }


# =============================================================================
# Function-scoped Fixtures (Created for Each Test)
# =============================================================================


@pytest.fixture(autouse=True)
def reset_torch_random():
    """Reset random seed before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def temp_entity_file(tmp_path, sample_entities):
    """Create a temporary entity file."""
    entity_file = tmp_path / "entities.json"
    with open(entity_file, "w") as f:
        json.dump(sample_entities, f)
    return entity_file


@pytest.fixture
def temp_kb_file(tmp_path, sample_knowledge_base):
    """Create a temporary knowledge base file."""
    kb_file = tmp_path / "knowledge_base.json"
    with open(kb_file, "w") as f:
        json.dump(sample_knowledge_base, f)
    return kb_file


# =============================================================================
# Mock Fixtures for Fast Unit Testing
# =============================================================================


@pytest.fixture
def mock_transformers_model(monkeypatch):
    """Mock transformer models to avoid downloads."""

    class MockModel(torch.nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config or type("Config", (), {"hidden_size": 768})()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            return type(
                "Output",
                (),
                {
                    "last_hidden_state": torch.randn(batch_size, seq_len, 768),
                    "pooler_output": torch.randn(batch_size, 768),
                },
            )()

    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            self.model_max_length = 512
            self.vocab_size = 30522

        def __call__(self, text, **kwargs):
            if isinstance(text, str):
                text = [text]
            batch_size = len(text)
            seq_len = kwargs.get("max_length", 128)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }

        def encode(self, text, **kwargs):
            return [1, 2, 3, 4, 5]

        def decode(self, ids, **kwargs):
            return "decoded text"

        def batch_decode(self, ids_list, **kwargs):
            return ["decoded text"] * len(ids_list)

    with patch("transformers.AutoModel.from_pretrained", return_value=MockModel()):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer()):
            yield


@pytest.fixture
def mock_faiss_index(monkeypatch):
    """Mock FAISS index for testing without actual embeddings."""

    class MockIndex:
        def __init__(self, dimension):
            self.dimension = dimension
            self.ntotal = 0
            self._vectors = []

        def add(self, vectors):
            self.ntotal += len(vectors)
            self._vectors.extend(vectors)

        def search(self, queries, k):
            batch_size = len(queries)
            distances = np.random.rand(batch_size, k).astype(np.float32)
            indices = np.random.randint(0, max(self.ntotal, 1), (batch_size, k)).astype(np.int64)
            return distances, indices

        def reset(self):
            self.ntotal = 0
            self._vectors = []

    def mock_index_flat_ip(dimension):
        return MockIndex(dimension)

    try:
        import faiss

        monkeypatch.setattr(faiss, "IndexFlatIP", mock_index_flat_ip)
    except ImportError:
        pass

    return MockIndex


@pytest.fixture
def mock_torch_cuda(monkeypatch):
    """Mock CUDA availability for testing on CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


# =============================================================================
# Parametrized Fixtures
# =============================================================================


@pytest.fixture(params=["bert-base-uncased", "roberta-base", "microsoft/deberta-v3-small"])
def model_name(request):
    """Parametrized fixture for testing multiple models."""
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def batch_size(request):
    """Parametrized batch sizes for testing."""
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device_type(request):
    """Parametrized device types."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def capture_warnings():
    """Capture warnings during tests."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


@pytest.fixture
def assert_no_memory_leak():
    """Helper to assert no memory leak occurred."""
    import gc

    def check():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial = torch.cuda.memory_allocated()
            yield
            gc.collect()
            torch.cuda.empty_cache()
            final = torch.cuda.memory_allocated()
            leak = final - initial
            assert leak < 1_000_000, f"Memory leak detected: {leak} bytes"
        else:
            yield

    return check
