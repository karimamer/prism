"""Performance benchmarks for PRISM components."""

import numpy as np
import pytest


@pytest.mark.benchmark
def test_vector_search_performance(benchmark, mock_faiss_index):
    """Benchmark vector search performance."""

    def search_vectors():
        index = mock_faiss_index(768)
        # Add some dummy vectors
        vectors = np.random.rand(1000, 768).astype(np.float32)
        index.add(vectors)

        # Search
        query = np.random.rand(1, 768).astype(np.float32)
        distances, indices = index.search(query, k=10)
        return distances, indices

    result = benchmark(search_vectors)
    assert result is not None


@pytest.mark.benchmark
def test_text_encoding_performance(benchmark, simple_config):
    """Benchmark text encoding performance."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(simple_config["retriever_model"])
    text = "Apple Inc. is a technology company founded by Steve Jobs in California."

    def encode_text():
        return tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    result = benchmark(encode_text)
    assert result is not None


@pytest.mark.benchmark
def test_batch_processing_performance(benchmark, simple_config):
    """Benchmark batch processing performance."""

    texts = [
        "Apple Inc. is a technology company.",
        "Steve Jobs founded Apple in California.",
        "The iPhone was released in 2007.",
    ] * 10  # 30 texts

    def process_batch():
        # Simple processing simulation
        results = []
        for text in texts:
            results.append({"text": text, "entities": []})
        return results

    result = benchmark(process_batch)
    assert len(result) == len(texts)
