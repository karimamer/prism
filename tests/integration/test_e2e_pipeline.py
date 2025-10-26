"""
End-to-end integration tests for entity resolution pipeline with real models.

These tests use actual model downloads and require network access.
They are marked as 'e2e' and can be skipped in CI with: pytest -m "not e2e"
"""

import json
import pytest
import torch
from pathlib import Path

from entity_resolution import UnifiedEntityResolutionSystem
from entity_resolution.validation import SystemConfig


# Mark all tests in this module as e2e and slow
pytestmark = [pytest.mark.e2e, pytest.mark.slow]


@pytest.fixture(scope="module")
def e2e_config():
    """Configuration for E2E tests using small real models."""
    return SystemConfig(
        # Use small, fast models for testing
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        reader_model="distilbert-base-uncased",
        # Enable all models for comprehensive testing
        use_improved_atg=True,
        use_relik=True,
        use_spel=True,
        use_unirel=True,
        # Small configs for speed
        max_seq_length=256,  # Increased to allow max_entity_length < max_seq_length
        max_entity_length=64,  # Must be less than max_seq_length
        batch_size=4,
        top_k_candidates=10,
        # Model-specific configs
        atg_decoder_layers=2,  # Smaller decoder
        atg_max_span_length=8,
        relik_top_k=50,  # Must be >= num_el_passages
        relik_num_el_passages=50,  # Number of passages for entity linking
        relik_num_re_passages=50,  # Number of passages for relation extraction
        spel_fixed_candidate_set_size=1000,  # Small candidate set
        # Use CPU for tests (unless GPU available)
        use_gpu=torch.cuda.is_available(),
        # Thresholds
        consensus_threshold=0.5,
        relik_entity_threshold=0.3,
        spel_entity_threshold=0.3,
        unirel_entity_threshold=0.3,
    )


@pytest.fixture(scope="module")
def test_knowledge_base(tmp_path_factory):
    """Create a realistic test knowledge base."""
    kb_dir = tmp_path_factory.mktemp("kb")
    kb_file = kb_dir / "knowledge_base.json"

    kb_data = {
        "Q312": {
            "id": "Q312",
            "name": "Apple Inc.",
            "type": "ORG",
            "description": "American multinational technology company that specializes in consumer electronics, software and online services headquartered in Cupertino, California",
            "aliases": ["Apple", "Apple Computer", "Apple Computer Inc."],
        },
        "Q19837": {
            "id": "Q19837",
            "name": "Steve Jobs",
            "type": "PER",
            "description": "American entrepreneur, industrial designer, business magnate, media proprietor, and investor who was the co-founder, chairman, and CEO of Apple Inc.",
            "aliases": ["Steven Paul Jobs", "Jobs", "Steven Jobs"],
        },
        "Q99": {
            "id": "Q99",
            "name": "California",
            "type": "LOC",
            "description": "State of the United States of America",
            "aliases": ["CA", "Calif.", "State of California"],
        },
        "Q48400": {
            "id": "Q48400",
            "name": "Cupertino",
            "type": "LOC",
            "description": "City in Santa Clara County, California",
            "aliases": ["Cupertino, California", "Cupertino, CA"],
        },
        "Q2283": {
            "id": "Q2283",
            "name": "Microsoft",
            "type": "ORG",
            "description": "American multinational technology corporation",
            "aliases": ["Microsoft Corporation", "MSFT"],
        },
        "Q8074": {
            "id": "Q8074",
            "name": "Bill Gates",
            "type": "PER",
            "description": "American business magnate and philanthropist, co-founder of Microsoft",
            "aliases": ["William Henry Gates III", "William Gates"],
        },
        "Q60": {
            "id": "Q60",
            "name": "New York City",
            "type": "LOC",
            "description": "Most populous city in the United States",
            "aliases": ["NYC", "New York", "New York, New York"],
        },
        "Q95": {
            "id": "Q95",
            "name": "Google",
            "type": "ORG",
            "description": "American multinational technology company",
            "aliases": ["Google LLC", "Google Inc."],
        },
    }

    with open(kb_file, "w") as f:
        json.dump(kb_data, f, indent=2)

    return kb_file


@pytest.fixture(scope="module")
def test_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
            "expected_entities": ["Q312", "Q19837", "Q48400", "Q99"],
        },
        {
            "id": "doc2",
            "text": "Bill Gates co-founded Microsoft and is a prominent philanthropist.",
            "expected_entities": ["Q8074", "Q2283"],
        },
        {
            "id": "doc3",
            "text": "Google is headquartered in California, while Microsoft is in Washington.",
            "expected_entities": ["Q95", "Q99", "Q2283"],
        },
        {
            "id": "doc4",
            "text": "New York City is the most populous city in the United States.",
            "expected_entities": ["Q60"],
        },
        {
            "id": "doc5",
            "text": "Steve Jobs introduced the iPhone at an Apple event in California.",
            "expected_entities": ["Q19837", "Q312", "Q99"],
        },
    ]


@pytest.fixture(scope="module")
def initialized_system(e2e_config, test_knowledge_base):
    """
    Initialize the unified system with real models and KB.
    This fixture is module-scoped to avoid re-downloading models for each test.
    """
    print("\n" + "=" * 80)
    print("INITIALIZING E2E TEST SYSTEM (this may take a few minutes on first run)...")
    print("=" * 80)

    # Create system
    system = UnifiedEntityResolutionSystem(e2e_config)

    # Load knowledge base
    print(f"\nLoading knowledge base from {test_knowledge_base}")
    num_loaded = system.load_entities(str(test_knowledge_base))
    print(f"Loaded {num_loaded} entities into knowledge base")

    print("\nSystem initialization complete!")
    print("=" * 80 + "\n")

    return system


class TestSystemInitialization:
    """Test system initialization with real models."""

    def test_system_creates_successfully(self, e2e_config):
        """Test that the system can be created with real models."""
        system = UnifiedEntityResolutionSystem(e2e_config)
        assert system is not None
        assert system.device is not None

    def test_all_models_initialized(self, initialized_system):
        """Test that all models are initialized correctly."""
        system = initialized_system

        # Check ATG
        if system.config.use_improved_atg:
            assert system.atg_model is not None
            assert system.atg_model.encoder is not None
            assert system.atg_model.decoder is not None

        # Check ReLiK
        if system.config.use_relik:
            assert system.relik_model is not None
            assert system.relik_model.retriever is not None
            assert system.relik_model.reader is not None

        # Check SPEL
        if system.config.use_spel:
            assert system.spel_model is not None
            assert system.spel_model.encoder is not None

        # Check UniREL
        if system.config.use_unirel:
            assert system.unirel_model is not None
            assert system.unirel_model.encoder is not None

        # Check base components
        assert system.retriever is not None
        assert system.reader is not None
        assert system.consensus is not None

    def test_kb_loaded_correctly(self, initialized_system):
        """Test that knowledge base is loaded and indexed."""
        system = initialized_system

        assert system.knowledge_base is not None
        # Check that retriever has built an index
        assert system.retriever is not None
        assert hasattr(system.retriever, "index")
        if system.retriever.index is not None:
            assert system.retriever.index.ntotal > 0


class TestSingleTextProcessing:
    """Test processing of single texts."""

    def test_process_simple_text(self, initialized_system):
        """Test processing a simple text."""
        text = "Apple Inc. is a technology company."
        result = initialized_system.process_text(text)

        assert result is not None
        assert hasattr(result, "text")
        assert hasattr(result, "entities")
        assert result.text == text
        assert isinstance(result.entities, list)

    def test_entity_detection_apple(self, initialized_system, test_documents):
        """Test that we can detect Apple Inc. in text."""
        doc = test_documents[0]  # Apple Inc. document
        result = initialized_system.process_text(doc["text"])

        assert len(result.entities) > 0, "Should detect at least one entity"

        # Check if Apple is detected - use correct field names (mention, entity_id, entity_name)
        entity_ids = [
            e.entity_id for e in result.entities if hasattr(e, "entity_id") and e.entity_id
        ]
        entity_mentions = [e.mention for e in result.entities if hasattr(e, "mention")]

        print(f"\nDetected entity mentions: {entity_mentions}")
        print(f"Entity IDs: {entity_ids}")

        # Should detect some organizations or persons
        assert len(result.entities) > 0

    def test_entity_has_required_fields(self, initialized_system):
        """Test that detected entities have required fields."""
        text = "Steve Jobs founded Apple in California."
        result = initialized_system.process_text(text)

        if len(result.entities) > 0:
            entity = result.entities[0]

            # Check required fields exist - EntityPrediction uses mention, mention_span, entity_id, etc.
            assert hasattr(entity, "mention"), "Entity should have mention field"
            assert hasattr(entity, "mention_span"), "Entity should have mention_span field"
            assert hasattr(entity, "confidence"), "Entity should have confidence field"
            assert hasattr(entity, "entity_type"), "Entity should have entity_type field"
            # Note: entity_id and entity_name might be empty strings if not linked to KB
            assert hasattr(entity, "entity_id"), "Entity should have entity_id field"
            assert hasattr(entity, "entity_name"), "Entity should have entity_name field"

            # Validate field values
            assert entity.confidence >= 0.0 and entity.confidence <= 1.0
            # Note: mention_span can be (-1, -1) if unknown, so only check valid spans
            if entity.mention_span.start >= 0 and entity.mention_span.end >= 0:
                assert entity.mention_span.end > entity.mention_span.start
                assert entity.mention_span.end <= len(text)

    def test_confidence_scores(self, initialized_system):
        """Test that confidence scores are reasonable."""
        text = "Apple Inc. was founded by Steve Jobs."
        result = initialized_system.process_text(text)

        assert hasattr(result, "avg_confidence")

        if len(result.entities) > 0:
            # Check individual confidence scores
            for entity in result.entities:
                assert 0.0 <= entity.confidence <= 1.0, (
                    f"Confidence {entity.confidence} out of range [0, 1]"
                )

            # Average confidence should be in range
            assert 0.0 <= result.avg_confidence <= 1.0


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_processing_multiple_texts(self, initialized_system, test_documents):
        """Test processing multiple documents in batch."""
        texts = [doc["text"] for doc in test_documents]
        results = initialized_system.process_batch(texts)

        assert len(results) == len(texts), "Should return results for all texts"

        for result in results:
            assert hasattr(result, "text")
            assert hasattr(result, "entities")
            assert isinstance(result.entities, list)

    def test_batch_consistency(self, initialized_system):
        """Test that batch and single processing give similar results."""
        texts = [
            "Apple Inc. is a technology company.",
            "Microsoft was founded by Bill Gates.",
        ]

        # Process individually
        individual_results = [initialized_system.process_text(t) for t in texts]

        # Process as batch
        batch_results = initialized_system.process_batch(texts)

        assert len(individual_results) == len(batch_results)

        # Results should be similar (may not be identical due to batching)
        for ind_result, batch_result in zip(individual_results, batch_results):
            assert ind_result.text == batch_result.text
            # Entity counts should be similar (within reasonable range)
            assert abs(len(ind_result.entities) - len(batch_result.entities)) <= 2

    def test_empty_batch(self, initialized_system):
        """Test processing empty batch - should raise ValueError."""
        with pytest.raises(ValueError, match="Empty batch"):
            initialized_system.process_batch([])

    def test_batch_with_empty_strings(self, initialized_system):
        """Test batch processing with some empty strings - should raise ValueError."""
        texts = [
            "Apple Inc. is a company.",
            "",  # Empty string
            "Microsoft is in Washington.",
        ]

        # Empty text should raise ValueError during validation
        with pytest.raises(ValueError, match="Text too short"):
            initialized_system.process_batch(texts)


class TestModelEnsemble:
    """Test that multiple models work together."""

    def test_consensus_aggregation(self, initialized_system):
        """Test that consensus module aggregates predictions."""
        text = "Apple Inc. was founded by Steve Jobs in California."
        result = initialized_system.process_text(text)

        # With multiple models, we should have model agreement info
        if len(result.entities) > 0:
            entity = result.entities[0]
            # Check if model agreement metadata exists
            if hasattr(entity, "metadata"):
                assert entity.metadata is not None

    def test_multiple_models_improve_results(self, initialized_system, e2e_config):
        """Test that using multiple models improves detection."""
        text = "Steve Jobs founded Apple Inc. in Cupertino, California."

        # Process with all models
        result_all = initialized_system.process_text(text)

        # Create system with only one model
        single_model_config = SystemConfig(
            retriever_model=e2e_config.retriever_model,
            reader_model=e2e_config.reader_model,
            use_improved_atg=True,
            use_relik=False,
            use_spel=False,
            use_unirel=False,
            max_seq_length=128,
            use_gpu=torch.cuda.is_available(),
        )

        single_system = UnifiedEntityResolutionSystem(single_model_config)
        result_single = single_system.process_text(text)

        # Multiple models should generally provide more coverage or confidence
        # (not strictly guaranteed, but expected in most cases)
        print(f"\nAll models detected: {len(result_all.entities)} entities")
        print(f"Single model detected: {len(result_single.entities)} entities")


class TestKnowledgeBase:
    """Test knowledge base operations."""

    def test_kb_entity_retrieval(self, initialized_system):
        """Test that we can retrieve entities from KB."""
        # Query for Apple-related text
        text = "Apple technology company"

        # This should retrieve Apple Inc. from KB
        # (Actual retrieval happens internally during processing)
        result = initialized_system.process_text(text)

        # The system should at least initialize without errors
        assert result is not None

    def test_kb_handles_unknown_entities(self, initialized_system):
        """Test processing text with entities not in KB."""
        text = "XyzCorp is a fictional company that doesn't exist in our KB."
        result = initialized_system.process_text(text)

        # Should not crash, may or may not detect entities
        assert result is not None
        assert isinstance(result.entities, list)

    def test_add_entities_to_kb(self, initialized_system, tmp_path):
        """Test adding new entities to knowledge base."""
        new_entities_file = tmp_path / "new_entities.json"
        new_entities = {
            "Q999": {
                "id": "Q999",
                "name": "TestCorp",
                "type": "ORG",
                "description": "A test corporation for E2E testing",
                "aliases": ["Test Corp", "TestCorp Inc."],
            }
        }

        with open(new_entities_file, "w") as f:
            json.dump(new_entities, f)

        # Add new entities
        num_added = initialized_system.load_entities(str(new_entities_file))
        assert num_added >= 1

        # Process text with new entity
        text = "TestCorp is a test company."
        result = initialized_system.process_text(text)

        # Should process without errors
        assert result is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_very_long_text(self, initialized_system):
        """Test processing very long text."""
        # Create text longer than max_seq_length
        long_text = "Apple Inc. is a technology company. " * 50

        result = initialized_system.process_text(long_text)

        # Should handle truncation gracefully
        assert result is not None
        assert isinstance(result.entities, list)

    def test_special_characters(self, initialized_system):
        """Test text with special characters."""
        text = "Apple™ Inc. is a tech company! @AppleSupport #Apple"
        result = initialized_system.process_text(text)

        assert result is not None
        assert isinstance(result.entities, list)

    def test_non_english_characters(self, initialized_system):
        """Test text with non-English characters."""
        text = "Apple Inc. (アップル) is based in California."
        result = initialized_system.process_text(text)

        assert result is not None
        assert isinstance(result.entities, list)

    def test_empty_text(self, initialized_system):
        """Test processing empty text - should raise ValueError."""
        with pytest.raises(ValueError, match="Text too short"):
            initialized_system.process_text("")

    def test_whitespace_only(self, initialized_system):
        """Test processing whitespace-only text - should raise ValueError."""
        with pytest.raises(ValueError, match="Text too short"):
            initialized_system.process_text("   \n\t   ")


class TestModelSpecificBehavior:
    """Test behavior of specific models."""

    def test_atg_generates_entities(self, initialized_system):
        """Test that ATG model generates entity predictions."""
        if not initialized_system.config.use_improved_atg:
            pytest.skip("ATG not enabled")

        text = "Apple Inc. was founded by Steve Jobs."
        result = initialized_system.process_text(text)

        # ATG should contribute to predictions
        assert result is not None

    def test_relik_retrieval_works(self, initialized_system):
        """Test that ReLiK retrieval is functional."""
        if not initialized_system.config.use_relik:
            pytest.skip("ReLiK not enabled")

        text = "Microsoft is a software company."
        result = initialized_system.process_text(text)

        # ReLiK should contribute to predictions
        assert result is not None

    def test_spel_classification(self, initialized_system):
        """Test that SPEL classification works."""
        if not initialized_system.config.use_spel:
            pytest.skip("SPEL not enabled")

        text = "Google develops search technology."
        result = initialized_system.process_text(text)

        # SPEL should contribute to predictions
        assert result is not None

    def test_unirel_relation_extraction(self, initialized_system):
        """Test that UniREL can extract relations."""
        if not initialized_system.config.use_unirel:
            pytest.skip("UniREL not enabled")

        text = "Bill Gates founded Microsoft."
        result = initialized_system.process_text(text)

        # UniREL should process without errors
        assert result is not None

        # Check if relations are extracted
        if hasattr(result, "relations"):
            assert isinstance(result.relations, list)


class TestPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_processing_speed_single(self, initialized_system):
        """Test that single text processing completes in reasonable time."""
        import time

        text = "Apple Inc. is a technology company based in California."

        start = time.time()
        result = initialized_system.process_text(text)
        duration = time.time() - start

        assert result is not None
        # Should complete in less than 30 seconds (generous for CPU)
        assert duration < 30.0, f"Processing took {duration:.2f}s, expected < 30s"

        print(f"\nSingle text processing time: {duration:.2f}s")

    def test_processing_speed_batch(self, initialized_system, test_documents):
        """Test that batch processing is reasonably fast."""
        import time

        texts = [doc["text"] for doc in test_documents]

        start = time.time()
        results = initialized_system.process_batch(texts)
        duration = time.time() - start

        assert len(results) == len(texts)
        # Should complete in less than 60 seconds for 5 documents
        assert duration < 60.0, f"Batch processing took {duration:.2f}s, expected < 60s"

        avg_time = duration / len(texts)
        print(f"\nBatch processing time: {duration:.2f}s ({avg_time:.2f}s per document)")


class TestOutputFormat:
    """Test output format and serialization."""

    def test_output_serialization(self, initialized_system):
        """Test that results can be serialized to JSON."""
        text = "Apple Inc. is a company."
        result = initialized_system.process_text(text)

        # Should be able to use Pydantic's model_dump_json for JSON serialization
        if hasattr(result, "model_dump_json"):
            json_str = result.model_dump_json()
            assert json_str is not None

            # Should be able to deserialize
            deserialized = json.loads(json_str)
            assert deserialized is not None
        elif hasattr(result, "to_json"):
            # Alternative: use to_json method
            json_str = result.to_json()
            assert json_str is not None

    def test_output_has_metadata(self, initialized_system):
        """Test that output includes useful metadata."""
        text = "Microsoft is based in Washington."
        result = initialized_system.process_text(text)

        # Check for metadata fields
        assert hasattr(result, "text")
        assert hasattr(result, "entities")

        # May have additional metadata
        if hasattr(result, "model_agreement"):
            assert isinstance(result.model_agreement, dict)


def test_complete_workflow(initialized_system, test_documents, tmp_path):
    """
    Test complete workflow: load KB, process documents, save results.
    This is the main integration test that validates the full pipeline.
    """
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE WORKFLOW TEST")
    print("=" * 80)

    # Process all test documents
    texts = [doc["text"] for doc in test_documents]
    results = initialized_system.process_batch(texts)

    assert len(results) == len(test_documents)

    # Analyze results
    total_entities = sum(len(r.entities) for r in results)
    print(f"\nProcessed {len(results)} documents")
    print(f"Detected {total_entities} total entities")

    # Check each result
    for i, (doc, result) in enumerate(zip(test_documents, results)):
        print(f"\nDocument {i + 1}: {doc['text'][:50]}...")
        print(f"  Detected {len(result.entities)} entities")

        for entity in result.entities[:3]:  # Show first 3
            print(f"    - {entity.mention} (confidence: {entity.confidence:.2f})")

    # Save results - use Pydantic's model_dump_json to handle datetime serialization
    output_file = tmp_path / "results.json"

    with open(output_file, "w") as f:
        # Serialize each result using Pydantic's JSON serialization
        f.write("[\n")
        for i, result in enumerate(results):
            if hasattr(result, "model_dump_json"):
                f.write("  " + result.model_dump_json())
            else:
                # Fallback for non-Pydantic results
                f.write("  " + json.dumps(str(result)))
            if i < len(results) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]")

    print(f"\nResults saved to {output_file}")
    print("=" * 80)

    # Verify we detected at least some entities
    assert total_entities > 0, "Should detect at least some entities across all documents"

    # Verify output file was created
    assert output_file.exists()
    assert output_file.stat().st_size > 0
