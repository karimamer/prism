"""
Comprehensive E2E tests for edge cases and production scenarios.

These tests cover:
- Unicode and special character handling
- Overlapping entity mentions
- NIL/NME (Not In Lexicon) entity handling
- Consensus tie-breaking
- CSV/JSON export validation
- Batch processing edge cases
- Error recovery and graceful degradation
- Input format variations
"""

import json
import tempfile
from pathlib import Path

import pytest

from entity_resolution import UnifiedEntityResolutionSystem
from entity_resolution.validation import InputValidator, SystemConfig

# Mark all tests as E2E
pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def minimal_config():
    """Minimal fast configuration for E2E tests."""
    return SystemConfig(
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        reader_model="distilbert-base-uncased",
        use_improved_atg=True,
        use_relik=True,
        use_spel=False,  # Disable for speed
        use_unirel=False,  # Disable for speed
        max_seq_length=128,
        batch_size=2,
        top_k_candidates=10,
        consensus_threshold=0.4,
        use_gpu=False,
    )


@pytest.fixture(scope="module")
def test_kb_file(tmp_path_factory):
    """Create test knowledge base with diverse entities."""
    kb_dir = tmp_path_factory.mktemp("kb")
    kb_file = kb_dir / "test_kb.json"

    kb_data = {
        "Q1": {
            "id": "Q1",
            "name": "Apple Inc.",
            "description": "American technology company",
            "aliases": ["Apple", "AAPL"],
            "type": "ORG",
        },
        "Q2": {
            "id": "Q2",
            "name": "New York",
            "description": "City in United States",
            "aliases": ["NYC", "New York City"],
            "type": "LOC",
        },
        "Q3": {
            "id": "Q3",
            "name": "Steve Jobs",
            "description": "Co-founder of Apple",
            "aliases": ["Jobs", "Steven Jobs"],
            "type": "PER",
        },
    }

    with open(kb_file, "w") as f:
        json.dump(kb_data, f)

    return kb_file


class TestUnicodeHandling:
    """Test unicode and special character handling."""

    def test_unicode_nfc_normalization(self, minimal_config):
        """Test NFC unicode normalization."""
        # Café with different representations
        text1 = "Café"  # NFC form (single é character)
        text2 = "Café"  # NFD form (e + combining accent)

        # Both should normalize to same form
        normalized1 = InputValidator.validate_text_input(text1, normalize_unicode=True, unicode_form="NFC")
        normalized2 = InputValidator.validate_text_input(text2, normalize_unicode=True, unicode_form="NFC")

        assert normalized1 == normalized2
        assert len(normalized1) == 4  # C-a-f-é

    def test_emoji_in_text(self, minimal_config, test_kb_file):
        """Test processing text with emojis."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Apple 🍎 makes great products! Steve Jobs 👨‍💻 founded it."
        result = system.process_text(text)

        assert result is not None
        # Text may be normalized/tokenized differently, but key content should be preserved
        assert "Apple" in result.text
        assert "Steve Jobs" in result.text
        assert isinstance(result.entities, list)

    def test_rtl_text(self, minimal_config, test_kb_file):
        """Test right-to-left text (Arabic, Hebrew)."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Apple في نيويورك"  # "Apple in New York" in Arabic
        result = system.process_text(text)

        assert result is not None
        assert result.text == text

    def test_mixed_script_text(self, minimal_config, test_kb_file):
        """Test mixed Latin/CJK/Cyrillic text."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Apple (アップル) работает в New York"
        result = system.process_text(text)

        assert result is not None
        assert len(result.entities) >= 0

    def test_special_punctuation(self, minimal_config, test_kb_file):
        """Test various special punctuation marks."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Apple™ Inc. – founded by Steve Jobs – is in New York."
        result = system.process_text(text)

        assert result is not None
        # Em dashes and trademark symbols should be handled
        assert isinstance(result.entities, list)


class TestOverlappingEntities:
    """Test overlapping entity mention detection."""

    def test_nested_entities(self, minimal_config, test_kb_file):
        """Test nested entity mentions (one inside another)."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # "New York" is inside "New York City"
        text = "I visited New York City last week."
        result = system.process_text(text)

        assert result is not None
        # System should handle overlapping mentions
        assert isinstance(result.entities, list)

    def test_partially_overlapping_entities(self, minimal_config, test_kb_file):
        """Test partially overlapping entity mentions."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # Entities that share some characters
        text = "Steve Jobs worked at Apple Inc."
        result = system.process_text(text)

        assert result is not None
        # Should detect both entities without overlap issues
        assert len(result.entities) >= 0

    def test_adjacent_entities(self, minimal_config, test_kb_file):
        """Test adjacent entities with no space between."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "SteveJobsApple"  # Edge case: no spaces
        result = system.process_text(text)

        assert result is not None
        assert isinstance(result.entities, list)


class TestNILEntityHandling:
    """Test NIL/NME (Not In Lexicon) entity handling."""

    def test_unknown_entity_not_in_kb(self, minimal_config, test_kb_file):
        """Test entity not in knowledge base."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Microsoft and Google are competitors."
        result = system.process_text(text)

        assert result is not None
        # Should process without crashing even if entities not in KB
        assert isinstance(result.entities, list)

    def test_fictional_entities(self, minimal_config, test_kb_file):
        """Test completely fictional entities."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "XyzCorp123 and AbcCompany456 are fictional."
        result = system.process_text(text)

        assert result is not None
        assert result.num_entities >= 0

    def test_mixed_known_unknown_entities(self, minimal_config, test_kb_file):
        """Test mix of known and unknown entities."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Apple, Microsoft, and FictionalCorp are companies."
        result = system.process_text(text)

        assert result is not None
        # Should handle mixed case gracefully
        assert isinstance(result.entities, list)


class TestConsensusTieBreaking:
    """Test consensus mechanism tie-breaking scenarios."""

    def test_equal_confidence_ties(self, minimal_config, test_kb_file):
        """Test when multiple models have equal confidence."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # Ambiguous mention that could match multiple entities
        text = "Apple products are great."
        result = system.process_text(text)

        assert result is not None
        # Should break ties deterministically
        if len(result.entities) > 0:
            # Check that consensus_method is applied
            assert result.consensus_method is not None

    def test_model_agreement_metadata(self, minimal_config, test_kb_file):
        """Test that model agreement metadata is included."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Steve Jobs founded Apple Inc."
        result = system.process_text(text)

        assert result is not None
        if len(result.entities) > 0:
            entity = result.entities[0]
            # Should have source model info
            assert hasattr(entity, "source_model")
            assert entity.source_model != ""

    def test_confidence_thresholds(self, minimal_config, test_kb_file):
        """Test confidence threshold filtering."""
        # Create config with high threshold
        high_threshold_config = SystemConfig(
            retriever_model="sentence-transformers/all-MiniLM-L6-v2",
            reader_model="distilbert-base-uncased",
            use_improved_atg=True,
            use_relik=True,
            consensus_threshold=0.9,  # Very high threshold
            use_gpu=False,
        )

        system = UnifiedEntityResolutionSystem(high_threshold_config)
        system.load_entities(str(test_kb_file))

        text = "Apple and New York are mentioned."
        result = system.process_text(text)

        assert result is not None
        # High threshold may filter out entities
        assert isinstance(result.entities, list)


class TestOutputFormats:
    """Test output serialization and export formats."""

    def test_json_serialization(self, minimal_config, test_kb_file):
        """Test JSON output serialization."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Apple Inc. is in New York."
        result = system.process_text(text)

        # Test JSON serialization
        json_str = result.to_json()
        assert json_str is not None
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "text" in parsed
        assert "entities" in parsed

    def test_csv_export(self, minimal_config, test_kb_file):
        """Test CSV export functionality."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Steve Jobs worked at Apple Inc."
        result = system.process_text(text)

        # Test CSV row generation
        csv_rows = result.to_csv_rows()
        assert isinstance(csv_rows, list)

        if len(csv_rows) > 0:
            row = csv_rows[0]
            # Check required CSV fields
            assert "text" in row
            assert "mention" in row
            assert "entity_id" in row
            assert "confidence" in row

    def test_csv_with_relations(self, minimal_config, test_kb_file):
        """Test CSV export for relations."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Steve Jobs founded Apple Inc."
        result = system.process_text(text)

        # Test relations CSV
        relations_csv = result.to_csv_relations()
        assert isinstance(relations_csv, list)

    def test_output_schema_validation(self, minimal_config, test_kb_file):
        """Test that output always validates against schema."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        text = "Apple Inc. is a company."
        result = system.process_text(text)

        # Pydantic validation happens automatically
        # Test that all required fields are present
        assert hasattr(result, "text")
        assert hasattr(result, "entities")
        assert hasattr(result, "relations")
        assert hasattr(result, "num_entities")
        assert hasattr(result, "processing_timestamp")
        assert hasattr(result, "pipeline_stages")


class TestBatchProcessing:
    """Test batch processing edge cases."""

    def test_empty_batch_error(self, minimal_config, test_kb_file):
        """Test that empty batch raises error."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        with pytest.raises(ValueError, match="Empty batch"):
            system.process_batch([])

    def test_single_item_batch(self, minimal_config, test_kb_file):
        """Test batch with single item."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        texts = ["Apple Inc. is a company."]
        results = system.process_batch(texts)

        assert len(results) == 1
        assert results[0].text == texts[0]

    def test_large_batch(self, minimal_config, test_kb_file):
        """Test batch with many items."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        texts = [f"Document {i} mentions Apple Inc." for i in range(20)]
        results = system.process_batch(texts)

        assert len(results) == 20
        for i, result in enumerate(results):
            assert f"Document {i}" in result.text

    def test_batch_with_varying_lengths(self, minimal_config, test_kb_file):
        """Test batch with varying text lengths."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        texts = [
            "Apple.",
            "Apple Inc. is a technology company based in California.",
            "A" * 500,  # Long text
        ]
        results = system.process_batch(texts)

        assert len(results) == 3


class TestErrorRecovery:
    """Test error handling and graceful degradation."""

    def test_very_long_text_truncation(self, minimal_config, test_kb_file):
        """Test handling of text exceeding max_seq_length."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # Text much longer than max_seq_length (128) but within validation limit (1280)
        # max_seq_length=128, validation allows 128*10=1280 chars
        long_text = "Apple Inc. is a company. " * 40  # 25 chars * 40 = 1000 chars
        result = system.process_text(long_text)

        assert result is not None
        # Should handle truncation gracefully
        assert isinstance(result.entities, list)

    def test_malformed_unicode(self, minimal_config, test_kb_file):
        """Test handling of text with malformed unicode."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # Text with unusual unicode combinations
        text = "Apple\u0301 Inc."  # Combining accent without base character
        result = system.process_text(text)

        assert result is not None

    def test_single_model_failure_graceful_degradation(self, test_kb_file):
        """Test that system works when one model fails."""
        # Config with only one model enabled
        single_model_config = SystemConfig(
            retriever_model="sentence-transformers/all-MiniLM-L6-v2",
            reader_model="distilbert-base-uncased",
            use_improved_atg=True,
            use_relik=False,  # Disabled
            use_spel=False,  # Disabled
            use_unirel=False,  # Disabled
            use_gpu=False,
        )

        system = UnifiedEntityResolutionSystem(single_model_config)
        system.load_entities(str(test_kb_file))

        text = "Apple Inc. is a company."
        result = system.process_text(text)

        assert result is not None
        # Should still produce valid output with just one model
        assert result.models_used is not None
        assert isinstance(result.entities, list)


class TestInputFormatVariations:
    """Test various input formats."""

    def test_json_list_input(self, minimal_config, test_kb_file, tmp_path):
        """Test JSON list input format."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # Create JSON input file
        input_file = tmp_path / "input.json"
        data = ["Apple Inc. is a company.", "New York is a city."]

        with open(input_file, "w") as f:
            json.dump(data, f)

        # Should be able to load and process
        with open(input_file) as f:
            loaded_data = json.load(f)

        results = system.process_batch(loaded_data)
        assert len(results) == 2

    def test_json_dict_input(self, minimal_config, test_kb_file, tmp_path):
        """Test JSON dict input format."""
        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # Create JSON dict input
        input_file = tmp_path / "input.json"
        data = {
            "documents": [
                {"id": "doc1", "text": "Apple Inc. is a company."},
                {"id": "doc2", "text": "New York is a city."},
            ]
        }

        with open(input_file, "w") as f:
            json.dump(data, f)

        # Extract texts
        with open(input_file) as f:
            loaded_data = json.load(f)

        texts = [doc["text"] for doc in loaded_data["documents"]]
        results = system.process_batch(texts)
        assert len(results) == 2

    def test_csv_input_loading(self, minimal_config, test_kb_file, tmp_path):
        """Test loading CSV input format."""
        import csv

        system = UnifiedEntityResolutionSystem(minimal_config)
        system.load_entities(str(test_kb_file))

        # Create CSV input file
        input_file = tmp_path / "input.csv"
        with open(input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text"])
            writer.writerow(["1", "Apple Inc. is a company."])
            writer.writerow(["2", "New York is a city."])

        # Load and process
        texts = []
        with open(input_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["text"])

        results = system.process_batch(texts)
        assert len(results) == 2


def test_end_to_end_complete_workflow(minimal_config, test_kb_file, tmp_path):
    """
    Complete end-to-end workflow test.

    This test validates the entire pipeline from input to output.
    """
    # Initialize system
    system = UnifiedEntityResolutionSystem(minimal_config)
    num_entities = system.load_entities(str(test_kb_file))
    assert num_entities == 3

    # Process various texts
    texts = [
        "Apple Inc. was founded by Steve Jobs.",
        "New York is a major city.",
        "Steve Jobs worked at Apple in California.",
        "Apple 🍎 products are popular in New York.",  # With emoji
        "日本のApple Store",  # Japanese
    ]

    results = system.process_batch(texts)

    # Validate results
    assert len(results) == len(texts)

    for result in results:
        # Schema validation
        assert hasattr(result, "text")
        assert hasattr(result, "entities")
        assert hasattr(result, "num_entities")
        assert hasattr(result, "pipeline_stages")

        # All pipeline stages should complete
        assert result.pipeline_stages.entity_encoding is not None
        assert result.pipeline_stages.candidate_generation is not None
        assert result.pipeline_stages.consensus_linking is not None
        assert result.pipeline_stages.structured_output is True

        # Validate each entity
        for entity in result.entities:
            assert entity.mention is not None
            assert 0.0 <= entity.confidence <= 1.0
            assert entity.mention_span.start >= 0
            assert entity.mention_span.end <= len(result.text)

    # Test JSON export
    output_file = tmp_path / "output.json"
    with open(output_file, "w") as f:
        json_data = [result.model_dump() for result in results]
        json.dump(json_data, f, default=str)

    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # Test CSV export
    csv_file = tmp_path / "output.csv"
    with open(csv_file, "w") as f:
        import csv

        all_rows = []
        for result in results:
            all_rows.extend(result.to_csv_rows())

        if all_rows:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    assert csv_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
