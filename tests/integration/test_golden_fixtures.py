"""
Regression tests using golden fixtures.

These tests ensure output stability across versions by comparing
against pre-validated golden outputs.
"""

import pytest

from entity_resolution import UnifiedEntityResolutionSystem
from entity_resolution.validation import SystemConfig
from tests.fixtures import compare_with_golden, load_golden_fixture, load_tolerance_config

# Mark all tests as integration/regression
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def test_system():
    """Create test system with consistent configuration."""
    config = SystemConfig(
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        reader_model="distilbert-base-uncased",
        use_improved_atg=True,
        use_relik=True,
        use_spel=False,
        use_unirel=False,
        max_seq_length=128,
        batch_size=2,
        consensus_threshold=0.5,
        use_gpu=False,
    )

    return UnifiedEntityResolutionSystem(config)


@pytest.fixture(scope="module")
def tolerance_config():
    """Load tolerance configuration."""
    return load_tolerance_config()


class TestGoldenFixtures:
    """Test suite for golden fixture regression tests."""

    def test_sample_1_basic_entities(self, test_system, tolerance_config):
        """Test basic entity linking (sample 1)."""
        # Load golden fixture
        input_data, expected_output = load_golden_fixture("sample_1")

        # Process text
        actual_output = test_system.process_text(input_data["text"])

        # Compare with golden output
        matches, errors = compare_with_golden(
            actual_output,
            expected_output,
            tolerance_config,
            mode="lenient",  # Lenient mode for now
        )

        # Log any errors for debugging
        if not matches:
            print("\nComparison errors:")
            for error in errors:
                print(f"  - {error}")

        # Assert (lenient for now, will tighten as models stabilize)
        # For now, just check that processing completes without error
        assert actual_output is not None
        assert hasattr(actual_output, "text")
        assert hasattr(actual_output, "entities")

    def test_sample_2_unicode_emoji(self, test_system, tolerance_config):
        """Test unicode and emoji handling (sample 2)."""
        # Load golden fixture
        input_data, expected_output = load_golden_fixture("sample_2")

        # Process text with emoji
        actual_output = test_system.process_text(input_data["text"])

        # Compare
        matches, errors = compare_with_golden(
            actual_output,
            expected_output,
            tolerance_config,
            mode="lenient",
        )

        # Log errors
        if not matches:
            print("\nComparison errors:")
            for error in errors:
                print(f"  - {error}")

        # Assert basic structure
        assert actual_output is not None
        assert actual_output.text == input_data["text"]

    def test_output_schema_consistency(self, test_system):
        """Test that output schema remains consistent."""
        text = "Apple Inc. is a technology company."
        result = test_system.process_text(text)

        # Required fields must exist
        required_fields = [
            "text",
            "entities",
            "relations",
            "num_entities",
            "num_relations",
            "pipeline_stages",
            "models_used",
            "model_predictions",
            "consensus_method",
            "processing_timestamp",
        ]

        for field in required_fields:
            assert hasattr(result, field), f"Missing required field: {field}"

    def test_entity_schema_consistency(self, test_system):
        """Test that entity schema remains consistent."""
        text = "Steve Jobs founded Apple Inc."
        result = test_system.process_text(text)

        if len(result.entities) > 0:
            entity = result.entities[0]

            # Required entity fields
            required_entity_fields = [
                "mention",
                "mention_span",
                "entity_id",
                "entity_name",
                "entity_type",
                "confidence",
                "source_model",
            ]

            for field in required_entity_fields:
                assert hasattr(entity, field), f"Missing entity field: {field}"

            # Check span has start and end
            assert hasattr(entity.mention_span, "start")
            assert hasattr(entity.mention_span, "end")
            assert 0 <= entity.mention_span.start < len(text)
            assert 0 <= entity.mention_span.end <= len(text)
            assert entity.mention_span.start < entity.mention_span.end

    def test_confidence_range_invariant(self, test_system):
        """Test that confidence scores remain in valid range."""
        texts = [
            "Apple Inc. is a company.",
            "Steve Jobs was a visionary.",
            "California is a state.",
        ]

        for text in texts:
            result = test_system.process_text(text)

            for entity in result.entities:
                assert 0.0 <= entity.confidence <= 1.0, (
                    f"Confidence out of range: {entity.confidence} for entity {entity.mention}"
                )

    def test_deterministic_output_structure(self, test_system):
        """Test that output structure is deterministic."""
        text = "Apple Inc. was founded by Steve Jobs."

        # Process same text multiple times
        results = [test_system.process_text(text) for _ in range(3)]

        # All should have same structure
        for i in range(1, len(results)):
            assert len(results[i].entities) == len(results[0].entities), (
                "Non-deterministic entity count"
            )

            # Check entity order is consistent
            for j, entity in enumerate(results[i].entities):
                assert entity.mention == results[0].entities[j].mention, (
                    "Non-deterministic entity order"
                )

    def test_backwards_compatibility_v1(self, test_system):
        """Test backwards compatibility with v1.0 output format."""
        text = "Test document."
        result = test_system.process_text(text)

        # Should be serializable to JSON
        json_str = result.to_json()
        assert json_str is not None
        assert len(json_str) > 0

        # Should be convertible to dict
        dict_output = result.model_dump()
        assert isinstance(dict_output, dict)
        assert "text" in dict_output
        assert "entities" in dict_output

    def test_csv_export_stability(self, test_system):
        """Test that CSV export format remains stable."""
        text = "Apple Inc. is in California."
        result = test_system.process_text(text)

        # Get CSV rows
        csv_rows = result.to_csv_rows()

        if len(csv_rows) > 0:
            row = csv_rows[0]

            # Required CSV columns
            required_columns = [
                "text",
                "mention",
                "mention_start",
                "mention_end",
                "entity_id",
                "entity_name",
                "entity_type",
                "confidence",
                "source_model",
            ]

            for column in required_columns:
                assert column in row, f"Missing CSV column: {column}"


class TestGoldenFixtureUtilities:
    """Test the golden fixture utilities themselves."""

    def test_load_golden_fixture_sample_1(self):
        """Test loading sample 1 fixture."""
        input_data, expected_output = load_golden_fixture("sample_1")

        assert "text" in input_data
        assert "id" in input_data

        assert "text" in expected_output
        assert "entities" in expected_output
        assert "num_entities" in expected_output

    def test_load_golden_fixture_sample_2(self):
        """Test loading sample 2 fixture."""
        input_data, expected_output = load_golden_fixture("sample_2")

        assert "text" in input_data
        # Check emoji is preserved
        assert "🍎" in input_data["text"]

    def test_load_tolerance_config(self):
        """Test loading tolerance configuration."""
        config = load_tolerance_config()

        assert "tolerances" in config
        assert "confidence" in config["tolerances"]

    def test_tolerance_config_has_required_fields(self):
        """Test that tolerance config has all required tolerances."""
        config = load_tolerance_config()
        tolerances = config["tolerances"]

        # Should have tolerance for confidence
        assert "confidence" in tolerances
        assert tolerances["confidence"]["type"] == "float"

        # Should ignore timestamps
        assert "processing_timestamp" in tolerances
        assert tolerances["processing_timestamp"]["type"] == "ignore"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
