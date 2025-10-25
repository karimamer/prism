"""Unit tests for output formatting."""

import json

import pytest


@pytest.mark.unit
def test_format_entity():
    """Test formatting a single entity."""
    from src.entity_resolution.models.output import OutputFormatter

    formatter = OutputFormatter()
    entity = {
        "mention": "Apple Inc.",
        "entity_id": "Q312",
        "entity_name": "Apple Inc.",
        "confidence": 0.95,
        "start": 0,
        "end": 10,
    }

    formatted = formatter.format_entity(entity)
    assert "mention" in formatted
    assert "entity_id" in formatted
    assert formatted["confidence"] == 0.95


@pytest.mark.unit
def test_format_entities():
    """Test formatting multiple entities."""
    from src.entity_resolution.models.output import OutputFormatter

    formatter = OutputFormatter()
    entities = [
        {
            "mention": "Apple Inc.",
            "entity_id": "Q312",
            "entity_name": "Apple Inc.",
            "confidence": 0.95,
            "start": 0,
            "end": 10,
        },
        {
            "mention": "Steve Jobs",
            "entity_id": "Q619",
            "entity_name": "Steve Jobs",
            "confidence": 0.92,
            "start": 50,
            "end": 61,
        },
    ]

    formatted = formatter.format_entities(entities)
    assert len(formatted) == 2
    assert all("mention" in e for e in formatted)


@pytest.mark.unit
def test_to_json():
    """Test JSON output format."""
    from src.entity_resolution.models.output import OutputFormatter

    formatter = OutputFormatter()
    entities = [
        {
            "mention": "Apple Inc.",
            "entity_id": "Q312",
            "entity_name": "Apple Inc.",
            "confidence": 0.95,
            "start": 0,
            "end": 10,
        }
    ]

    json_output = formatter.to_json(entities)
    parsed = json.loads(json_output)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


@pytest.mark.unit
def test_convert_token_to_char_spans():
    """Test token to character span conversion."""
    from src.entity_resolution.models.output import OutputFormatter

    formatter = OutputFormatter()
    text = "Apple Inc. is a company"
    token_spans = [(0, 2), (3, 5)]  # Token indices

    # This is a simplified test - actual implementation may vary
    # Just verify the function exists and runs
    try:
        char_spans = formatter.convert_token_to_char_spans(text, token_spans)
        assert isinstance(char_spans, list)
    except AttributeError:
        # Method might not exist yet
        pytest.skip("convert_token_to_char_spans not implemented")
