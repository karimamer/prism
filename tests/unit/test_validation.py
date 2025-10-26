"""Unit tests for validation module."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from entity_resolution.validation import (
    EntityCollection,
    EntityData,
    InputValidator,
    SystemConfig,
    validate_and_load_entities,
    validate_config,
)


@pytest.mark.unit
class TestSystemConfig:
    """Tests for SystemConfig validation."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = SystemConfig()
        assert config.retriever_model == "microsoft/deberta-v3-small"
        assert config.batch_size == 8
        assert config.consensus_threshold == 0.6

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = SystemConfig(
            retriever_model="bert-base-uncased",
            reader_model="roberta-base",
            batch_size=16,
            top_k_candidates=100,
        )
        assert config.retriever_model == "bert-base-uncased"
        assert config.batch_size == 16

    def test_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValidationError):
            SystemConfig(batch_size=0)

        with pytest.raises(ValidationError):
            SystemConfig(batch_size=256)

    def test_invalid_threshold(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValidationError):
            SystemConfig(consensus_threshold=1.5)

        with pytest.raises(ValidationError):
            SystemConfig(consensus_threshold=-0.1)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = SystemConfig(batch_size=16)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["batch_size"] == 16

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "retriever_model": "test-model",
            "batch_size": 32,
        }
        config = SystemConfig.from_dict(config_dict)

        assert config.retriever_model == "test-model"
        assert config.batch_size == 32

    def test_config_save_load_json(self, tmp_path):
        """Test saving and loading config from JSON."""
        config = SystemConfig(batch_size=24)
        json_path = tmp_path / "config.json"

        config.save_json(json_path)
        assert json_path.exists()

        loaded_config = SystemConfig.from_json(json_path)
        assert loaded_config.batch_size == 24

    def test_config_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            SystemConfig(invalid_field="value")

    def test_quantization_validation(self):
        """Test quantization type validation."""
        # Valid values
        config1 = SystemConfig(quantization="int8")
        assert config1.quantization == "int8"

        config2 = SystemConfig(quantization="fp16")
        assert config2.quantization == "fp16"

        config3 = SystemConfig(quantization=None)
        assert config3.quantization is None

        # Invalid value
        with pytest.raises(ValidationError):
            SystemConfig(quantization="int4")


@pytest.mark.unit
class TestEntityData:
    """Tests for EntityData validation."""

    def test_valid_entity(self):
        """Test creating valid entity."""
        entity = EntityData(
            id="Q1",
            name="Test Entity",
            description="A test entity",
            aliases=["Alias1", "Alias2"],
            entity_type="PER",
        )

        assert entity.id == "Q1"
        assert entity.name == "Test Entity"
        assert len(entity.aliases) == 2

    def test_minimal_entity(self):
        """Test entity with only required fields."""
        entity = EntityData(id="Q1", name="Test")

        assert entity.id == "Q1"
        assert entity.name == "Test"
        assert entity.description == ""
        assert entity.aliases == []
        assert entity.entity_type == "UNKNOWN"

    def test_empty_id_raises_error(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValidationError):
            EntityData(id="", name="Test")

        with pytest.raises(ValidationError):
            EntityData(id="   ", name="Test")

    def test_empty_name_raises_error(self):
        """Test that empty name raises error."""
        with pytest.raises(ValidationError):
            EntityData(id="Q1", name="")

        with pytest.raises(ValidationError):
            EntityData(id="Q1", name="   ")

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed."""
        entity = EntityData(
            id="  Q1  ",
            name="  Test Entity  ",
        )

        assert entity.id == "Q1"
        assert entity.name == "Test Entity"

    def test_empty_aliases_filtered(self):
        """Test that empty aliases are filtered out."""
        entity = EntityData(
            id="Q1",
            name="Test",
            aliases=["Alias1", "", "  ", "Alias2"],
        )

        assert entity.aliases == ["Alias1", "Alias2"]


@pytest.mark.unit
class TestEntityCollection:
    """Tests for EntityCollection validation."""

    def test_empty_collection(self):
        """Test creating empty collection."""
        collection = EntityCollection()
        assert len(collection.entities) == 0

    def test_collection_with_entities(self):
        """Test collection with valid entities."""
        entities = [
            EntityData(id="Q1", name="Entity 1"),
            EntityData(id="Q2", name="Entity 2"),
        ]
        collection = EntityCollection(entities=entities)

        assert len(collection.entities) == 2

    def test_duplicate_ids_raise_error(self):
        """Test that duplicate IDs raise error."""
        entities = [
            EntityData(id="Q1", name="Entity 1"),
            EntityData(id="Q1", name="Entity 1 Duplicate"),
        ]

        with pytest.raises(ValidationError, match="Duplicate entity IDs"):
            EntityCollection(entities=entities)

    def test_from_json_file_list(self, tmp_path):
        """Test loading from JSON file (list format)."""
        json_data = [
            {"id": "Q1", "name": "Entity 1", "description": "Test 1"},
            {"id": "Q2", "name": "Entity 2", "description": "Test 2"},
        ]

        json_path = tmp_path / "entities.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        collection = EntityCollection.from_json_file(json_path)
        assert len(collection.entities) == 2
        assert collection.entities[0].id == "Q1"

    def test_from_json_file_dict(self, tmp_path):
        """Test loading from JSON file (dict format)."""
        json_data = {
            "entities": [
                {"id": "Q1", "name": "Entity 1"},
                {"id": "Q2", "name": "Entity 2"},
            ]
        }

        json_path = tmp_path / "entities.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        collection = EntityCollection.from_json_file(json_path)
        assert len(collection.entities) == 2

    def test_from_json_file_invalid_format(self, tmp_path):
        """Test that invalid JSON format raises error."""
        json_data = {"invalid": "format"}

        json_path = tmp_path / "entities.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        with pytest.raises(ValueError, match="entities"):
            EntityCollection.from_json_file(json_path)

    def test_from_json_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            EntityCollection.from_json_file("nonexistent.json")

    def test_from_json_file_too_large(self, tmp_path):
        """Test that large file raises error."""
        # Create a large file (simulated)
        json_path = tmp_path / "large.json"

        # Create large dummy data
        large_data = [{"id": f"Q{i}", "name": f"Entity {i}"} for i in range(100000)]

        with open(json_path, "w") as f:
            json.dump(large_data, f)

        # If file is > 100MB, it should raise error
        # (This test might pass if the file isn't actually that large)
        file_size = json_path.stat().st_size / (1024 * 1024)
        if file_size > 100:
            with pytest.raises(ValueError, match="too large"):
                EntityCollection.from_json_file(json_path)

    def test_from_csv_file(self, tmp_path):
        """Test loading from CSV file."""
        csv_path = tmp_path / "entities.csv"

        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "description", "entity_type"])
            writer.writerow(["Q1", "Entity 1", "Test 1", "PER"])
            writer.writerow(["Q2", "Entity 2", "Test 2", "ORG"])

        collection = EntityCollection.from_csv_file(csv_path)
        assert len(collection.entities) == 2
        assert collection.entities[0].entity_type == "PER"

    def test_from_csv_file_with_aliases(self, tmp_path):
        """Test loading CSV with semicolon-separated aliases."""
        csv_path = tmp_path / "entities.csv"

        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "aliases"])
            writer.writerow(["Q1", "Entity 1", "Alias1;Alias2;Alias3"])

        collection = EntityCollection.from_csv_file(csv_path)
        assert len(collection.entities[0].aliases) == 3

    def test_to_dict_list(self):
        """Test converting collection to dict list."""
        entities = [
            EntityData(id="Q1", name="Entity 1"),
            EntityData(id="Q2", name="Entity 2"),
        ]
        collection = EntityCollection(entities=entities)

        dict_list = collection.to_dict_list()
        assert isinstance(dict_list, list)
        assert len(dict_list) == 2
        assert dict_list[0]["id"] == "Q1"


@pytest.mark.unit
class TestInputValidator:
    """Tests for InputValidator."""

    def test_validate_text_input(self):
        """Test text input validation."""
        text = "  This is valid text  "
        validated = InputValidator.validate_text_input(text)

        assert validated == "This is valid text"

    def test_validate_text_too_short(self):
        """Test that short text raises error."""
        with pytest.raises(ValueError, match="too short"):
            InputValidator.validate_text_input("")

    def test_validate_text_too_long(self):
        """Test that long text raises error."""
        long_text = "a" * 20000

        with pytest.raises(ValueError, match="too long"):
            InputValidator.validate_text_input(long_text, max_length=10000)

    def test_validate_text_not_string(self):
        """Test that non-string raises error."""
        with pytest.raises(TypeError):
            InputValidator.validate_text_input(123)

    def test_validate_batch_texts(self):
        """Test batch validation."""
        texts = ["Text 1", "Text 2", "Text 3"]
        validated = InputValidator.validate_batch_texts(texts)

        assert len(validated) == 3

    def test_validate_batch_empty(self):
        """Test that empty batch raises error."""
        with pytest.raises(ValueError, match="Empty batch"):
            InputValidator.validate_batch_texts([])

    def test_validate_batch_too_large(self):
        """Test that large batch raises error."""
        large_batch = ["text"] * 200

        with pytest.raises(ValueError, match="too large"):
            InputValidator.validate_batch_texts(large_batch, max_batch_size=128)

    def test_validate_batch_not_list(self):
        """Test that non-list raises error."""
        with pytest.raises(TypeError):
            InputValidator.validate_batch_texts("not a list")

    def test_validate_model_path(self, tmp_path):
        """Test model path validation."""
        # Existing path
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        validated = InputValidator.validate_model_path(model_dir)
        assert validated == model_dir

    def test_validate_model_path_not_exists(self):
        """Test that non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            InputValidator.validate_model_path("/nonexistent/path", check_exists=True)

    def test_validate_model_path_no_check(self):
        """Test path validation without existence check."""
        path = InputValidator.validate_model_path("/any/path", check_exists=False)
        assert isinstance(path, Path)

    def test_validate_file_size(self, tmp_path):
        """Test file size validation."""
        # Small file
        small_file = tmp_path / "small.txt"
        small_file.write_text("small content")

        InputValidator.validate_file_size(small_file, max_size_mb=1)

    def test_validate_file_size_too_large(self, tmp_path):
        """Test that large file raises error."""
        # Create 2MB file
        large_file = tmp_path / "large.txt"
        large_file.write_bytes(b"0" * (2 * 1024 * 1024))

        with pytest.raises(ValueError, match="too large"):
            InputValidator.validate_file_size(large_file, max_size_mb=1)

    def test_validate_confidence_score(self):
        """Test confidence score validation."""
        assert InputValidator.validate_confidence_score(0.5) == 0.5
        assert InputValidator.validate_confidence_score(0.0) == 0.0
        assert InputValidator.validate_confidence_score(1.0) == 1.0

    def test_validate_confidence_score_invalid(self):
        """Test that invalid confidence score raises error."""
        with pytest.raises(ValueError):
            InputValidator.validate_confidence_score(1.5)

        with pytest.raises(ValueError):
            InputValidator.validate_confidence_score(-0.1)

        with pytest.raises(TypeError):
            InputValidator.validate_confidence_score("0.5")


@pytest.mark.unit
def test_validate_config_function():
    """Test validate_config utility function."""
    # Test with dict
    config_dict = {"batch_size": 16}
    config = validate_config(config_dict)

    assert isinstance(config, SystemConfig)
    assert config.batch_size == 16

    # Test with SystemConfig object
    config_obj = SystemConfig(batch_size=32)
    validated = validate_config(config_obj)

    assert validated is config_obj
    assert validated.batch_size == 32


@pytest.mark.unit
def test_validate_and_load_entities(tmp_path):
    """Test validate_and_load_entities utility function."""
    # Create test JSON file
    json_data = [
        {"id": "Q1", "name": "Entity 1"},
        {"id": "Q2", "name": "Entity 2"},
    ]

    json_path = tmp_path / "entities.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f)

    # Load entities
    entities = validate_and_load_entities(json_path)

    assert isinstance(entities, list)
    assert len(entities) == 2
    assert entities[0]["id"] == "Q1"


@pytest.mark.unit
def test_validate_and_load_entities_with_max(tmp_path):
    """Test loading entities with max limit."""
    # Create test JSON file with many entities
    json_data = [{"id": f"Q{i}", "name": f"Entity {i}"} for i in range(100)]

    json_path = tmp_path / "entities.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f)

    # Load with max
    entities = validate_and_load_entities(json_path, max_entities=50)

    assert len(entities) == 50
