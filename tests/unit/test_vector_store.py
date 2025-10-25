"""Unit tests for vector store functionality."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
def test_load_entities_from_json(temp_entity_file, sample_entities):
    """Test loading entities from JSON file."""
    from src.entity_resolution.database.vector_store import VectorStore

    store = VectorStore()
    entities = store.load_entities(str(temp_entity_file))

    assert len(entities) == len(sample_entities)
    assert entities[0]["id"] == sample_entities[0]["id"]
    assert entities[0]["name"] == sample_entities[0]["name"]


@pytest.mark.unit
def test_vector_store_initialization():
    """Test VectorStore initialization."""
    from src.entity_resolution.database.vector_store import VectorStore

    store = VectorStore(dimension=768)
    assert store.dimension == 768
    assert store.entities == []


@pytest.mark.unit
def test_add_entity():
    """Test adding a single entity."""
    from src.entity_resolution.database.vector_store import VectorStore

    store = VectorStore(dimension=768)
    entity = {
        "id": "Q1",
        "name": "Test Entity",
        "description": "A test entity",
    }

    # Mock the embedding
    embedding = np.random.rand(768).astype(np.float32)

    store.add_entity(entity, embedding)
    assert len(store.entities) == 1
    assert store.entities[0]["id"] == "Q1"


@pytest.mark.unit
def test_get_entity():
    """Test retrieving entity by ID."""
    from src.entity_resolution.database.vector_store import VectorStore

    store = VectorStore(dimension=768)
    entity = {
        "id": "Q1",
        "name": "Test Entity",
        "description": "A test entity",
    }

    embedding = np.random.rand(768).astype(np.float32)
    store.add_entity(entity, embedding)

    retrieved = store.get_entity("Q1")
    assert retrieved is not None
    assert retrieved["id"] == "Q1"
    assert retrieved["name"] == "Test Entity"


@pytest.mark.unit
def test_get_nonexistent_entity():
    """Test retrieving non-existent entity."""
    from src.entity_resolution.database.vector_store import VectorStore

    store = VectorStore(dimension=768)
    retrieved = store.get_entity("Q999")
    assert retrieved is None
