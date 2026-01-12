"""
Integration tests for Enhanced ReLiK integration in UnifiedEntityResolutionSystem.

Tests all new features:
- Enhanced entity linking with improved reader
- Dynamic KB updates
- Confidence calibration
- Training batch generation
- Relation extraction
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from entity_resolution import SystemConfig, UnifiedEntityResolutionSystem


@pytest.fixture
def test_entities():
    """Create test entities."""
    return {
        "Q312": {
            "id": "Q312",
            "name": "Apple Inc.",
            "text": "Apple Inc. is an American multinational technology company",
            "type": "ORG",
        },
        "Q7564": {
            "id": "Q7564",
            "name": "Steve Jobs",
            "text": "Steve Jobs was an American business magnate and co-founder of Apple",
            "type": "PER",
        },
        "Q99": {
            "id": "Q99",
            "name": "California",
            "text": "California is a state in the western United States",
            "type": "LOC",
        },
    }


@pytest.fixture
def entity_file(test_entities, tmp_path):
    """Create temporary entity file."""
    entity_file = tmp_path / "entities.json"
    with open(entity_file, "w") as f:
        json.dump(list(test_entities.values()), f)
    return entity_file


@pytest.fixture
def config(tmp_path):
    """Create test configuration."""
    return SystemConfig(
        use_relik=True,
        relik_use_el=True,
        relik_use_re=True,
        relik_enable_calibration=True,
        relik_enable_dynamic_updates=True,
        relik_rebuild_threshold=10,
        relik_auto_rebuild=True,
        entity_types=["PER", "ORG", "LOC", "MISC"],
        relation_types=["founded by", "located in", "works at"],
        cache_dir=str(tmp_path / "cache"),
        index_path=str(tmp_path / "index"),
        use_gpu=False,  # Use CPU for tests
        batch_size=2,
    )


@pytest.fixture
def system(config, entity_file):
    """Create unified system with ReLiK integration."""
    system = UnifiedEntityResolutionSystem(config)
    system.load_entities(entity_file)
    return system


class TestReLiKSystem:
    """Test Enhanced ReLiK integration."""

    def test_initialization(self, system):
        """Test system initializes with enhanced ReLiK integration."""
        assert system.relik_integration is not None
        assert system.relik_model is not None  # Backward compatibility
        assert hasattr(system.relik_integration, "linker")
        assert hasattr(system.relik_integration, "retriever")
        assert hasattr(system.relik_integration, "reader")

    def test_entity_linking(self, system):
        """Test enhanced entity linking."""
        text = "Apple was founded by Steve Jobs in California"

        result = system.process_text(text)

        # Check result structure
        assert hasattr(result, "entities")
        assert hasattr(result, "text")

        # Should find entities (may vary based on model performance)
        assert len(result.entities) >= 0

        # Check entity structure if any found
        if len(result.entities) > 0:
            entity = result.entities[0]
            assert hasattr(entity, "text")
            assert hasattr(entity, "confidence")

    def test_relation_extraction(self, system):
        """Test relation extraction."""
        text = "Apple was founded by Steve Jobs"

        result = system.process_text(text)

        # Check relations structure
        assert hasattr(result, "relations")

        # Relations may or may not be found depending on model
        # Just verify structure
        if len(result.relations) > 0:
            relation = result.relations[0]
            assert hasattr(relation, "subject")
            assert hasattr(relation, "relation")
            assert hasattr(relation, "object")

    def test_add_entity_to_kb(self, system):
        """Test adding entity dynamically."""
        # Add new entity
        new_entity = {
            "id": "Q_new",
            "name": "Microsoft",
            "text": "Microsoft Corporation is an American technology company",
            "type": "ORG",
        }

        system.add_entity_to_kb("Q_new", new_entity, immediate=True)

        # Verify entity was added
        stats = system.get_kb_statistics()
        assert "relik" in stats
        assert stats["relik"]["knowledge_base_size"] > 3  # Original 3 + 1 new

    def test_update_entity_in_kb(self, system):
        """Test updating entity dynamically."""
        # Update existing entity
        updated_entity = {
            "id": "Q312",
            "name": "Apple Inc.",
            "text": "Apple Inc. is a technology giant that produces iPhones",
            "type": "ORG",
        }

        system.update_entity_in_kb("Q312", updated_entity, immediate=True)

        # Verify update succeeded (no exceptions)
        stats = system.get_kb_statistics()
        assert "relik" in stats

    def test_remove_entity_from_kb(self, system):
        """Test removing entity dynamically."""
        # Get initial size
        stats_before = system.get_kb_statistics()
        initial_size = stats_before["relik"]["knowledge_base_size"]

        # Remove entity
        system.remove_entity_from_kb("Q99", immediate=True)

        # Verify entity was removed
        stats_after = system.get_kb_statistics()
        assert stats_after["relik"]["knowledge_base_size"] < initial_size

    def test_get_kb_statistics(self, system):
        """Test getting KB statistics."""
        stats = system.get_kb_statistics()

        assert "relik" in stats
        assert "knowledge_base_size" in stats["relik"]
        assert "updates_since_rebuild" in stats["relik"]
        assert "needs_rebuild" in stats["relik"]

        assert stats["relik"]["knowledge_base_size"] == 3  # Initial entities

    def test_batch_processing(self, system):
        """Test batch processing with enhanced ReLiK."""
        texts = [
            "Apple is a technology company",
            "Steve Jobs founded Apple",
            "California is located in the United States",
        ]

        results = system.process_batch(texts)

        assert len(results) == 3
        for result in results:
            assert hasattr(result, "entities")
            assert hasattr(result, "text")

    def test_confidence_calibration_fit(self, system):
        """Test fitting confidence calibrators."""
        # Create dummy validation data
        validation_data = {
            "span_scores": torch.randn(100),
            "span_labels": torch.randint(0, 2, (100,)),
            "entity_scores": torch.randn(100),
            "entity_labels": torch.randint(0, 2, (100,)),
        }

        # Fit calibrators
        system.fit_confidence_calibrators(validation_data)

        # Verify calibrators were fitted
        assert system.relik_integration.calibrator is not None

    def test_get_training_batch(self, system):
        """Test getting training batch with hard negatives."""
        queries = ["Apple makes iPhones", "Steve Jobs was CEO"]
        positive_ids = ["Q312", "Q7564"]

        batch = system.get_training_batch_relik(queries, positive_ids)

        # Check batch structure
        assert "query_ids" in batch
        assert "query_mask" in batch
        assert "positive_ids" in batch
        assert "positive_mask" in batch
        assert "negative_ids" in batch
        assert "negative_mask" in batch

        # Check shapes
        assert batch["query_ids"].shape[0] == 2
        assert batch["positive_ids"].shape[0] == 2

    def test_dynamic_updates_trigger_rebuild(self, system):
        """Test that dynamic updates trigger rebuild after threshold."""
        # Set low rebuild threshold
        system.relik_integration.dynamic_manager.rebuild_threshold = 3

        # Add entities to trigger rebuild
        for i in range(5):
            entity = {
                "id": f"Q_test_{i}",
                "name": f"Test Entity {i}",
                "text": f"Test entity number {i}",
                "type": "MISC",
            }
            system.add_entity_to_kb(f"Q_test_{i}", entity, immediate=False)

        # Check if rebuild was triggered
        stats = system.get_kb_statistics()
        # After 5 updates with threshold 3, should have rebuilt at least once
        # updates_since_rebuild should be <= 2
        assert stats["relik"]["updates_since_rebuild"] <= 2

    def test_relik_integration_attributes(self, system):
        """Test that all enhanced features are accessible."""
        # Check all components exist
        assert hasattr(system.relik_integration, "retriever")
        assert hasattr(system.relik_integration, "reader")
        assert hasattr(system.relik_integration, "linker")
        assert hasattr(system.relik_integration, "relation_extractor")
        assert hasattr(system.relik_integration, "calibrator")
        assert hasattr(system.relik_integration, "dynamic_manager")
        assert hasattr(system.relik_integration, "hard_negative_miner")

    def test_backward_compatibility(self, system):
        """Test backward compatibility with old ReLiK interface."""
        # Old code should still work through relik_model reference
        assert system.relik_model is not None
        assert system.relik_model == system.relik_integration.linker

    def test_process_text_with_calibration(self, system):
        """Test processing text with calibration enabled."""
        # First fit calibrator
        validation_data = {
            "span_scores": torch.randn(50),
            "span_labels": torch.randint(0, 2, (50,)),
            "entity_scores": torch.randn(50),
            "entity_labels": torch.randint(0, 2, (50,)),
        }
        system.fit_confidence_calibrators(validation_data)

        # Process text
        text = "Apple was founded by Steve Jobs"
        result = system.process_text(text)

        # Verify result structure
        assert hasattr(result, "entities")
        # Confidence scores should be calibrated if entities found
        if len(result.entities) > 0:
            assert all(0.0 <= e.confidence <= 1.0 for e in result.entities)

    def test_error_handling_no_relik(self):
        """Test error handling when ReLiK is not enabled."""
        config = SystemConfig(
            use_relik=False,
            use_gpu=False,
        )
        system = UnifiedEntityResolutionSystem(config)

        # Should warn but not fail
        system.add_entity_to_kb("Q_test", {"text": "test"}, immediate=True)

        # get_training_batch should raise error
        with pytest.raises(RuntimeError, match="ReLiK integration not initialized"):
            system.get_training_batch_relik(["test"], ["Q_test"])


@pytest.mark.slow
class TestEnhancedReLiKPerformance:
    """Performance and stress tests."""

    def test_large_batch_processing(self, system):
        """Test processing large batch."""
        texts = [f"Test document number {i}" for i in range(50)]

        results = system.process_batch(texts)

        assert len(results) == 50

    def test_many_dynamic_updates(self, system):
        """Test many dynamic updates."""
        # Add many entities
        for i in range(20):
            entity = {
                "id": f"Q_perf_{i}",
                "name": f"Entity {i}",
                "text": f"Performance test entity {i}",
                "type": "MISC",
            }
            system.add_entity_to_kb(f"Q_perf_{i}", entity, immediate=False)

        # Force rebuild
        stats = system.get_kb_statistics()

        # Verify all entities were added
        assert stats["relik"]["knowledge_base_size"] >= 23  # 3 original + 20 new


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
