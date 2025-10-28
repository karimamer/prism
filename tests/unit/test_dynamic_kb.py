"""
Tests for dynamic knowledge base updates.

Tests the EntityKnowledgeBase with dynamic add/update/remove operations.
"""

import tempfile

import numpy as np

from src.entity_resolution.database.vector_store import EntityKnowledgeBase


def test_basic_add_entity():
    """Test adding a single entity."""
    print("\n=== Testing Basic Add Entity ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create knowledge base
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Add entity
        entity_id = "Q1"
        entity_data = {"name": "Test Entity", "text": "This is a test entity"}
        entity_embedding = np.random.randn(128).astype(np.float32)

        kb.add_entity(entity_id, entity_data, entity_embedding)

        # Verify
        assert entity_id in kb.entities
        assert kb.entities[entity_id] == entity_data
        assert len(kb.entity_ids) == 1

        print(f"✓ Added entity {entity_id}")
        print(f"  Total entities: {len(kb.entities)}")

    print("✓ Basic add entity test passed")


def test_update_entity():
    """Test updating an existing entity."""
    print("\n=== Testing Update Entity ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Add initial entity
        entity_id = "Q1"
        entity_data = {"name": "Original", "text": "Original text"}
        entity_embedding = np.random.randn(128).astype(np.float32)

        kb.add_entity(entity_id, entity_data, entity_embedding)

        print(f"✓ Added entity: {kb.entities[entity_id]['name']}")

        # Update entity
        new_data = {"name": "Updated", "text": "Updated text"}
        new_embedding = np.random.randn(128).astype(np.float32)

        kb.update_entity(entity_id, new_data, new_embedding)

        # Verify
        assert kb.entities[entity_id]["name"] == "Updated"
        assert kb.entities[entity_id]["text"] == "Updated text"

        print(f"✓ Updated entity: {kb.entities[entity_id]['name']}")

    print("✓ Update entity test passed")


def test_remove_entity():
    """Test removing an entity."""
    print("\n=== Testing Remove Entity ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Add entities
        for i in range(3):
            entity_id = f"Q{i + 1}"
            entity_data = {"name": f"Entity {i + 1}"}
            entity_embedding = np.random.randn(128).astype(np.float32)
            kb.add_entity(entity_id, entity_data, entity_embedding)

        print(f"✓ Added {len(kb.entities)} entities")

        # Remove one
        kb.remove_entity("Q2")

        # Verify
        assert "Q2" not in kb.entities
        assert len(kb.entities) == 2
        assert "Q2" not in kb.entity_ids

        print(f"✓ Removed entity Q2")
        print(f"  Remaining entities: {len(kb.entities)}")

    print("✓ Remove entity test passed")


def test_batch_operations():
    """Test batch add and remove operations."""
    print("\n=== Testing Batch Operations ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Batch add
        entities = {f"Q{i}": {"name": f"Entity {i}", "text": f"Text {i}"} for i in range(1, 11)}

        embeddings = {f"Q{i}": np.random.randn(128).astype(np.float32) for i in range(1, 11)}

        kb.batch_add_entities(entities, embeddings)

        print(f"✓ Batch added {len(entities)} entities")
        assert len(kb.entities) == 10

        # Batch remove
        to_remove = ["Q2", "Q4", "Q6"]
        kb.batch_remove_entities(to_remove)

        print(f"✓ Batch removed {len(to_remove)} entities")
        assert len(kb.entities) == 7

        for entity_id in to_remove:
            assert entity_id not in kb.entities

    print("✓ Batch operations test passed")


def test_search_after_updates():
    """Test that search still works after updates."""
    print("\n=== Testing Search After Updates ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Add initial entities
        embeddings_dict = {}
        for i in range(5):
            entity_id = f"Q{i + 1}"
            entity_data = {"name": f"Entity {i + 1}"}
            entity_embedding = np.random.randn(128).astype(np.float32)
            embeddings_dict[entity_id] = entity_embedding
            kb.add_entity(entity_id, entity_data, entity_embedding)

        print(f"✓ Added {len(kb.entities)} entities")

        # Perform search
        query = np.random.randn(128).astype(np.float32)
        results = kb.search(query, k=3)

        print(f"✓ Found {len(results)} results")
        for entity_id, score in results:
            print(f"  - {entity_id}: {score:.3f}")

        assert len(results) == 3

        # Add more entities
        for i in range(5, 8):
            entity_id = f"Q{i + 1}"
            entity_data = {"name": f"Entity {i + 1}"}
            entity_embedding = np.random.randn(128).astype(np.float32)
            kb.add_entity(entity_id, entity_data, entity_embedding)

        print(f"✓ Added 3 more entities (total: {len(kb.entities)})")

        # Search again
        results2 = kb.search(query, k=5)

        print(f"✓ Found {len(results2)} results after additions")
        assert len(results2) == 5

    print("✓ Search after updates test passed")


def test_statistics():
    """Test knowledge base statistics."""
    print("\n=== Testing Statistics ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Add entities
        for i in range(5):
            entity_id = f"Q{i + 1}"
            entity_data = {"name": f"Entity {i + 1}"}
            entity_embedding = np.random.randn(128).astype(np.float32)
            kb.add_entity(entity_id, entity_data, entity_embedding)

        # Get statistics
        stats = kb.get_statistics()

        print("✓ Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        assert stats["total_entities"] == 5
        assert stats["indexed_entities"] == 5
        assert stats["dimension"] == 128
        assert stats["use_gpu"] == False

    print("✓ Statistics test passed")


def test_rebuild_index():
    """Test index rebuilding."""
    print("\n=== Testing Index Rebuild ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Add entities
        embeddings_dict = {}
        for i in range(5):
            entity_id = f"Q{i + 1}"
            entity_data = {"name": f"Entity {i + 1}"}
            entity_embedding = np.random.randn(128).astype(np.float32)
            embeddings_dict[entity_id] = entity_embedding
            kb.add_entity(entity_id, entity_data, entity_embedding)

        print(f"✓ Added {len(kb.entities)} entities")

        # Remove some (marks for rebuild)
        kb.remove_entity("Q2")
        kb.remove_entity("Q4")

        print("✓ Removed 2 entities")
        print(f"  Needs rebuild: {kb.needs_rebuild()}")

        assert kb.needs_rebuild() == True

        # Rebuild
        remaining_embeddings = {eid: embeddings_dict[eid] for eid in ["Q1", "Q3", "Q5"]}
        kb.rebuild_index(remaining_embeddings)

        print("✓ Rebuilt index")
        print(f"  Needs rebuild: {kb.needs_rebuild()}")

        assert kb.needs_rebuild() == False
        assert len(kb.entity_ids) == 3

    print("✓ Index rebuild test passed")


def test_compact():
    """Test knowledge base compaction."""
    print("\n=== Testing Compact ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Add entities
        for i in range(5):
            entity_id = f"Q{i + 1}"
            entity_data = {"name": f"Entity {i + 1}"}
            entity_embedding = np.random.randn(128).astype(np.float32)
            kb.add_entity(entity_id, entity_data, entity_embedding)

        # Manually add orphaned entity (in storage but not index)
        kb.entities["Q_orphan"] = {"name": "Orphaned Entity"}

        print(f"✓ Added orphaned entity")
        print(f"  Total in storage: {len(kb.entities)}")
        print(f"  Total in index: {len(kb.entity_ids)}")

        # Compact
        kb.compact()

        print("✓ Compacted knowledge base")
        print(f"  Total in storage: {len(kb.entities)}")

        assert "Q_orphan" not in kb.entities
        assert len(kb.entities) == 5

    print("✓ Compact test passed")


def test_persistence():
    """Test save and load functionality."""
    print("\n=== Testing Persistence ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate KB
        kb1 = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        embeddings_dict = {}
        for i in range(5):
            entity_id = f"Q{i + 1}"
            entity_data = {"name": f"Entity {i + 1}", "value": i}
            entity_embedding = np.random.randn(128).astype(np.float32)
            embeddings_dict[entity_id] = entity_embedding
            kb1.add_entity(entity_id, entity_data, entity_embedding)

        print(f"✓ Created KB with {len(kb1.entities)} entities")

        # Create new KB instance and load
        kb2 = EntityKnowledgeBase(
            index_path=f"{tmpdir}/index",
            cache_dir=f"{tmpdir}/cache",
            dimension=128,
            use_gpu=False,
        )

        # Load index
        success = kb2.load_index()
        assert success

        print(f"✓ Loaded KB with {len(kb2.entity_ids)} entities from index")

        # Verify
        assert len(kb2.entity_ids) == 5
        assert "Q1" in kb2.entity_ids

        # Test search
        query = np.random.randn(128).astype(np.float32)
        results = kb2.search(query, k=3)

        print(f"✓ Search works: {len(results)} results")
        assert len(results) == 3

    print("✓ Persistence test passed")


def run_all_tests():
    """Run all dynamic KB tests."""
    print("\n" + "=" * 60)
    print("Running Dynamic Knowledge Base Tests")
    print("=" * 60)

    try:
        test_basic_add_entity()
        test_update_entity()
        test_remove_entity()
        test_batch_operations()
        test_search_after_updates()
        test_statistics()
        test_rebuild_index()
        test_compact()
        test_persistence()

        print("\n" + "=" * 60)
        print("✅ ALL DYNAMIC KB TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
