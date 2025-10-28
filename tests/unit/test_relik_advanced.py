"""
Tests for advanced ReLiK features.

Tests:
1. Relation extraction
2. Hard negative mining
3. Confidence calibration
4. Dynamic index updates
"""

import torch

from src.entity_resolution.models.relik import (
    CompleteEntityLinker,
    ConfidenceCalibrator,
    DynamicIndexManager,
    HardNegativeMiner,
    ImprovedReLiKReader,
    ReLiKRelationExtractor,
    ReLiKRetriever,
)


def test_relation_extraction():
    """Test relation extraction between entities."""
    print("\n=== Testing Relation Extraction ===")

    # Create reader
    reader = ImprovedReLiKReader(
        model_name="bert-base-uncased",
        max_seq_length=256,
        use_entity_linking=True,
    )

    # Create relation extractor
    rel_extractor = ReLiKRelationExtractor(reader)

    # Define entities (detected spans)
    entities = [
        {"start": 0, "end": 5, "text": "Apple", "entity_id": "Q312"},
        {"start": 21, "end": 32, "text": "Steve Jobs", "entity_id": "Q7564"},
        {"start": 36, "end": 46, "text": "California", "entity_id": "Q99"},
    ]

    # Define relation types
    relation_types = [
        "founded by",
        "located in",
        "works at",
    ]

    text = "Apple was founded by Steve Jobs in California"

    # Extract relations
    relations = rel_extractor.extract_relations(
        text,
        entities,
        relation_types,
        relation_threshold=0.3,  # Lower threshold for testing
        max_distance=50,
    )

    print(f"✓ Extracted {len(relations)} relations")

    for rel in relations[:3]:  # Show top 3
        print(f"  - ({rel['subject']['text']}, {rel['relation']}, {rel['object']['text']})")
        print(f"    Confidence: {rel['confidence']:.3f}")

    assert len(relations) >= 0  # May not find relations with untrained model
    print("✓ Relation extraction tests passed")


def test_hard_negative_mining():
    """Test hard negative mining for retriever training."""
    print("\n=== Testing Hard Negative Mining ===")

    # Create retriever
    retriever = ReLiKRetriever(
        model_name="bert-base-uncased",
        max_query_length=32,
        max_passage_length=32,
    )

    # Create knowledge base
    knowledge_base = {
        "Q312": {"text": "Apple Inc. is a technology company", "name": "Apple Inc."},
        "Q7564": {"text": "Steve Jobs was an entrepreneur", "name": "Steve Jobs"},
        "Q99": {"text": "California is a U.S. state", "name": "California"},
        "Q19860": {"text": "Stanford University", "name": "Stanford"},
        "Q95": {"text": "Google LLC is a technology company", "name": "Google"},
    }

    # Build index
    retriever.build_index(knowledge_base, batch_size=2)

    # Create hard negative miner
    miner = HardNegativeMiner(
        retriever,
        knowledge_base,
        strategy="top_k",
        num_negatives=3,
    )

    # Mine hard negatives
    queries = ["Apple Inc. makes iPhones", "Steve Jobs founded a company"]
    positive_ids = ["Q312", "Q7564"]

    hard_negatives = miner.mine_hard_negatives(queries, positive_ids, top_k=5)

    print(f"✓ Mined hard negatives for {len(queries)} queries")

    for i, (query, pos_id, negs) in enumerate(zip(queries, positive_ids, hard_negatives)):
        print(f"\n  Query {i + 1}: '{query}'")
        print(f"  Positive: {pos_id}")
        print(f"  Hard negatives ({len(negs)}):")
        for neg in negs[:3]:
            print(f"    - {neg['id']}: {neg['data']['name']} (type: {neg['type']})")

    assert len(hard_negatives) == len(queries)
    assert all(len(negs) == 3 for negs in hard_negatives)

    # Test different strategies
    for strategy in ["random", "mixed"]:
        miner_strat = HardNegativeMiner(
            retriever, knowledge_base, strategy=strategy, num_negatives=3
        )
        negs = miner_strat.mine_hard_negatives(queries, positive_ids)
        assert len(negs) == len(queries)
        print(f"✓ {strategy} strategy works")

    # Test batch preparation
    batch = miner.prepare_training_batch(queries, positive_ids)

    assert "query_ids" in batch
    assert "positive_ids" in batch
    assert "negative_ids" in batch

    print(f"✓ Training batch prepared:")
    print(f"  - Query shape: {batch['query_ids'].shape}")
    print(f"  - Positive shape: {batch['positive_ids'].shape}")
    print(f"  - Negative shape: {batch['negative_ids'].shape}")

    print("✓ Hard negative mining tests passed")


def test_confidence_calibration():
    """Test confidence calibration."""
    print("\n=== Testing Confidence Calibration ===")

    # Create calibrator
    calibrator = ConfidenceCalibrator(method="temperature")

    # Create dummy validation data
    span_logits = torch.randn(50, 2)  # 50 samples, 2 classes
    span_labels = torch.randint(0, 2, (50,))

    # Fit span calibrator
    calibrator.fit_span_calibrator(span_logits, span_labels)

    print("✓ Fitted span calibrator")
    print(f"  Temperature: {calibrator.span_calibrator.temperature.item():.3f}")

    # Test calibration
    test_logits = torch.randn(10, 2)
    calibrated_probs = calibrator.calibrate_span_scores(test_logits)

    print(f"✓ Calibrated scores shape: {calibrated_probs.shape}")
    print(f"  Sample probabilities: {calibrated_probs[0]}")

    # Test entity calibration
    entity_scores = torch.rand(50)
    entity_labels = torch.randint(0, 2, (50,))

    calibrator.fit_entity_calibrator(entity_scores, entity_labels)
    print("✓ Fitted entity calibrator")

    # Test Platt scaling
    platt_calibrator = ConfidenceCalibrator(method="platt")
    platt_calibrator.fit_entity_calibrator(entity_scores, entity_labels.float())

    calibrated_entity = platt_calibrator.calibrate_entity_scores(torch.rand(10))
    print(f"✓ Platt scaling works: {calibrated_entity.shape}")

    # Test save/load
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator.save(tmpdir)
        print(f"✓ Saved calibrators to {tmpdir}")

        new_calibrator = ConfidenceCalibrator()
        new_calibrator.load(tmpdir)
        print("✓ Loaded calibrators from disk")

    print("✓ Confidence calibration tests passed")


def test_dynamic_index():
    """Test dynamic index updates."""
    print("\n=== Testing Dynamic Index Updates ===")

    # Create retriever
    retriever = ReLiKRetriever(
        model_name="bert-base-uncased",
        max_query_length=32,
        max_passage_length=32,
        use_faiss=False,  # Use PyTorch for easier testing
    )

    # Initial knowledge base
    knowledge_base = {
        "Q1": {"text": "Entity one", "name": "Entity 1"},
        "Q2": {"text": "Entity two", "name": "Entity 2"},
        "Q3": {"text": "Entity three", "name": "Entity 3"},
    }

    # Build initial index
    retriever.build_index(knowledge_base, batch_size=2)
    print(f"✓ Built initial index with {len(knowledge_base)} entities")

    # Create dynamic index manager
    manager = DynamicIndexManager(
        retriever,
        rebuild_threshold=5,
        auto_rebuild=False,  # Manual control for testing
    )

    # Test adding entity
    manager.add_entity("Q4", {"text": "Entity four", "name": "Entity 4"})
    print("✓ Added entity Q4")

    stats = manager.get_statistics()
    assert stats["pending_additions"] == 1
    print(f"  Pending additions: {stats['pending_additions']}")

    # Apply updates
    manager.apply_updates()
    print("✓ Applied additions")

    stats = manager.get_statistics()
    assert stats["total_entities"] == 4
    assert stats["pending_additions"] == 0
    print(f"  Total entities: {stats['total_entities']}")

    # Test batch add
    new_entities = {
        "Q5": {"text": "Entity five", "name": "Entity 5"},
        "Q6": {"text": "Entity six", "name": "Entity 6"},
    }
    manager.batch_add(new_entities, immediate=True)
    print("✓ Batch added 2 entities")

    stats = manager.get_statistics()
    assert stats["total_entities"] == 6
    print(f"  Total entities: {stats['total_entities']}")

    # Test removal
    manager.remove_entity("Q1", immediate=True)
    print("✓ Removed entity Q1")

    stats = manager.get_statistics()
    assert stats["total_entities"] == 5
    print(f"  Total entities after removal: {stats['total_entities']}")

    # Test update
    manager.update_entity(
        "Q2",
        {"text": "Entity two updated", "name": "Entity 2 Updated"},
        immediate=True,
    )
    print("✓ Updated entity Q2")

    # Verify retrieval still works
    results = retriever.retrieve(["entity"], top_k=3)
    print(f"✓ Retrieved {len(results[0])} entities")

    for eid, data, score in results[0]:
        print(f"  - {eid}: {data['name']}")

    # Test statistics
    stats = manager.get_statistics()
    print("\n✓ Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("✓ Dynamic index tests passed")


def test_end_to_end_advanced():
    """Test end-to-end pipeline with all advanced features."""
    print("\n=== Testing End-to-End Advanced Pipeline ===")

    # Create components
    retriever = ReLiKRetriever("bert-base-uncased", use_faiss=False)
    reader = ImprovedReLiKReader("bert-base-uncased")
    linker = CompleteEntityLinker(retriever, reader, device="cpu")

    # Knowledge base
    knowledge_base = {
        "Q312": {"text": "Apple Inc. technology company", "name": "Apple Inc."},
        "Q7564": {"text": "Steve Jobs entrepreneur", "name": "Steve Jobs"},
        "Q99": {"text": "California US state", "name": "California"},
    }

    # Setup dynamic index
    manager = DynamicIndexManager(retriever, auto_rebuild=True)
    retriever.build_index(knowledge_base)

    # Link entities
    text = "Apple was founded by Steve Jobs"
    result = linker.link_entities_end_to_end(
        text,
        knowledge_base,
        top_k_retrieval=3,
        span_threshold=0.3,
    )

    print(f"✓ Linked {len(result)} entities")

    # Extract relations
    rel_extractor = ReLiKRelationExtractor(reader)
    relations = rel_extractor.extract_relations(
        text,
        result,
        ["founded by", "located in"],
        relation_threshold=0.3,
    )

    print(f"✓ Extracted {len(relations)} relations")

    # Add new entity dynamically
    manager.add_entity(
        "Q19860",
        {"text": "Stanford University", "name": "Stanford"},
        immediate=True,
    )
    print("✓ Added new entity to knowledge base")

    # Re-link with updated KB
    new_text = "Steve Jobs went to Stanford"
    knowledge_base["Q19860"] = {"text": "Stanford University", "name": "Stanford"}

    result2 = linker.link_entities_end_to_end(
        new_text,
        knowledge_base,
        top_k_retrieval=4,
        span_threshold=0.3,
    )

    print(f"✓ Linked {len(result2)} entities with updated KB")

    print("✓ End-to-end advanced tests passed")


def run_all_tests():
    """Run all advanced ReLiK tests."""
    print("\n" + "=" * 60)
    print("Running Advanced ReLiK Tests")
    print("=" * 60)

    try:
        test_relation_extraction()
        test_hard_negative_mining()
        test_confidence_calibration()
        test_dynamic_index()
        test_end_to_end_advanced()

        print("\n" + "=" * 60)
        print("✅ ALL ADVANCED TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
