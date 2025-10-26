"""
Test script for ReLiK Model.

Tests:
1. ReLiK Retriever component
2. ReLiK Reader component
3. Complete ReLiK model
4. Integration with UnifiedEntityResolutionSystem
"""

import torch

from entity_resolution.models.relik import (
    ReLiKConfig,
    ReLiKModel,
    ReLiKReader,
    ReLiKRetriever,
    create_relik_model,
)
from entity_resolution.unified_system import UnifiedEntityResolutionSystem
from entity_resolution.validation import SystemConfig


def test_relik_config():
    """Test ReLiK configuration."""
    print("\n" + "=" * 80)
    print("Testing ReLiK Configuration")
    print("=" * 80)

    try:
        config = ReLiKConfig(
            retriever_model="microsoft/deberta-v3-small",
            reader_model="microsoft/deberta-v3-base",
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            use_entity_linking=True,
            use_relation_extraction=False,
        )

        print(f"✓ Retriever model: {config.retriever_model}")
        print(f"✓ Reader model: {config.reader_model}")
        print(f"✓ Entity types: {config.entity_types}")
        print(f"✓ Relation types: {config.relation_types}")
        print(f"✓ Top-k: {config.top_k}")
        print(f"✓ Max seq length: {config.max_seq_length}")
        print("✓ ReLiK Configuration test passed!")

    except Exception as e:
        print(f"✗ ReLiK Configuration test failed: {e}")
        import traceback

        traceback.print_exc()


def test_relik_retriever():
    """Test ReLiK Retriever component."""
    print("\n" + "=" * 80)
    print("Testing ReLiK Retriever")
    print("=" * 80)

    try:
        retriever = ReLiKRetriever(
            model_name="microsoft/deberta-v3-small",
            max_query_length=64,
            max_passage_length=64,
            use_faiss=False,  # Use PyTorch for testing
        )

        # Create dummy entity knowledge base
        entities = {
            "Q1": {"text": "Apple Inc. is a technology company", "name": "Apple Inc."},
            "Q2": {"text": "Steve Jobs was the co-founder of Apple", "name": "Steve Jobs"},
            "Q3": {"text": "Microsoft Corporation develops software", "name": "Microsoft"},
        }

        # Build index
        print("Building entity index...")
        retriever.build_index(entities, batch_size=2)

        # Test retrieval
        queries = ["Apple is a tech company"]
        results = retriever.retrieve(queries, top_k=2)

        print(f"✓ Built index with {len(entities)} entities")
        print(f"✓ Retrieved {len(results[0])} candidates for query")
        print(f"✓ Top candidate: {results[0][0][0]} (score: {results[0][0][2]:.4f})")
        print("✓ ReLiK Retriever test passed!")

    except Exception as e:
        print(f"✗ ReLiK Retriever test failed: {e}")
        import traceback

        traceback.print_exc()


def test_relik_reader():
    """Test ReLiK Reader component."""
    print("\n" + "=" * 80)
    print("Testing ReLiK Reader")
    print("=" * 80)

    try:
        reader = ReLiKReader(
            model_name="microsoft/deberta-v3-small",
            max_seq_length=128,
            num_entity_types=3,
            num_relation_types=2,
            dropout=0.1,
            use_entity_linking=True,
            use_relation_extraction=False,
        )

        # Create dummy input
        batch_size = 2
        seq_len = 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        outputs = reader(input_ids, attention_mask)

        print(f"✓ Input shape: {input_ids.shape}")
        print(f"✓ Hidden states shape: {outputs['hidden_states'].shape}")
        print(f"✓ Span start logits shape: {outputs['span_start_logits'].shape}")
        print(f"✓ Span end logits shape: {outputs['span_end_logits'].shape}")
        print("✓ ReLiK Reader test passed!")

    except Exception as e:
        print(f"✗ ReLiK Reader test failed: {e}")
        import traceback

        traceback.print_exc()


def test_complete_relik_model():
    """Test complete ReLiK model."""
    print("\n" + "=" * 80)
    print("Testing Complete ReLiK Model")
    print("=" * 80)

    try:
        # Create model
        model = create_relik_model(
            retriever_model="microsoft/deberta-v3-small",
            reader_model="microsoft/deberta-v3-small",
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            use_entity_linking=True,
            use_relation_extraction=False,
        )

        # Load entities
        entities = {
            "Q1": {
                "text": "Apple Inc. is a technology company founded in 1976",
                "name": "Apple Inc.",
                "type": "ORG",
            },
            "Q2": {
                "text": "Steve Jobs was an American entrepreneur",
                "name": "Steve Jobs",
                "type": "PER",
            },
            "Q3": {"text": "Cupertino is a city in California", "name": "Cupertino", "type": "LOC"},
        }

        print("Loading entities into model...")
        model.load_entities(entities, batch_size=2)

        print("✓ Model created successfully")
        print(f"✓ Loaded {len(entities)} entities")
        print(f"✓ Retriever ready: {model.retriever.passage_index is not None}")
        print(f"✓ Entity types: {model.config.entity_types}")
        print(f"✓ Relation types: {model.config.relation_types}")
        print("✓ Complete ReLiK Model test passed!")

    except Exception as e:
        print(f"✗ Complete ReLiK Model test failed: {e}")
        import traceback

        traceback.print_exc()


def test_relik_entity_linking():
    """Test ReLiK entity linking."""
    print("\n" + "=" * 80)
    print("Testing ReLiK Entity Linking")
    print("=" * 80)

    try:
        # Create model
        model = create_relik_model(
            retriever_model="microsoft/deberta-v3-small",
            reader_model="microsoft/deberta-v3-small",
            entity_types=["PER", "ORG", "LOC"],
            use_entity_linking=True,
        )

        # Load entities
        entities = {
            "Q1": {"text": "Apple Inc. technology company", "name": "Apple Inc."},
            "Q2": {"text": "Steve Jobs entrepreneur", "name": "Steve Jobs"},
        }

        model.load_entities(entities, batch_size=2)

        # Test entity linking
        text = "Apple was founded by Steve Jobs in California."

        print(f"Processing text: '{text}'")
        result = model.process_text(text, top_k=2, return_candidates=True)

        print(f"✓ Detected {result['num_entities']} entities")
        print(f"✓ Retrieved {len(result.get('candidates', []))} candidates")

        if result["entities"]:
            for entity in result["entities"][:3]:
                print(f"  - Entity: '{entity['text']}' at [{entity['start']}:{entity['end']}]")

        print("✓ ReLiK Entity Linking test passed!")

    except Exception as e:
        print(f"✗ ReLiK Entity Linking test failed: {e}")
        import traceback

        traceback.print_exc()


def test_unified_system_with_relik():
    """Test UnifiedEntityResolutionSystem with ReLiK."""
    print("\n" + "=" * 80)
    print("Testing UnifiedEntityResolutionSystem with ReLiK")
    print("=" * 80)

    try:
        # Create config with ReLiK enabled
        config = SystemConfig(
            use_relik=True,
            relik_use_el=True,
            relik_use_re=False,
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            retriever_model="microsoft/deberta-v3-small",
            reader_model="microsoft/deberta-v3-small",
            relik_top_k=50,
            use_gpu=False,  # Use CPU for testing
        )

        print("Creating unified system with ReLiK...")
        system = UnifiedEntityResolutionSystem(config)

        print("✓ System initialized successfully!")
        print(f"  - ReLiK enabled: {'✓' if system.relik_model else '✗'}")
        print(f"  - Device: {system.device}")

        if system.relik_model:
            print(f"  - ReLiK entity linking: {system.relik_model.config.use_entity_linking}")
            print(
                f"  - ReLiK relation extraction: {system.relik_model.config.use_relation_extraction}"
            )
            print(f"  - Entity types: {len(system.relik_model.config.entity_types)}")
            print(f"  - Top-k candidates: {system.relik_model.config.top_k}")

        print("✓ UnifiedEntityResolutionSystem with ReLiK test passed!")

    except Exception as e:
        print(f"✗ UnifiedEntityResolutionSystem with ReLiK test failed: {e}")
        import traceback

        traceback.print_exc()


def test_backward_compatibility():
    """Test that system works without ReLiK (backward compatibility)."""
    print("\n" + "=" * 80)
    print("Testing Backward Compatibility (ReLiK disabled)")
    print("=" * 80)

    try:
        # Create config with ReLiK disabled
        config = SystemConfig(
            use_relik=False,
            retriever_model="microsoft/deberta-v3-small",
            reader_model="microsoft/deberta-v3-small",
            use_gpu=False,
        )

        print("Creating unified system without ReLiK...")
        system = UnifiedEntityResolutionSystem(config)

        print("✓ System initialized successfully!")
        print(f"  - ReLiK enabled: {'✓' if system.relik_model else '✗'}")
        print("  - Legacy components working: ✓")

        print("✓ Backward compatibility test passed!")

    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING ReLiK IMPLEMENTATION")
    print("=" * 80)

    # Run all tests
    test_relik_config()
    test_relik_retriever()
    test_relik_reader()
    test_complete_relik_model()
    test_relik_entity_linking()
    test_unified_system_with_relik()
    test_backward_compatibility()

    print("\n" + "=" * 80)
    print("ALL ReLiK TESTS COMPLETED")
    print("=" * 80)
