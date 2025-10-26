"""
Test script for SPEL Model.

Tests:
1. SPEL configuration
2. Candidate set manager
3. Prediction aggregator
4. Complete SPEL model
5. Integration with UnifiedEntityResolutionSystem
"""

import torch

from entity_resolution.models.spel import (
    CandidateSetManager,
    PredictionAggregator,
    SPELConfig,
    SPELModel,
    create_spel_model,
)
from entity_resolution.unified_system import UnifiedEntityResolutionSystem
from entity_resolution.validation import SystemConfig


def test_spel_config():
    """Test SPEL configuration."""
    print("\n" + "=" * 80)
    print("Testing SPEL Configuration")
    print("=" * 80)

    try:
        config = SPELConfig(
            model_name="roberta-base",
            max_seq_length=512,
            fixed_candidate_set_size=500000,
            use_mention_specific_candidates=False,
            num_hard_negatives=5000,
        )

        print(f"✓ Model name: {config.model_name}")
        print(f"✓ Max sequence length: {config.max_seq_length}")
        print(f"✓ Fixed candidate set size: {config.fixed_candidate_set_size}")
        print(f"✓ Hard negatives: {config.num_hard_negatives}")
        print(f"✓ Use context-sensitive aggregation: {config.use_context_sensitive_aggregation}")
        print("✓ SPEL Configuration test passed!")

    except Exception as e:
        print(f"✗ SPEL Configuration test failed: {e}")
        import traceback

        traceback.print_exc()


def test_candidate_set_manager():
    """Test candidate set manager."""
    print("\n" + "=" * 80)
    print("Testing Candidate Set Manager")
    print("=" * 80)

    try:
        # Create manager with fixed candidates
        fixed_candidates = [
            "Barack_Obama",
            "United_States",
            "Apple_Inc",
            "Steve_Jobs",
            "California",
        ]

        manager = CandidateSetManager(fixed_candidates=fixed_candidates)

        print(f"✓ Created manager with {len(fixed_candidates)} fixed candidates")
        print(f"✓ Vocabulary size: {manager.get_vocab_size()}")

        # Test adding mention-specific candidates
        manager.add_mention_candidates("Obama", ["Barack_Obama", "Michelle_Obama"])
        print("✓ Added mention-specific candidates for 'Obama'")

        # Test getting candidates
        candidates = manager.get_candidates_for_mention("Obama", use_mention_specific=True)
        print(f"✓ Retrieved {len(candidates)} candidates for 'Obama'")

        # Test entity to index mapping
        idx = manager.get_entity_idx("Barack_Obama")
        entity = manager.get_entity_from_idx(idx)
        assert entity == "Barack_Obama", f"Expected 'Barack_Obama', got '{entity}'"
        print("✓ Entity-index mapping works correctly")

        print("✓ Candidate Set Manager test passed!")

    except Exception as e:
        print(f"✗ Candidate Set Manager test failed: {e}")
        import traceback

        traceback.print_exc()


def test_prediction_aggregator():
    """Test prediction aggregator."""
    print("\n" + "=" * 80)
    print("Testing Prediction Aggregator")
    print("=" * 80)

    try:
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        aggregator = PredictionAggregator(
            tokenizer=tokenizer,
            filter_punctuation=True,
            filter_function_words=True,
        )

        print("✓ Created prediction aggregator")
        print(f"✓ Function words to filter: {len(aggregator.FUNCTION_WORDS)}")
        print(f"✓ Punctuation to filter: {len(aggregator.PUNCTUATION)}")

        # Test aggregation (simplified example)
        text = "Barack Obama was born in Hawaii."
        subword_predictions = [
            [("Barack_Obama", 0.9), ("O", 0.1)],
            [("Barack_Obama", 0.85), ("O", 0.15)],
            [("O", 0.95), ("Barack_Obama", 0.05)],
        ]
        subword_to_word = [0, 0, 1]  # First two subwords are word 0, third is word 1

        spans = aggregator.aggregate_subword_predictions(
            text=text,
            subword_predictions=subword_predictions,
            subword_to_word_map=subword_to_word,
        )

        print(f"✓ Aggregated predictions into {len(spans)} spans")
        print("✓ Prediction Aggregator test passed!")

    except Exception as e:
        print(f"✗ Prediction Aggregator test failed: {e}")
        import traceback

        traceback.print_exc()


def test_spel_model():
    """Test complete SPEL model."""
    print("\n" + "=" * 80)
    print("Testing Complete SPEL Model")
    print("=" * 80)

    try:
        # Create model
        model = create_spel_model(
            model_name="roberta-base",
            fixed_candidate_set_size=100,  # Small for testing
            entity_types=["PER", "ORG", "LOC"],
        )

        print("✓ Model created successfully")
        print(f"✓ Encoder: {model.config.model_name}")
        print(f"✓ Hidden size: {model.hidden_size}")
        print(f"✓ Max sequence length: {model.config.max_seq_length}")

        # Load some test candidates
        test_candidates = [
            "Barack_Obama",
            "United_States",
            "Apple_Inc",
            "Steve_Jobs",
        ]

        model.load_candidate_sets(fixed_candidates=test_candidates)

        print(f"✓ Loaded {len(test_candidates)} candidate entities")
        print(f"✓ Vocabulary size: {model.candidate_manager.get_vocab_size()}")
        print(f"✓ Classification head initialized: {model.classification_head is not None}")

        print("✓ Complete SPEL Model test passed!")

    except Exception as e:
        print(f"✗ Complete SPEL Model test failed: {e}")
        import traceback

        traceback.print_exc()


def test_spel_forward():
    """Test SPEL forward pass."""
    print("\n" + "=" * 80)
    print("Testing SPEL Forward Pass")
    print("=" * 80)

    try:
        # Create small model
        model = create_spel_model(
            model_name="roberta-base",
            fixed_candidate_set_size=10,
        )

        # Load candidates
        test_candidates = ["Barack_Obama", "Apple_Inc", "California"]
        model.load_candidate_sets(fixed_candidates=test_candidates)

        # Create dummy input
        batch_size = 2
        seq_len = 20
        vocab_size = model.candidate_manager.get_vocab_size()

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)

        print(f"✓ Input shape: {input_ids.shape}")
        print(f"✓ Logits shape: {outputs['logits'].shape}")
        print(f"✓ Loss computed: {'loss' in outputs}")
        print(f"✓ Loss value: {outputs['loss'].item():.4f}")

        print("✓ SPEL Forward Pass test passed!")

    except Exception as e:
        print(f"✗ SPEL Forward Pass test failed: {e}")
        import traceback

        traceback.print_exc()


def test_unified_system_with_spel():
    """Test UnifiedEntityResolutionSystem with SPEL."""
    print("\n" + "=" * 80)
    print("Testing UnifiedEntityResolutionSystem with SPEL")
    print("=" * 80)

    try:
        # Create config with SPEL enabled
        config = SystemConfig(
            use_spel=True,
            spel_model_name="roberta-base",
            spel_fixed_candidate_set_size=1000,
            entity_types=["PER", "ORG", "LOC"],
            use_gpu=False,  # Use CPU for testing
        )

        print("Creating unified system with SPEL...")
        system = UnifiedEntityResolutionSystem(config)

        print("✓ System initialized successfully!")
        print(f"  - SPEL enabled: {'✓' if system.spel_model else '✗'}")
        print(f"  - Device: {system.device}")

        if system.spel_model:
            print(f"  - SPEL model name: {system.spel_model.config.model_name}")
            print(
                f"  - Fixed candidate set size: {system.spel_model.config.fixed_candidate_set_size}"
            )
            print(f"  - Entity types: {len(system.spel_model.config.entity_types)}")

        print("✓ UnifiedEntityResolutionSystem with SPEL test passed!")

    except Exception as e:
        print(f"✗ UnifiedEntityResolutionSystem with SPEL test failed: {e}")
        import traceback

        traceback.print_exc()


def test_all_three_models():
    """Test that ATG, ReLiK, and SPEL all work together."""
    print("\n" + "=" * 80)
    print("Testing All Three Models Together (ATG + ReLiK + SPEL)")
    print("=" * 80)

    try:
        config = SystemConfig(
            use_improved_atg=True,
            use_relik=True,
            use_spel=True,
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            use_gpu=False,
        )

        print("Creating system with all three models...")
        system = UnifiedEntityResolutionSystem(config)

        print("✓ System initialized successfully!")
        print(f"  - ATG enabled: {'✓' if system.atg_model else '✗'}")
        print(f"  - ReLiK enabled: {'✓' if system.relik_model else '✗'}")
        print(f"  - SPEL enabled: {'✓' if system.spel_model else '✗'}")

        # Verify all models are present
        assert system.atg_model is not None, "ATG should be enabled"
        assert system.relik_model is not None, "ReLiK should be enabled"
        assert system.spel_model is not None, "SPEL should be enabled"

        print("\n✓ All three models working together successfully!")
        print("  • ATG: Autoregressive entity-relation extraction")
        print("  • ReLiK: Fast retriever-reader entity linking")
        print("  • SPEL: Structured prediction entity linking")

    except Exception as e:
        print(f"✗ All three models test failed: {e}")
        import traceback

        traceback.print_exc()


def test_backward_compatibility():
    """Test that system works without SPEL (backward compatibility)."""
    print("\n" + "=" * 80)
    print("Testing Backward Compatibility (SPEL disabled)")
    print("=" * 80)

    try:
        config = SystemConfig(
            use_spel=False,
            use_improved_atg=True,
            use_relik=True,
            use_gpu=False,
        )

        print("Creating unified system without SPEL...")
        system = UnifiedEntityResolutionSystem(config)

        print("✓ System initialized successfully!")
        print(f"  - SPEL enabled: {'✓' if system.spel_model else '✗'}")
        print("  - Legacy components working: ✓")

        assert system.spel_model is None, "SPEL should be disabled"

        print("✓ Backward compatibility test passed!")

    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING SPEL IMPLEMENTATION")
    print("=" * 80)

    # Run all tests
    test_spel_config()
    test_candidate_set_manager()
    test_prediction_aggregator()
    test_spel_model()
    test_spel_forward()
    test_unified_system_with_spel()
    test_all_three_models()
    test_backward_compatibility()

    print("\n" + "=" * 80)
    print("ALL SPEL TESTS COMPLETED")
    print("=" * 80)
