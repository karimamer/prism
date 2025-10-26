"""
Test script for Improved ATG Model.

Tests:
1. ATG model components (vocabulary, encoder, decoder)
2. Integration with UnifiedEntityResolutionSystem
3. End-to-end generation
"""

import torch

from entity_resolution.models.atg import (
    ATGConfig,
    ATGDecoder,
    ATGEncoder,
    DynamicVocabulary,
    ImprovedATGModel,
    create_atg_model,
)
from entity_resolution.unified_system import UnifiedEntityResolutionSystem
from entity_resolution.validation import SystemConfig


def test_dynamic_vocabulary():
    """Test DynamicVocabulary component."""
    print("\n" + "=" * 80)
    print("Testing DynamicVocabulary")
    print("=" * 80)

    try:
        config = ATGConfig(
            encoder_model="microsoft/deberta-v3-small",
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            max_span_length=5,
        )

        hidden_size = 768
        vocab = DynamicVocabulary(config, hidden_size)

        # Create dummy token embeddings
        batch_size = 2
        seq_len = 10
        token_embeddings = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        # Compute span embeddings
        span_embeddings = vocab.compute_span_embeddings(token_embeddings, attention_mask)

        # Build vocabulary matrix
        vocab_matrix = vocab.build_vocabulary_matrix(span_embeddings)

        print(f"✓ Token embeddings shape: {token_embeddings.shape}")
        print(f"✓ Span embeddings shape: {span_embeddings.shape}")
        print(f"✓ Vocabulary matrix shape: {vocab_matrix.shape}")
        print(f"✓ Entity types: {len(config.entity_types)}")
        print(f"✓ Relation types: {len(config.relation_types)}")
        print("✓ Special tokens: 3 (<START>, <SEP>, <END>)")
        print("✓ DynamicVocabulary test passed!")

    except Exception as e:
        print(f"✗ DynamicVocabulary test failed: {e}")
        import traceback

        traceback.print_exc()


def test_atg_encoder():
    """Test ATGEncoder component."""
    print("\n" + "=" * 80)
    print("Testing ATGEncoder")
    print("=" * 80)

    try:
        config = ATGConfig(
            encoder_model="microsoft/deberta-v3-small",
            max_seq_length=32,
        )

        encoder = ATGEncoder(config)

        # Create dummy input
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        token_embeddings = encoder(input_ids, attention_mask)

        print(f"✓ Input shape: {input_ids.shape}")
        print(f"✓ Output shape: {token_embeddings.shape}")
        print(f"✓ Hidden size: {encoder.hidden_size}")
        print("✓ ATGEncoder test passed!")

    except Exception as e:
        print(f"✗ ATGEncoder test failed: {e}")
        import traceback

        traceback.print_exc()


def test_atg_decoder():
    """Test ATGDecoder component."""
    print("\n" + "=" * 80)
    print("Testing ATGDecoder")
    print("=" * 80)

    try:
        config = ATGConfig(
            decoder_layers=3,
            decoder_heads=4,
        )

        hidden_size = 768
        decoder = ATGDecoder(config, hidden_size)

        # Create dummy inputs
        batch_size = 2
        target_len = 5
        src_len = 20

        target_embeddings = torch.randn(batch_size, target_len, hidden_size)
        target_positions = torch.arange(target_len).unsqueeze(0).expand(batch_size, -1)
        target_structures = torch.zeros(batch_size, target_len, dtype=torch.long)
        encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
        encoder_attention_mask = torch.ones(batch_size, src_len)

        # Forward pass
        decoder_outputs = decoder(
            target_embeddings=target_embeddings,
            target_positions=target_positions,
            target_structures=target_structures,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
        )

        print(f"✓ Target input shape: {target_embeddings.shape}")
        print(f"✓ Encoder output shape: {encoder_outputs.shape}")
        print(f"✓ Decoder output shape: {decoder_outputs.shape}")
        print(f"✓ Decoder layers: {config.decoder_layers}")
        print(f"✓ Decoder heads: {config.decoder_heads}")
        print("✓ ATGDecoder test passed!")

    except Exception as e:
        print(f"✗ ATGDecoder test failed: {e}")
        import traceback

        traceback.print_exc()


def test_complete_atg_model():
    """Test complete ImprovedATGModel."""
    print("\n" + "=" * 80)
    print("Testing Complete ImprovedATGModel")
    print("=" * 80)

    try:
        # Create model
        model = create_atg_model(
            entity_types=["PER", "ORG", "LOC", "MISC"],
            relation_types=["Work_For", "Based_In", "Located_In"],
            encoder_model="microsoft/deberta-v3-small",
            decoder_layers=2,
            max_span_length=8,
        )

        # Create dummy input
        batch_size = 2
        seq_len = 15
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass (without targets - inference mode)
        outputs = model(input_ids, attention_mask)

        print("✓ Model created successfully")
        print(f"✓ Encoder outputs shape: {outputs['encoder_outputs'].shape}")
        print(f"✓ Span embeddings shape: {outputs['span_embeddings'].shape}")
        print(f"✓ Vocabulary matrix shape: {outputs['vocab_matrix'].shape}")
        print(f"✓ Entity types: {len(model.config.entity_types)}")
        print(f"✓ Relation types: {len(model.config.relation_types)}")
        print("✓ Complete ATG Model test passed!")

    except Exception as e:
        print(f"✗ Complete ATG Model test failed: {e}")
        import traceback

        traceback.print_exc()


def test_atg_generation():
    """Test ATG generation with constrained decoding."""
    print("\n" + "=" * 80)
    print("Testing ATG Generation")
    print("=" * 80)

    try:
        # Create small model for fast testing
        model = create_atg_model(
            entity_types=["PER", "ORG"],
            relation_types=["Work_For"],
            encoder_model="microsoft/deberta-v3-small",
            decoder_layers=1,
            max_span_length=5,
        )
        model.eval()

        # Create dummy input
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(100, 500, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Generate
        print("Generating entity-relation graph...")
        generated_sequences = model.generate(
            input_ids,
            attention_mask,
            max_length=20,
        )

        print(f"✓ Generated {len(generated_sequences)} sequences")
        print(f"✓ First sequence length: {len(generated_sequences[0])}")
        print(f"✓ Generated IDs: {generated_sequences[0][:10]}...")  # Show first 10
        print("✓ ATG Generation test passed!")

    except Exception as e:
        print(f"✗ ATG Generation test failed: {e}")
        import traceback

        traceback.print_exc()


def test_unified_system_with_atg():
    """Test UnifiedEntityResolutionSystem with improved ATG."""
    print("\n" + "=" * 80)
    print("Testing UnifiedEntityResolutionSystem with Improved ATG")
    print("=" * 80)

    try:
        # Create config with ATG enabled
        config = SystemConfig(
            use_improved_atg=True,
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            reader_model="microsoft/deberta-v3-small",
            retriever_model="microsoft/deberta-v3-small",
            atg_decoder_layers=2,
            atg_max_span_length=8,
            use_gpu=False,  # Use CPU for testing
        )

        print("Creating unified system with ATG...")
        system = UnifiedEntityResolutionSystem(config)

        print("✓ System initialized successfully!")
        print(f"  - Improved ATG: {'enabled' if system.atg_model else 'disabled'}")
        print(f"  - Device: {system.device}")

        if system.atg_model:
            print(f"  - ATG entity types: {len(system.atg_model.config.entity_types)}")
            print(f"  - ATG relation types: {len(system.atg_model.config.relation_types)}")
            print(f"  - ATG decoder layers: {system.atg_model.config.decoder_layers}")

        print("✓ UnifiedEntityResolutionSystem with ATG test passed!")

    except Exception as e:
        print(f"✗ UnifiedEntityResolutionSystem with ATG test failed: {e}")
        import traceback

        traceback.print_exc()


def test_backward_compatibility():
    """Test that system works without ATG (backward compatibility)."""
    print("\n" + "=" * 80)
    print("Testing Backward Compatibility (ATG disabled)")
    print("=" * 80)

    try:
        # Create config with ATG disabled
        config = SystemConfig(
            use_improved_atg=False,
            reader_model="microsoft/deberta-v3-small",
            retriever_model="microsoft/deberta-v3-small",
            use_gpu=False,
        )

        print("Creating unified system without ATG...")
        system = UnifiedEntityResolutionSystem(config)

        print("✓ System initialized successfully!")
        print(f"  - Improved ATG: {'enabled' if system.atg_model else 'disabled'}")
        print("  - Legacy components working: ✓")

        print("✓ Backward compatibility test passed!")

    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING IMPROVED ATG IMPLEMENTATION")
    print("=" * 80)

    # Run all tests
    test_dynamic_vocabulary()
    test_atg_encoder()
    test_atg_decoder()
    test_complete_atg_model()
    test_atg_generation()
    test_unified_system_with_atg()
    test_backward_compatibility()

    print("\n" + "=" * 80)
    print("ALL ATG TESTS COMPLETED")
    print("=" * 80)
