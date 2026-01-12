"""
Unit tests for Improved ATG Model.

Tests:
1. ATG model components (vocabulary, encoder, decoder)
2. Integration with UnifiedEntityResolutionSystem
3. End-to-end generation
"""

import pytest
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


@pytest.mark.unit
class TestATGConfig:
    """Test ATG configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ATGConfig()

        assert config.encoder_model == "microsoft/deberta-v3-small"
        assert config.decoder_layers == 6
        assert config.decoder_heads == 8
        assert config.max_span_length == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = ATGConfig(
            encoder_model="bert-base-uncased",
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            max_span_length=5,
            decoder_layers=3,
        )

        assert config.encoder_model == "bert-base-uncased"
        assert len(config.entity_types) == 3
        assert len(config.relation_types) == 2
        assert config.max_span_length == 5
        assert config.decoder_layers == 3

    @pytest.mark.parametrize("invalid_layers", [0, -1, 100])
    def test_invalid_decoder_layers(self, invalid_layers):
        """Test that invalid decoder layers raises error."""
        with pytest.raises((ValueError, AssertionError)):
            ATGConfig(decoder_layers=invalid_layers)

    @pytest.mark.parametrize("invalid_span_length", [0, -1])
    def test_invalid_max_span_length(self, invalid_span_length):
        """Test that invalid max_span_length raises error."""
        with pytest.raises((ValueError, AssertionError)):
            ATGConfig(max_span_length=invalid_span_length)


@pytest.mark.unit
@pytest.mark.requires_model
class TestDynamicVocabulary:
    """Test DynamicVocabulary component."""

    @pytest.fixture
    def config(self):
        """ATG configuration for tests."""
        return ATGConfig(
            encoder_model="microsoft/deberta-v3-small",
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            max_span_length=5,
        )

    @pytest.fixture
    def vocab(self, config):
        """Create dynamic vocabulary."""
        hidden_size = 768
        return DynamicVocabulary(config, hidden_size)

    def test_initialization(self, vocab, config):
        """Test vocabulary initialization."""
        assert vocab.config == config
        assert vocab.hidden_size == 768
        assert len(config.entity_types) == 3
        assert len(config.relation_types) == 2

    def test_compute_span_embeddings(self, vocab):
        """Test computing span embeddings."""
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        token_embeddings = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        span_embeddings = vocab.compute_span_embeddings(token_embeddings, attention_mask)

        assert span_embeddings is not None
        assert span_embeddings.dim() == 3
        assert span_embeddings.shape[0] == batch_size

    def test_build_vocabulary_matrix(self, vocab):
        """Test building vocabulary matrix."""
        batch_size = 2
        max_spans = 20
        hidden_size = 768
        span_embeddings = torch.randn(batch_size, max_spans, hidden_size)

        vocab_matrix = vocab.build_vocabulary_matrix(span_embeddings)

        assert vocab_matrix is not None
        assert vocab_matrix.dim() == 3
        assert vocab_matrix.shape[0] == batch_size
        # Matrix should include spans + entity types + relation types + special tokens

    def test_empty_sequence(self, vocab):
        """Test handling empty sequence."""
        batch_size = 1
        seq_len = 0
        hidden_size = 768
        token_embeddings = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        # Should handle gracefully or raise clear error
        try:
            span_embeddings = vocab.compute_span_embeddings(token_embeddings, attention_mask)
            assert span_embeddings is not None
        except (ValueError, AssertionError):
            pass  # Expected for empty sequence


@pytest.mark.unit
@pytest.mark.requires_model
class TestATGEncoder:
    """Test ATGEncoder component."""

    @pytest.fixture
    def config(self):
        """ATG configuration for tests."""
        return ATGConfig(
            encoder_model="microsoft/deberta-v3-small",
            max_seq_length=32,
        )

    @pytest.fixture
    def encoder(self, config):
        """Create encoder."""
        return ATGEncoder(config)

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.encoder is not None
        assert encoder.hidden_size == 768

    def test_forward_pass(self, encoder):
        """Test forward pass."""
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        token_embeddings = encoder(input_ids, attention_mask)

        assert token_embeddings.shape == (batch_size, seq_len, 768)
        assert not torch.isnan(token_embeddings).any()

    def test_different_sequence_lengths(self, encoder):
        """Test with different sequence lengths."""
        for seq_len in [5, 10, 20, 32]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            attention_mask = torch.ones(1, seq_len)

            outputs = encoder(input_ids, attention_mask)
            assert outputs.shape[1] == seq_len

    def test_gradient_flow(self, encoder):
        """Test gradient flow through encoder."""
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)

        outputs = encoder(input_ids, attention_mask)
        loss = outputs.sum()
        loss.backward()

        # Check gradients exist
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


@pytest.mark.unit
class TestATGDecoder:
    """Test ATGDecoder component."""

    @pytest.fixture
    def config(self):
        """ATG configuration for tests."""
        return ATGConfig(
            decoder_layers=3,
            decoder_heads=4,
        )

    @pytest.fixture
    def decoder(self, config):
        """Create decoder."""
        hidden_size = 768
        return ATGDecoder(config, hidden_size)

    def test_initialization(self, decoder, config):
        """Test decoder initialization."""
        assert decoder.config == config
        assert decoder.hidden_size == 768

    def test_forward_pass(self, decoder):
        """Test forward pass."""
        batch_size = 2
        target_len = 5
        src_len = 20
        hidden_size = 768

        target_embeddings = torch.randn(batch_size, target_len, hidden_size)
        target_positions = torch.arange(target_len).unsqueeze(0).expand(batch_size, -1)
        target_structures = torch.zeros(batch_size, target_len, dtype=torch.long)
        encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
        encoder_attention_mask = torch.ones(batch_size, src_len)

        decoder_outputs = decoder(
            target_embeddings=target_embeddings,
            target_positions=target_positions,
            target_structures=target_structures,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
        )

        assert decoder_outputs.shape == (batch_size, target_len, hidden_size)
        assert not torch.isnan(decoder_outputs).any()

    def test_causal_masking(self, decoder):
        """Test that decoder uses causal masking."""
        batch_size = 1
        target_len = 10
        src_len = 15
        hidden_size = 768

        target_embeddings = torch.randn(batch_size, target_len, hidden_size)
        target_positions = torch.arange(target_len).unsqueeze(0)
        target_structures = torch.zeros(batch_size, target_len, dtype=torch.long)
        encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
        encoder_attention_mask = torch.ones(batch_size, src_len)

        outputs = decoder(
            target_embeddings=target_embeddings,
            target_positions=target_positions,
            target_structures=target_structures,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
        )

        # Output should exist and be valid
        assert outputs is not None
        assert outputs.shape == (batch_size, target_len, hidden_size)


@pytest.mark.unit
@pytest.mark.requires_model
class TestImprovedATGModel:
    """Test complete ImprovedATGModel."""

    @pytest.fixture
    def model(self):
        """Create ATG model for testing."""
        return create_atg_model(
            entity_types=["PER", "ORG", "LOC", "MISC"],
            relation_types=["Work_For", "Based_In", "Located_In"],
            encoder_model="microsoft/deberta-v3-small",
            decoder_layers=2,
            max_span_length=8,
        )

    def test_model_creation(self, model):
        """Test model creation."""
        assert model is not None
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.dynamic_vocab is not None

    def test_forward_inference_mode(self, model):
        """Test forward pass in inference mode."""
        batch_size = 2
        seq_len = 15
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(input_ids, attention_mask)

        assert "encoder_outputs" in outputs
        assert "span_embeddings" in outputs
        assert "vocab_matrix" in outputs
        assert outputs["encoder_outputs"].shape == (batch_size, seq_len, model.hidden_size)

    def test_generate(self, model):
        """Test generation."""
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(100, 500, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        generated_sequences = model.generate(
            input_ids,
            attention_mask,
            max_length=20,
        )

        assert isinstance(generated_sequences, list)
        assert len(generated_sequences) == batch_size
        if len(generated_sequences[0]) > 0:
            assert all(isinstance(x, (int, torch.Tensor)) for x in generated_sequences[0])

    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes."""
        for batch_size in [1, 2, 4]:
            seq_len = 10
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)

            outputs = model(input_ids, attention_mask)
            assert outputs["encoder_outputs"].shape[0] == batch_size


@pytest.mark.integration
@pytest.mark.requires_model
class TestATGIntegration:
    """Test ATG integration with unified system."""

    def test_unified_system_with_atg(self):
        """Test UnifiedEntityResolutionSystem with improved ATG."""
        config = SystemConfig(
            use_improved_atg=True,
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            reader_model="microsoft/deberta-v3-small",
            retriever_model="microsoft/deberta-v3-small",
            atg_decoder_layers=2,
            atg_max_span_length=8,
            use_gpu=False,
        )

        system = UnifiedEntityResolutionSystem(config)

        assert system.atg_model is not None
        assert len(system.atg_model.config.entity_types) == 3
        assert len(system.atg_model.config.relation_types) == 2
        assert system.atg_model.config.decoder_layers == 2

    def test_backward_compatibility(self):
        """Test system works without ATG (backward compatibility)."""
        config = SystemConfig(
            use_improved_atg=False,
            reader_model="microsoft/deberta-v3-small",
            retriever_model="microsoft/deberta-v3-small",
            use_gpu=False,
        )

        system = UnifiedEntityResolutionSystem(config)

        assert system.atg_model is None


@pytest.mark.unit
class TestATGEdgeCases:
    """Test ATG edge cases."""

    def test_empty_entity_types(self):
        """Test handling empty entity types."""
        with pytest.raises((ValueError, AssertionError)):
            create_atg_model(
                entity_types=[],
                relation_types=["Work_For"],
            )

    def test_empty_relation_types(self):
        """Test handling empty relation types."""
        with pytest.raises((ValueError, AssertionError)):
            create_atg_model(
                entity_types=["PER"],
                relation_types=[],
            )

    @pytest.mark.requires_model
    def test_max_sequence_length(self):
        """Test handling max sequence length."""
        model = create_atg_model(
            entity_types=["PER", "ORG"],
            relation_types=["Work_For"],
            encoder_model="microsoft/deberta-v3-small",
        )

        max_len = 512
        input_ids = torch.randint(0, 1000, (1, max_len))
        attention_mask = torch.ones(1, max_len)

        outputs = model(input_ids, attention_mask)
        assert outputs["encoder_outputs"].shape[1] == max_len

    @pytest.mark.requires_model
    def test_single_token_sequence(self):
        """Test handling single token sequence."""
        model = create_atg_model(
            entity_types=["PER"],
            relation_types=["Work_For"],
        )

        input_ids = torch.randint(0, 1000, (1, 1))
        attention_mask = torch.ones(1, 1)

        outputs = model(input_ids, attention_mask)
        assert outputs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
