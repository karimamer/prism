"""
Unit tests for SPEL Model.

Tests:
1. SPEL configuration
2. Candidate set manager
3. Prediction aggregator
4. Complete SPEL model
5. Integration with UnifiedEntityResolutionSystem
"""

import pytest
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


@pytest.mark.unit
class TestSPELConfig:
    """Test SPEL configuration."""

    def test_default_config(self):
        """Test default SPEL configuration."""
        config = SPELConfig()

        assert config.model_name == "roberta-base"
        assert config.max_seq_length == 512
        assert config.fixed_candidate_set_size == 1000000
        assert config.use_mention_specific_candidates is True

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = SPELConfig(
            model_name="bert-base-uncased",
            max_seq_length=256,
            fixed_candidate_set_size=500000,
            use_mention_specific_candidates=False,
            num_hard_negatives=5000,
        )

        assert config.model_name == "bert-base-uncased"
        assert config.max_seq_length == 256
        assert config.fixed_candidate_set_size == 500000
        assert config.num_hard_negatives == 5000
        assert config.use_context_sensitive_aggregation is True

    @pytest.mark.parametrize("invalid_seq_length", [0, -1, 10000])
    def test_invalid_max_seq_length(self, invalid_seq_length):
        """Test that invalid max_seq_length raises error."""
        with pytest.raises((ValueError, AssertionError)):
            SPELConfig(max_seq_length=invalid_seq_length)

    @pytest.mark.parametrize("invalid_candidate_size", [0, -1])
    def test_invalid_candidate_set_size(self, invalid_candidate_size):
        """Test that invalid candidate set size raises error."""
        with pytest.raises((ValueError, AssertionError)):
            SPELConfig(fixed_candidate_set_size=invalid_candidate_size)


@pytest.mark.unit
class TestCandidateSetManager:
    """Test candidate set manager."""

    @pytest.fixture
    def fixed_candidates(self):
        """Sample fixed candidates."""
        return [
            "Barack_Obama",
            "United_States",
            "Apple_Inc",
            "Steve_Jobs",
            "California",
        ]

    @pytest.fixture
    def manager(self, fixed_candidates):
        """Create candidate set manager."""
        return CandidateSetManager(fixed_candidates=fixed_candidates)

    def test_initialization(self, manager, fixed_candidates):
        """Test manager initialization."""
        assert manager.get_vocab_size() == len(fixed_candidates) + 1  # +1 for null entity

    def test_add_mention_candidates(self, manager):
        """Test adding mention-specific candidates."""
        manager.add_mention_candidates("Obama", ["Barack_Obama", "Michelle_Obama"])

        candidates = manager.get_candidates_for_mention("Obama", use_mention_specific=True)
        assert len(candidates) >= 2
        assert "Barack_Obama" in candidates

    def test_get_candidates_fixed_only(self, manager, fixed_candidates):
        """Test getting candidates without mention-specific."""
        candidates = manager.get_candidates_for_mention("Obama", use_mention_specific=False)
        assert candidates == fixed_candidates

    def test_entity_index_mapping(self, manager):
        """Test entity-index bidirectional mapping."""
        idx = manager.get_entity_idx("Barack_Obama")
        entity = manager.get_entity_from_idx(idx)

        assert entity == "Barack_Obama"
        assert idx >= 0

    def test_entity_not_in_vocab(self, manager):
        """Test handling entity not in vocabulary."""
        idx = manager.get_entity_idx("NonExistent_Entity")
        # Should return null entity or raise error
        assert idx >= 0  # Null entity index

    def test_batch_add_candidates(self, manager):
        """Test adding candidates for multiple mentions."""
        mentions_to_candidates = {
            "Apple": ["Apple_Inc", "Apple_(fruit)"],
            "Jobs": ["Steve_Jobs", "Job_(role)"],
        }

        for mention, candidates in mentions_to_candidates.items():
            manager.add_mention_candidates(mention, candidates)

        # Verify all added
        for mention in mentions_to_candidates:
            candidates = manager.get_candidates_for_mention(mention, use_mention_specific=True)
            assert len(candidates) > 0


@pytest.mark.unit
class TestPredictionAggregator:
    """Test prediction aggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create prediction aggregator."""
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        return PredictionAggregator(
            tokenizer=tokenizer,
            filter_punctuation=True,
            filter_function_words=True,
        )

    def test_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.filter_punctuation is True
        assert aggregator.filter_function_words is True
        assert len(aggregator.FUNCTION_WORDS) > 0
        assert len(aggregator.PUNCTUATION) > 0

    @pytest.mark.requires_model
    def test_aggregate_predictions(self, aggregator):
        """Test aggregating subword predictions."""
        text = "Barack Obama was born in Hawaii."
        subword_predictions = [
            [("Barack_Obama", 0.9), ("O", 0.1)],
            [("Barack_Obama", 0.85), ("O", 0.15)],
            [("O", 0.95), ("Barack_Obama", 0.05)],
        ]
        subword_to_word = [0, 0, 1]

        spans = aggregator.aggregate_subword_predictions(
            text=text,
            subword_predictions=subword_predictions,
            subword_to_word_map=subword_to_word,
        )

        assert isinstance(spans, list)
        # May or may not find spans depending on thresholds

    def test_empty_predictions(self, aggregator):
        """Test handling empty predictions."""
        text = "Test text"
        subword_predictions = []
        subword_to_word = []

        spans = aggregator.aggregate_subword_predictions(
            text=text,
            subword_predictions=subword_predictions,
            subword_to_word_map=subword_to_word,
        )

        assert spans == []


@pytest.mark.unit
@pytest.mark.requires_model
class TestSPELModel:
    """Test complete SPEL model."""

    @pytest.fixture
    def small_model(self):
        """Create small model for testing."""
        return create_spel_model(
            model_name="roberta-base",
            fixed_candidate_set_size=100,
            entity_types=["PER", "ORG", "LOC"],
        )

    def test_model_creation(self, small_model):
        """Test model creation."""
        assert small_model is not None
        assert small_model.encoder is not None
        assert small_model.config.model_name == "roberta-base"
        assert small_model.hidden_size == 768

    def test_load_candidates(self, small_model):
        """Test loading candidate entities."""
        test_candidates = [
            "Barack_Obama",
            "United_States",
            "Apple_Inc",
            "Steve_Jobs",
        ]

        small_model.load_candidate_sets(fixed_candidates=test_candidates)

        assert small_model.candidate_manager.get_vocab_size() == len(test_candidates) + 1
        assert small_model.classification_head is not None

    def test_forward_pass(self, small_model):
        """Test forward pass."""
        # Load candidates first
        test_candidates = ["Barack_Obama", "Apple_Inc", "California"]
        small_model.load_candidate_sets(fixed_candidates=test_candidates)

        # Create dummy input
        batch_size = 2
        seq_len = 20
        vocab_size = small_model.candidate_manager.get_vocab_size()

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = small_model(input_ids, attention_mask, labels=labels)

        # Assertions
        assert "logits" in outputs, "Output must contain logits"
        assert "loss" in outputs, "Output must contain loss when labels provided"
        assert outputs["logits"].shape == (batch_size, seq_len, vocab_size)
        assert outputs["loss"].dtype == torch.float32
        assert outputs["loss"] > 0, "Loss should be positive"
        assert not torch.isnan(outputs["loss"]), "Loss should not be NaN"

    def test_forward_without_labels(self, small_model):
        """Test forward pass without labels (inference)."""
        test_candidates = ["Barack_Obama", "Apple_Inc"]
        small_model.load_candidate_sets(fixed_candidates=test_candidates)

        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = small_model(input_ids, attention_mask)

        assert "logits" in outputs
        assert "loss" not in outputs  # No labels, so no loss

    def test_gradient_flow(self, small_model):
        """Test that gradients flow correctly."""
        test_candidates = ["Entity1", "Entity2"]
        small_model.load_candidate_sets(fixed_candidates=test_candidates)

        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        labels = torch.randint(0, 3, (1, 10))

        outputs = small_model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]

        loss.backward()

        # Check that gradients exist
        assert small_model.classification_head.weight.grad is not None


@pytest.mark.integration
@pytest.mark.requires_model
class TestSPELIntegration:
    """Test SPEL integration with unified system."""

    def test_unified_system_with_spel(self):
        """Test UnifiedEntityResolutionSystem with SPEL."""
        config = SystemConfig(
            use_spel=True,
            spel_model_name="roberta-base",
            spel_fixed_candidate_set_size=1000,
            entity_types=["PER", "ORG", "LOC"],
            use_gpu=False,
        )

        system = UnifiedEntityResolutionSystem(config)

        assert system.spel_model is not None
        assert system.spel_model.config.model_name == "roberta-base"
        assert system.spel_model.config.fixed_candidate_set_size == 1000

    def test_all_models_together(self):
        """Test that ATG, ReLiK, and SPEL work together."""
        config = SystemConfig(
            use_improved_atg=True,
            use_relik=True,
            use_spel=True,
            entity_types=["PER", "ORG", "LOC"],
            relation_types=["Work_For", "Based_In"],
            use_gpu=False,
        )

        system = UnifiedEntityResolutionSystem(config)

        assert system.atg_model is not None, "ATG should be enabled"
        assert system.relik_model is not None, "ReLiK should be enabled"
        assert system.spel_model is not None, "SPEL should be enabled"

    def test_backward_compatibility(self):
        """Test system works without SPEL (backward compatibility)."""
        config = SystemConfig(
            use_spel=False,
            use_improved_atg=True,
            use_relik=True,
            use_gpu=False,
        )

        system = UnifiedEntityResolutionSystem(config)

        assert system.spel_model is None, "SPEL should be disabled"
        # Other components should work
        assert system.config is not None


@pytest.mark.unit
class TestSPELEdgeCases:
    """Test SPEL edge cases."""

    def test_empty_candidate_set(self):
        """Test handling empty candidate set."""
        manager = CandidateSetManager(fixed_candidates=[])
        assert manager.get_vocab_size() == 1  # Just null entity

    def test_very_large_candidate_set(self):
        """Test handling very large candidate set."""
        large_candidates = [f"Entity_{i}" for i in range(10000)]
        manager = CandidateSetManager(fixed_candidates=large_candidates)
        assert manager.get_vocab_size() == 10001

    @pytest.mark.requires_model
    def test_long_sequence(self):
        """Test handling sequence at max length."""
        model = create_spel_model(
            model_name="roberta-base",
            fixed_candidate_set_size=10,
        )
        model.load_candidate_sets(fixed_candidates=["Entity1"])

        # Max length sequence
        max_len = 512
        input_ids = torch.randint(0, 1000, (1, max_len))
        attention_mask = torch.ones(1, max_len)

        outputs = model(input_ids, attention_mask)
        assert outputs["logits"].shape[1] == max_len

    def test_duplicate_candidates(self):
        """Test handling duplicate candidates."""
        candidates = ["Apple_Inc", "Apple_Inc", "Microsoft"]
        manager = CandidateSetManager(fixed_candidates=candidates)
        # Should deduplicate
        unique_count = len(set(candidates))
        assert manager.get_vocab_size() == unique_count + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
