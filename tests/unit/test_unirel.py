"""
Unit tests for UniRel (Unified Representation and Interaction) model.
"""

import pytest
import torch
from pydantic import ValidationError

from entity_resolution.models.unirel import (
    InteractionDecoder,
    InteractionMap,
    UniRelConfig,
    UniRelModel,
    create_unirel_model,
)


class TestUniRelConfig:
    """Test UniRelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = UniRelConfig()

        assert config.encoder_model == "bert-base-cased"
        assert config.max_seq_length == 512
        assert config.hidden_size == 768
        assert len(config.relation_types) == 3
        assert len(config.entity_types) == 4
        assert config.interaction_threshold == 0.5
        assert config.entity_threshold == 0.5
        assert config.triple_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = UniRelConfig(
            encoder_model="roberta-base",
            max_seq_length=256,
            relation_types=["WorkFor", "LivesIn"],
            entity_types=["PERSON", "LOCATION"],
            interaction_threshold=0.7,
            entity_threshold=0.6,
            triple_threshold=0.8,
        )

        assert config.encoder_model == "roberta-base"
        assert config.max_seq_length == 256
        assert len(config.relation_types) == 2
        assert len(config.entity_types) == 2
        assert config.interaction_threshold == 0.7

    def test_relation_verbalizations(self):
        """Test relation verbalization auto-generation."""
        config = UniRelConfig(relation_types=["Work_For", "Based_In", "Located_In"])

        # Should auto-generate verbalizations
        assert config.relation_verbalizations is not None
        assert config.relation_verbalizations["Work_For"] == "work for"
        assert config.relation_verbalizations["Based_In"] == "based in"

    def test_custom_verbalizations(self):
        """Test custom relation verbalizations."""
        custom_verbs = {
            "Work_For": "is employed by",
            "Based_In": "is located in",
        }
        config = UniRelConfig(
            relation_types=["Work_For", "Based_In"],
            relation_verbalizations=custom_verbs,
        )

        assert config.relation_verbalizations["Work_For"] == "is employed by"
        assert config.relation_verbalizations["Based_In"] == "is located in"

    def test_invalid_thresholds(self):
        """Test that invalid thresholds raise errors."""
        with pytest.raises(ValidationError):
            UniRelConfig(interaction_threshold=1.5)

        with pytest.raises(ValidationError):
            UniRelConfig(entity_threshold=-0.1)

        with pytest.raises(ValidationError):
            UniRelConfig(triple_threshold=2.0)


class TestInteractionMap:
    """Test InteractionMap component."""

    def test_initialization(self):
        """Test InteractionMap initialization."""
        interaction_map = InteractionMap(
            hidden_size=768,
            num_attention_heads=12,
            dropout=0.1,
        )

        assert interaction_map.hidden_size == 768
        assert interaction_map.num_attention_heads == 12
        assert interaction_map.attention_head_size == 64

    def test_forward_pass(self):
        """Test forward pass through InteractionMap."""
        interaction_map = InteractionMap(
            hidden_size=768,
            num_attention_heads=12,
            dropout=0.1,
        )

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        # Run forward pass
        enhanced_hidden, ee_scores, er_scores = interaction_map(
            hidden_states,
            attention_mask=attention_mask,
        )

        # Check output shapes
        assert enhanced_hidden.shape == (batch_size, seq_len, 768)

    def test_entity_entity_interactions(self):
        """Test entity-entity interaction extraction."""
        interaction_map = InteractionMap(
            hidden_size=768,
            num_attention_heads=12,
            dropout=0.1,
        )

        batch_size = 1
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Mark some positions as entities
        entity_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        entity_positions[0, [2, 5, 10]] = True  # 3 entities

        enhanced_hidden, ee_scores, er_scores = interaction_map(
            hidden_states,
            entity_positions=entity_positions[0],
        )

        # Should have entity-entity interaction scores
        assert ee_scores is not None
        # 3 entities with pairwise interactions (excluding self)
        assert ee_scores.shape[0] == 3  # num_entities

    def test_entity_relation_interactions(self):
        """Test entity-relation interaction extraction."""
        interaction_map = InteractionMap(
            hidden_size=768,
            num_attention_heads=12,
            dropout=0.1,
        )

        batch_size = 1
        seq_len = 25
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Mark positions as entities and relations
        entity_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        entity_positions[0, [2, 5]] = True  # 2 entities

        relation_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        relation_positions[0, [20, 22]] = True  # 2 relations

        enhanced_hidden, ee_scores, er_scores = interaction_map(
            hidden_states,
            entity_positions=entity_positions[0],
            relation_positions=relation_positions[0],
        )

        # Should have entity-relation interaction scores
        assert er_scores is not None
        # 2 entities x 2 relations
        assert er_scores.shape == (2, 2)


class TestInteractionDecoder:
    """Test InteractionDecoder component."""

    def test_initialization(self):
        """Test InteractionDecoder initialization."""
        decoder = InteractionDecoder(
            entity_threshold=0.5,
            relation_threshold=0.5,
            triple_threshold=0.5,
            max_triples=20,
        )

        assert decoder.entity_threshold == 0.5
        assert decoder.relation_threshold == 0.5
        assert decoder.triple_threshold == 0.5
        assert decoder.max_triples == 20

    def test_triple_decoding(self):
        """Test triple extraction from interaction scores."""
        decoder = InteractionDecoder(
            entity_threshold=0.5,
            relation_threshold=0.3,
            triple_threshold=0.3,
            max_triples=10,
        )

        # Create mock scores
        # 2 entities, 1 relation
        ee_scores = torch.tensor([[0.8]])  # High entity-entity interaction
        er_scores = torch.tensor([[0.9], [0.85]])  # Both entities participate in relation

        entity_labels = ["Apple Inc.", "Steve Jobs"]
        relation_types = ["Founded_By"]

        triples = decoder.forward(ee_scores, er_scores, entity_labels, relation_types)

        # Should extract at least one triple
        assert len(triples) > 0

        # Check triple format
        for triple in triples:
            assert len(triple) == 3  # (subject, relation, object)
            assert triple[0] in entity_labels
            assert triple[1] in relation_types
            assert triple[2] in entity_labels

    def test_max_triples_limit(self):
        """Test that decoder respects max triples limit."""
        decoder = InteractionDecoder(
            entity_threshold=0.1,
            relation_threshold=0.1,
            triple_threshold=0.1,
            max_triples=3,
        )

        # Create scores that would generate many triples
        num_entities = 5
        num_relations = 2

        ee_scores = torch.ones(num_entities, num_entities - 1) * 0.9
        er_scores = torch.ones(num_entities, num_relations) * 0.9

        entity_labels = [f"Entity_{i}" for i in range(num_entities)]
        relation_types = [f"Relation_{i}" for i in range(num_relations)]

        triples = decoder.forward(ee_scores, er_scores, entity_labels, relation_types)

        # Should not exceed max_triples
        assert len(triples) <= 3


class TestUniRelModel:
    """Test UniRelModel."""

    def test_initialization(self):
        """Test UniRelModel initialization."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            max_seq_length=128,
            relation_types=["Work_For", "Based_In"],
            entity_types=["PER", "ORG", "LOC"],
        )

        model = UniRelModel(config)

        assert model.config == config
        assert model.encoder is not None
        assert model.tokenizer is not None
        assert model.entity_tagger is not None
        assert model.interaction_map is not None
        assert model.triple_decoder is not None

    def test_relation_verbalization(self):
        """Test relation verbalization."""
        config = UniRelConfig(relation_types=["Work_For", "Based_In"])

        model = UniRelModel(config)
        verbalizations = model.verbalize_relations()

        assert len(verbalizations) == 2
        assert "work for" in verbalizations
        assert "based in" in verbalizations

    def test_forward_pass(self):
        """Test forward pass."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            max_seq_length=64,
            relation_types=["Work_For"],
            entity_types=["PER", "ORG"],
        )

        model = UniRelModel(config)

        # Create dummy input
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Run forward pass
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_triples=False,
        )

        # Check outputs
        assert "entity_logits" in outputs
        assert "ee_scores" in outputs
        assert "er_scores" in outputs
        assert outputs["entity_logits"] is not None

    def test_forward_with_labels(self):
        """Test forward pass with training labels."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            max_seq_length=64,
            relation_types=["Work_For"],
            entity_types=["PER", "ORG"],
        )

        model = UniRelModel(config)

        # Create dummy input and labels
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Entity labels (BIO tagging: 0=O, 1=B-PER, 2=I-PER, etc.)
        num_entity_tags = len(config.entity_types) * 3  # B-I-O for each type
        entity_labels = torch.randint(0, num_entity_tags, (batch_size, seq_len))

        # Run forward pass
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_labels=entity_labels,
        )

        # Should compute loss
        assert outputs["loss"] is not None
        assert outputs["loss"].requires_grad

    def test_triple_extraction(self):
        """Test triple extraction in inference mode."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            max_seq_length=128,
            relation_types=["Work_For", "Based_In"],
            entity_types=["PER", "ORG", "LOC"],
        )

        model = UniRelModel(config)
        model.eval()

        # Test with real text
        text = "Steve Jobs founded Apple Inc. in California."

        with torch.no_grad():
            triples = model.predict(text)

        # Should return a list (may be empty without training)
        assert isinstance(triples, list)


class TestCreateUniRelModel:
    """Test create_unirel_model factory function."""

    def test_create_model(self):
        """Test model creation via factory function."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            relation_types=["Work_For"],
            entity_types=["PER", "ORG"],
        )

        model = create_unirel_model(config)

        assert isinstance(model, UniRelModel)
        assert model.config == config


class TestUniRelIntegration:
    """Integration tests for UniRel."""

    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            max_seq_length=128,
            relation_types=["Founded_By", "Based_In"],
            entity_types=["PER", "ORG", "LOC"],
            triple_threshold=0.0,  # Very low threshold for testing
        )

        model = create_unirel_model(config)
        model.eval()

        # Test text
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

        # Extract triples
        with torch.no_grad():
            triples = model.predict(text)

        # Should return list (may be empty without training)
        assert isinstance(triples, list)

        # If triples found, verify format
        for triple in triples:
            assert len(triple) == 3
            subject, relation, obj = triple
            assert isinstance(subject, str)
            assert isinstance(relation, str)
            assert isinstance(obj, str)
            assert relation in config.relation_types

    def test_batch_processing(self):
        """Test batch processing of multiple texts."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            max_seq_length=128,
        )

        model = create_unirel_model(config)
        model.eval()

        texts = [
            "Apple Inc. was founded by Steve Jobs.",
            "Google is based in Mountain View.",
        ]

        all_triples = []
        for text in texts:
            with torch.no_grad():
                triples = model.predict(text)
            all_triples.append(triples)

        assert len(all_triples) == 2
        assert all(isinstance(t, list) for t in all_triples)

    def test_gradient_checkpointing(self):
        """Test model with gradient checkpointing enabled."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            gradient_checkpointing=True,
        )

        model = create_unirel_model(config)

        # Should initialize without errors
        assert model.config.gradient_checkpointing is True

    def test_overlapping_patterns(self):
        """Test handling of overlapping triple patterns."""
        config = UniRelConfig(
            encoder_model="bert-base-cased",
            handle_overlapping=True,
        )

        model = create_unirel_model(config)

        # Should support overlapping patterns
        assert model.config.handle_overlapping is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
