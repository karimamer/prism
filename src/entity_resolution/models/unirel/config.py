"""
Configuration for UniRel (Unified Representation and Interaction) model.

Based on: "UniRel: Unified Representation and Interaction for Joint Relational Triple Extraction"
"""

from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from ..base_config import BaseEncoderConfig, BaseThresholdConfig


class UniRelConfig(BaseEncoderConfig, BaseThresholdConfig):
    """
    Configuration for UniRel model.

    UniRel uses unified representation and interaction for joint entity and relation extraction.

    Key components:
    - Unified Representation: Encodes entities and relations in natural language sequences
    - Interaction Map: Models entity-entity and entity-relation interactions via self-attention
    - Joint triple extraction: Extracts <subject-relation-object> triples simultaneously

    Inherits from:
    - BaseEncoderConfig: Common encoder parameters (encoder_model, hidden_size)
    - BaseThresholdConfig: Confidence thresholds (span, entity, relation)
    """

    # Override encoder settings with UniRel-specific defaults
    encoder_model: str = Field(
        default="bert-base-cased",
        description="Pre-trained encoder model (BERT recommended for UniRel)",
        examples=["bert-base-cased", "bert-large-cased", "roberta-base"],
    )

    max_seq_length: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum sequence length for input text",
        examples=[256, 512, 1024],
    )

    hidden_size: int = Field(
        default=768,
        ge=128,
        le=1024,
        description="Hidden size of encoder (768 for BERT-base, 1024 for BERT-large)",
        examples=[768, 1024],
    )

    # Relation types and verbalization
    relation_types: List[str] = Field(
        default_factory=lambda: ["Work_For", "Based_In", "Located_In"],
        description="List of relation types to extract",
        min_length=1,
        examples=[["Work_For", "Based_In"], ["org:founded_by", "per:employee_of"]],
    )

    relation_verbalizations: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Mapping from relation type to natural language form. "
            "E.g., {'Work_For': 'work for', 'Based_In': 'based in'}. "
            "If None, auto-generated from relation_types."
        ),
        examples=[{"Work_For": "work for", "Based_In": "based in"}],
    )

    # Interaction Map settings
    interaction_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for interaction map (filters weak entity interactions)",
        examples=[0.3, 0.5, 0.7],
    )

    num_attention_heads: int = Field(
        default=12,
        ge=1,
        le=24,
        description="Number of attention heads for interaction map (should match encoder)",
        examples=[8, 12, 16],
    )

    interaction_dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate for interaction map",
        examples=[0.1, 0.2, 0.3],
    )

    # Entity extraction settings
    entity_types: List[str] = Field(
        default_factory=lambda: ["PER", "ORG", "LOC", "MISC"],
        description="List of entity types for extraction",
        min_length=1,
    )

    entity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for entity extraction",
        examples=[0.3, 0.5, 0.7],
    )

    # Triple extraction settings
    triple_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for triple extraction",
        examples=[0.3, 0.5, 0.7],
    )

    max_triples_per_sentence: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of triples to extract per sentence",
        examples=[10, 20, 50],
    )

    handle_overlapping: bool = Field(
        default=True,
        description=(
            "Handle overlapping triple patterns: "
            "SEO (Single Entity Overlap), EPO (Entity Pair Overlap), SOO (Subject Object Overlap)"
        ),
    )

    # Training settings
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate for model training",
        examples=[0.1, 0.2, 0.3],
    )

    learning_rate: float = Field(
        default=2e-5,
        ge=1e-6,
        le=1e-3,
        description="Learning rate for training",
        examples=[1e-5, 2e-5, 5e-5],
    )

    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Warmup ratio for learning rate scheduler",
        examples=[0.0, 0.1, 0.2],
    )

    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce memory usage",
    )

    # Loss weights for multi-task learning
    entity_loss_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for entity extraction loss in multi-task learning",
        examples=[0.5, 1.0, 2.0],
    )

    relation_loss_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for relation extraction loss in multi-task learning",
        examples=[0.5, 1.0, 2.0],
    )

    interaction_loss_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description="Weight for interaction map loss in multi-task learning",
        examples=[0.1, 0.5, 1.0],
    )

    # Decoding settings
    use_beam_search: bool = Field(
        default=False,
        description="Use beam search for triple decoding (slower but potentially more accurate)",
    )

    beam_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Beam size for beam search decoding",
        examples=[3, 5, 10],
    )

    @field_validator("encoder_model")
    @classmethod
    def validate_encoder_model(cls, v: str) -> str:
        """Ensure encoder model name is valid."""
        if not v or not v.strip():
            raise ValueError("encoder_model cannot be empty")

        # Warn if not using BERT (recommended for UniRel)
        if "bert" not in v.lower():
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"UniRel works best with BERT models, but got: {v}. "
                f"Consider using 'bert-base-cased' or 'bert-large-cased'."
            )

        return v.strip()

    @field_validator("num_attention_heads")
    @classmethod
    def validate_attention_heads(cls, v: int) -> int:
        """Validate attention heads match common encoder architectures."""
        valid_heads = {8, 12, 16, 24}  # Common values for base/large models

        if v not in valid_heads:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"num_attention_heads={v} is uncommon. "
                f"Standard values are: {sorted(valid_heads)}. "
                f"Ensure this matches your encoder model."
            )

        return v

    @model_validator(mode="after")
    def initialize_relation_verbalizations(self) -> "UniRelConfig":
        """Initialize relation verbalizations if not provided."""
        if self.relation_verbalizations is None:
            # Auto-generate: convert underscore to space and lowercase
            self.relation_verbalizations = {
                rel: rel.replace("_", " ").lower() for rel in self.relation_types
            }

        return self

    @model_validator(mode="after")
    def validate_relation_verbalizations_complete(self) -> "UniRelConfig":
        """Ensure all relation types have verbalizations."""
        if self.relation_verbalizations:
            missing_relations = set(self.relation_types) - set(self.relation_verbalizations.keys())

            if missing_relations:
                raise ValueError(
                    f"Missing verbalizations for relation types: {missing_relations}. "
                    f"All relation types must have corresponding verbalizations."
                )

        return self

    @model_validator(mode="after")
    def validate_loss_weights(self) -> "UniRelConfig":
        """Validate loss weights are reasonable."""
        total_weight = (
            self.entity_loss_weight + self.relation_loss_weight + self.interaction_loss_weight
        )

        if total_weight == 0.0:
            raise ValueError("At least one loss weight must be > 0")

        if total_weight > 20.0:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Large total loss weight: {total_weight:.2f}. Consider normalizing loss weights."
            )

        return self

    @model_validator(mode="after")
    def validate_beam_search_settings(self) -> "UniRelConfig":
        """Validate beam search configuration."""
        if self.use_beam_search and self.beam_size < 2:
            raise ValueError(
                f"beam_size must be >= 2 when use_beam_search=True, got {self.beam_size}"
            )

        if not self.use_beam_search and self.beam_size != 5:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"beam_size is set to {self.beam_size} but use_beam_search=False. "
                f"This setting will be ignored."
            )

        return self

    @model_validator(mode="after")
    def validate_hidden_size_attention_compatibility(self) -> "UniRelConfig":
        """Validate hidden size is divisible by number of attention heads."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        return self


__all__ = ["UniRelConfig"]
