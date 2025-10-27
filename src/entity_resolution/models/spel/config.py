"""
Configuration for SPEL model.

SPEL (Structured Prediction for Entity Linking) uses token-level classification
with context-sensitive prediction aggregation for entity linking.
"""

from typing import List, Literal

from pydantic import Field, field_validator, model_validator

from ..base_config import BaseEncoderConfig, BaseThresholdConfig


class SPELConfig(BaseEncoderConfig, BaseThresholdConfig):
    """
    Configuration for SPEL (Structured Prediction for Entity Linking) model.

    SPEL uses structured prediction (token-level classification) for entity linking
    with context-sensitive prediction aggregation. It's particularly effective for
    fixed candidate sets (e.g., Wikipedia entities).

    Inherits from:
    - BaseEncoderConfig: Common encoder parameters (encoder_model, hidden_size)
    - BaseThresholdConfig: Confidence thresholds (span, entity, relation)
    """

    # Override encoder settings with SPEL-specific defaults
    encoder_model: str = Field(
        default="roberta-base",
        description="Pre-trained encoder model (RoBERTa recommended for SPEL)",
        examples=["roberta-base", "roberta-large", "bert-base-cased"],
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
        description="Hidden representation dimension (768 for base, 1024 for large)",
        examples=[768, 1024],
    )

    # Candidate set settings
    fixed_candidate_set_size: int = Field(
        default=500000,
        ge=1000,
        le=10000000,
        description="Size of fixed candidate set (e.g., 500K for Wikipedia)",
        examples=[100000, 500000, 1000000],
        json_schema_extra={"unit": "entities"},
    )

    use_mention_specific_candidates: bool = Field(
        default=False,
        description=(
            "Use mention-specific candidate sets instead of fixed set (more accurate but slower)"
        ),
    )

    candidate_set_type: Literal["fixed", "context_agnostic", "context_aware"] = Field(
        default="fixed",
        description="Type of candidate set to use for entity linking",
    )

    # Training settings
    num_hard_negatives: int = Field(
        default=5000,
        ge=0,
        le=100000,
        description="Number of hard negative examples per batch during training",
        examples=[1000, 5000, 10000],
    )

    num_random_negatives: int = Field(
        default=5000,
        ge=0,
        le=100000,
        description="Number of random negative examples per batch during training",
        examples=[1000, 5000, 10000],
    )

    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate for regularization",
        examples=[0.1, 0.2, 0.3],
    )

    gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing (recommended for SPEL due to large candidate sets)",
    )

    # Fine-tuning settings
    use_multi_step_finetuning: bool = Field(
        default=True,
        description=(
            "Use multi-step fine-tuning strategy: general → mention-agnostic → domain-specific"
        ),
    )

    freeze_layers: int = Field(
        default=4,
        ge=0,
        le=12,
        description="Number of bottom encoder layers to freeze during domain fine-tuning",
        examples=[0, 4, 8],
    )

    # Prediction aggregation settings
    span_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for span detection",
        examples=[0.3, 0.5, 0.7],
    )

    entity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for entity prediction",
        examples=[0.3, 0.5, 0.7],
    )

    use_context_sensitive_aggregation: bool = Field(
        default=True,
        description=(
            "Use context-sensitive prediction aggregation "
            "(improves accuracy for ambiguous mentions)"
        ),
    )

    filter_single_punctuation: bool = Field(
        default=True,
        description="Filter out single punctuation character spans",
    )

    filter_function_words: bool = Field(
        default=True,
        description="Filter out single function word spans (e.g., 'the', 'a', 'an')",
    )

    # Inference settings for long documents
    chunk_size: int = Field(
        default=254,
        ge=50,
        le=512,
        description="Size of text chunks in subwords for processing long documents",
        examples=[128, 254, 384],
        json_schema_extra={"unit": "subwords"},
    )

    chunk_overlap: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Overlap in subwords between consecutive chunks",
        examples=[10, 20, 50],
        json_schema_extra={"unit": "subwords"},
    )

    # Entity types (inherited from BaseModelConfig via BaseEncoderConfig)
    entity_types: List[str] = Field(
        default_factory=lambda: ["PER", "ORG", "LOC", "MISC"],
        description="List of entity type labels (for compatibility)",
        min_length=1,
    )

    @field_validator("encoder_model")
    @classmethod
    def validate_encoder_model(cls, v: str) -> str:
        """Ensure encoder model name is valid."""
        if not v or not v.strip():
            raise ValueError("encoder_model cannot be empty")

        # Warn if not using RoBERTa (recommended for SPEL)
        if "roberta" not in v.lower():
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"SPEL works best with RoBERTa models, but got: {v}. "
                f"Consider using 'roberta-base' or 'roberta-large'."
            )

        return v.strip()

    @field_validator("fixed_candidate_set_size")
    @classmethod
    def validate_candidate_set_size(cls, v: int) -> int:
        """Validate candidate set size is reasonable."""
        if v < 1000:
            raise ValueError(f"Candidate set too small: {v} (minimum 1000)")

        if v > 10000000:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Very large candidate set: {v:,}. This may cause memory issues. "
                f"Consider reducing to <= 1,000,000."
            )

        return v

    @field_validator("freeze_layers")
    @classmethod
    def validate_freeze_layers(cls, v: int) -> int:
        """Validate freeze_layers is reasonable."""
        if v < 0:
            raise ValueError(f"freeze_layers must be >= 0, got {v}")

        if v > 12:
            raise ValueError(f"freeze_layers too large: {v} (maximum 12 for most models)")

        return v

    @model_validator(mode="after")
    def validate_chunking_settings(self) -> "SPELConfig":
        """Validate chunk size and overlap make sense."""
        if self.chunk_size < self.chunk_overlap:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be >= chunk_overlap ({self.chunk_overlap})"
            )

        if self.chunk_overlap > self.chunk_size // 2:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Large overlap ({self.chunk_overlap}) relative to chunk_size "
                f"({self.chunk_size}). This may cause inefficiency."
            )

        return self

    @model_validator(mode="after")
    def validate_negative_samples(self) -> "SPELConfig":
        """Validate negative sample counts."""
        total_negatives = self.num_hard_negatives + self.num_random_negatives

        if total_negatives > 20000:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Very large number of negative samples: {total_negatives:,}. "
                f"This may slow down training significantly."
            )

        return self

    @model_validator(mode="after")
    def validate_candidate_set_consistency(self) -> "SPELConfig":
        """Validate candidate set configuration is consistent."""
        if self.use_mention_specific_candidates and self.candidate_set_type == "fixed":
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "use_mention_specific_candidates=True but candidate_set_type='fixed'. "
                "Consider setting candidate_set_type to 'context_agnostic' or 'context_aware'."
            )

        return self


__all__ = ["SPELConfig"]
