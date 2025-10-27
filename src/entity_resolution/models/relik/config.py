"""
Configuration for ReLiK model components.

ReLiK (Retrieval-based Entity Linking and Knowledge base completion) uses a
retriever-reader architecture for efficient entity linking and relation extraction.
"""

from pydantic import Field, field_validator, model_validator

from ..base_config import BaseRetrieverConfig, BaseThresholdConfig


class ReLiKConfig(BaseRetrieverConfig, BaseThresholdConfig):
    """
    Configuration for ReLiK (Retrieval-based Entity Linking) model.

    ReLiK uses a two-stage retriever-reader architecture:
    1. Retriever: Dense retrieval to find candidate entities/relations
    2. Reader: Span detection and entity/relation linking

    Inherits from:
    - BaseRetrieverConfig: Common retriever parameters (retriever_model, top_k)
    - BaseThresholdConfig: Confidence thresholds (span, entity, relation)
    """

    # Override retriever settings with ReLiK-specific defaults
    retriever_model: str = Field(
        default="microsoft/deberta-v3-small",
        description="Model name for the retriever encoder",
        examples=["microsoft/deberta-v3-small", "sentence-transformers/all-MiniLM-L6-v2"],
    )

    max_query_length: int = Field(
        default=64,
        ge=16,
        le=256,
        description="Maximum length for query sequences in retriever",
        examples=[32, 64, 128],
    )

    max_passage_length: int = Field(
        default=64,
        ge=16,
        le=256,
        description="Maximum length for passage sequences in retriever",
        examples=[32, 64, 128],
    )

    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of top candidates to retrieve from knowledge base",
        examples=[50, 100, 200],
    )

    # Reader settings
    reader_model: str = Field(
        default="microsoft/deberta-v3-base",
        description="Model name for the reader encoder",
        examples=["microsoft/deberta-v3-base", "bert-base-cased"],
    )

    max_seq_length: int = Field(
        default=1024,
        ge=128,
        le=2048,
        description="Maximum sequence length for reader (query + retrieved passages)",
        examples=[512, 1024, 2048],
    )

    max_entity_length: int = Field(
        default=128,
        ge=10,
        le=512,
        description="Maximum entity text length in reader",
        examples=[64, 128, 256],
    )

    # Entity Linking settings
    use_entity_linking: bool = Field(
        default=True, description="Enable entity linking task (detection + linking)"
    )

    num_el_passages: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of entity passages provided to reader for linking",
        examples=[50, 100, 200],
    )

    # Relation Extraction settings
    use_relation_extraction: bool = Field(
        default=False,
        description="Enable relation extraction task (currently experimental)",
    )

    num_re_passages: int = Field(
        default=24,
        ge=1,
        le=100,
        description="Number of relation passages provided to reader",
        examples=[12, 24, 48],
    )

    # Training settings
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate for regularization",
        examples=[0.1, 0.2, 0.3],
    )

    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce memory during training",
    )

    # Inference thresholds (inherited from BaseThresholdConfig)
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
        description="Confidence threshold for entity linking",
        examples=[0.3, 0.5, 0.7],
    )

    relation_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for relation extraction",
        examples=[0.3, 0.5, 0.7],
    )

    # Entity and relation types (inherited from BaseModelConfig)
    entity_types: list[str] = Field(
        default_factory=lambda: ["PER", "ORG", "LOC", "MISC"],
        description="List of entity type labels",
        min_length=1,
    )

    relation_types: list[str] = Field(
        default_factory=lambda: ["Work_For", "Based_In", "Located_In"],
        description="List of relation type labels",
        min_length=1,
    )

    @field_validator("reader_model", "retriever_model")
    @classmethod
    def validate_model_names(cls, v: str) -> str:
        """Ensure model names are not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("max_query_length", "max_passage_length")
    @classmethod
    def validate_length_parameters(cls, v: int) -> int:
        """Ensure length parameters are reasonable."""
        if v < 16:
            raise ValueError(f"Length parameter too small: {v} (minimum 16)")
        if v > 256:
            raise ValueError(f"Length parameter too large: {v} (maximum 256)")
        return v

    @model_validator(mode="after")
    def validate_passage_counts(self) -> "ReLiKConfig":
        """Validate that passage counts don't exceed top_k."""
        if self.num_el_passages > self.top_k:
            raise ValueError(
                f"num_el_passages ({self.num_el_passages}) cannot exceed top_k ({self.top_k})"
            )

        if self.use_relation_extraction and self.num_re_passages > self.top_k:
            raise ValueError(
                f"num_re_passages ({self.num_re_passages}) cannot exceed top_k ({self.top_k})"
            )

        return self

    @model_validator(mode="after")
    def validate_task_settings(self) -> "ReLiKConfig":
        """Validate at least one task is enabled."""
        if not self.use_entity_linking and not self.use_relation_extraction:
            raise ValueError(
                "At least one task must be enabled: use_entity_linking or use_relation_extraction"
            )

        return self

    @model_validator(mode="after")
    def validate_sequence_lengths(self) -> "ReLiKConfig":
        """Validate sequence length relationships."""
        if self.max_entity_length >= self.max_seq_length:
            raise ValueError(
                f"max_entity_length ({self.max_entity_length}) must be less than "
                f"max_seq_length ({self.max_seq_length})"
            )

        return self


__all__ = ["ReLiKConfig"]
