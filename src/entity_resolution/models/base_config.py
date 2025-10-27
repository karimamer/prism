"""
Base configuration classes for all entity resolution models.

This module provides base Pydantic models that are inherited by specific model configs
to reduce duplication and ensure consistency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)


class BaseModelConfig(BaseModel):
    """
    Base configuration shared by all entity resolution models.

    All model-specific configs should inherit from this to ensure consistency
    and reduce duplication of common parameters.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "title": "Base Model Configuration",
            "description": "Shared configuration for entity resolution models",
        },
    )

    # Common model parameters
    max_seq_length: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum sequence length for input text",
        examples=[256, 512, 1024],
    )

    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate for regularization",
        examples=[0.1, 0.2, 0.3],
    )

    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce memory usage during training",
    )

    # Entity and relation types
    entity_types: List[str] = Field(
        default_factory=lambda: ["PER", "ORG", "LOC", "MISC"],
        description="List of entity type labels",
        min_length=1,
    )

    relation_types: Optional[List[str]] = Field(
        default_factory=lambda: ["Work_For", "Based_In", "Located_In"],
        description="List of relation type labels (optional for entity-only models)",
    )

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, v: int) -> int:
        """Ensure max_seq_length is a reasonable value."""
        if v < 64:
            raise ValueError(f"max_seq_length too small: {v} (minimum 64)")
        if v > 2048:
            raise ValueError(f"max_seq_length too large: {v} (maximum 2048)")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        """Ensure dropout is within valid range."""
        if not 0.0 <= v <= 0.5:
            raise ValueError(f"dropout must be in [0.0, 0.5], got {v}")
        return v

    @field_validator("entity_types")
    @classmethod
    def validate_entity_types(cls, v: List[str]) -> List[str]:
        """Ensure entity types are valid."""
        if not v:
            raise ValueError("entity_types cannot be empty")
        # Remove duplicates while preserving order
        seen = set()
        unique_types = []
        for entity_type in v:
            if entity_type not in seen:
                seen.add(entity_type)
                unique_types.append(entity_type)
        return unique_types

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseModelConfig":
        """Create config from dictionary with validation."""
        return cls(**config_dict)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BaseModelConfig":
        """Load configuration from JSON file."""
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


class BaseEncoderConfig(BaseModelConfig):
    """
    Base configuration for models with an encoder component.

    Inherits from BaseModelConfig and adds encoder-specific parameters.
    """

    encoder_model: str = Field(
        default="microsoft/deberta-v3-base",
        description="HuggingFace model identifier for the encoder",
        examples=["microsoft/deberta-v3-base", "bert-base-cased", "roberta-base"],
    )

    hidden_size: int = Field(
        default=768,
        ge=128,
        le=1024,
        description="Hidden size of the encoder (768 for base, 1024 for large)",
        examples=[768, 1024],
    )

    @field_validator("encoder_model")
    @classmethod
    def validate_encoder_model(cls, v: str) -> str:
        """Validate encoder model name is not empty."""
        if not v or not v.strip():
            raise ValueError("encoder_model cannot be empty")
        return v.strip()


class BaseRetrieverConfig(BaseModelConfig):
    """
    Base configuration for models with a retrieval component.

    Inherits from BaseModelConfig and adds retriever-specific parameters.
    """

    retriever_model: str = Field(
        default="microsoft/deberta-v3-small",
        description="HuggingFace model identifier for the retriever",
        examples=["microsoft/deberta-v3-small", "sentence-transformers/all-MiniLM-L6-v2"],
    )

    top_k: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of top candidates to retrieve",
        examples=[50, 100, 200],
    )

    @field_validator("retriever_model")
    @classmethod
    def validate_retriever_model(cls, v: str) -> str:
        """Validate retriever model name is not empty."""
        if not v or not v.strip():
            raise ValueError("retriever_model cannot be empty")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k is reasonable."""
        if v < 1:
            raise ValueError(f"top_k must be >= 1, got {v}")
        if v > 1000:
            raise ValueError(f"top_k is very large: {v} (maximum recommended: 1000)")
        return v


class BaseThresholdConfig(BaseModel):
    """
    Base configuration for models that use confidence thresholds.

    This is a mixin-style config that can be used with other base configs.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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

    relation_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for relation extraction (if applicable)",
        examples=[0.3, 0.5, 0.7],
    )

    @model_validator(mode="after")
    def validate_thresholds_consistency(self) -> "BaseThresholdConfig":
        """Validate threshold relationships make sense."""
        if self.relation_threshold and self.entity_threshold:
            if self.relation_threshold < self.entity_threshold - 0.2:
                # Warn if relation threshold is much lower than entity threshold
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"relation_threshold ({self.relation_threshold}) is significantly "
                    f"lower than entity_threshold ({self.entity_threshold}). "
                    f"This may produce low-quality relations."
                )
        return self


__all__ = [
    "BaseModelConfig",
    "BaseEncoderConfig",
    "BaseRetrieverConfig",
    "BaseThresholdConfig",
]
