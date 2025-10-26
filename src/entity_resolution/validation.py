"""
Comprehensive input validation module using Pydantic for type safety and data validation.

This module provides validation for:
- System configuration
- Entity data (JSON/CSV)
- Model inputs
- File sizes and formats
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Models
# ============================================================================


class EntityEncoderConfig(BaseModel):
    """Configuration for the entity-focused encoder component."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_name: str = Field(
        default="microsoft/deberta-v3-base",
        description="HuggingFace model identifier for the entity encoder",
    )
    entity_knowledge_dim: int = Field(
        default=256, ge=64, le=1024, description="Entity knowledge projection dimension"
    )
    num_entity_types: int = Field(default=50, ge=1, le=200, description="Number of entity types")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout probability")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class CandidateGeneratorConfig(BaseModel):
    """Configuration for the candidate generator component."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    embedding_dim: int = Field(default=768, ge=128, le=2048, description="Embedding dimension")
    num_bio_tags: int = Field(default=3, ge=2, le=10, description="Number of BIO tags")
    top_k: int = Field(default=100, ge=1, le=1000, description="Number of candidates to retrieve")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout probability")


class ResolutionProcessorConfig(BaseModel):
    """Configuration for the resolution processor component."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    encoder_dim: int = Field(default=768, ge=128, le=2048, description="Encoder dimension")
    num_heads: int = Field(default=8, ge=1, le=16, description="Number of attention heads")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout probability")


class RetrieverConfig(BaseModel):
    """Configuration for the retriever component."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_name: str = Field(
        default="microsoft/deberta-v3-small",
        description="HuggingFace model identifier for the retriever",
    )
    use_faiss: bool = Field(default=True, description="Whether to use FAISS index")
    top_k: int = Field(default=100, ge=1, le=1000, description="Number of candidates to retrieve")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size for encoding")
    max_length: int = Field(default=128, ge=32, le=512, description="Maximum sequence length")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class ReaderConfig(BaseModel):
    """Configuration for the reader component."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_name: str = Field(
        default="microsoft/deberta-v3-base",
        description="HuggingFace model identifier for the reader",
    )
    max_seq_length: int = Field(default=512, ge=64, le=2048, description="Maximum sequence length")
    max_entity_length: int = Field(
        default=100, ge=10, le=512, description="Maximum entity text length"
    )
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing for memory efficiency"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class ConsensusConfig(BaseModel):
    """Configuration for the consensus module."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for consensus",
    )
    use_auxiliary_judge: bool = Field(
        default=True, description="Use auxiliary judge for conflict resolution"
    )


class SystemConfig(BaseModel):
    """Complete system configuration with validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Component configurations
    retriever_model: str = Field(
        default="microsoft/deberta-v3-small",
        description="Retriever model name",
    )
    reader_model: str = Field(default="microsoft/deberta-v3-base", description="Reader model name")
    entity_dim: int = Field(default=256, ge=64, le=2048, description="Entity embedding dimension")
    max_seq_length: int = Field(default=512, ge=64, le=2048, description="Maximum sequence length")
    max_entity_length: int = Field(default=100, ge=10, le=512, description="Maximum entity length")
    top_k_candidates: int = Field(
        default=50, ge=1, le=500, description="Number of candidate entities"
    )
    consensus_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Consensus threshold"
    )
    batch_size: int = Field(default=8, ge=1, le=128, description="Processing batch size")

    # New component flags
    use_entity_encoder: bool = Field(
        default=False, description="Use EntityFocusedEncoder for entity-aware encoding"
    )
    use_candidate_generator: bool = Field(
        default=False, description="Use unified EntityCandidateGenerator"
    )
    use_resolution_processor: bool = Field(
        default=False, description="Use EntityResolutionProcessor with cross-model attention"
    )

    # New component configurations
    entity_encoder_dim: int = Field(
        default=256, ge=64, le=1024, description="Entity knowledge dimension for encoder"
    )
    num_entity_types: int = Field(default=50, ge=1, le=200, description="Number of entity types")
    num_attention_heads: int = Field(
        default=8, ge=1, le=16, description="Number of attention heads"
    )
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout probability")

    # Paths
    index_path: str = Field(default="./entity_index", description="Entity index path")
    cache_dir: str = Field(default="./cache", description="Cache directory")

    # Hardware
    use_gpu: bool = Field(default=True, description="Use GPU if available")
    quantization: Optional[Literal["int8", "fp16"]] = Field(
        default=None, description="Quantization type"
    )

    # Advanced settings
    gradient_checkpointing: bool = Field(default=True, description="Use gradient checkpointing")
    mixed_precision: bool = Field(default=False, description="Use mixed precision training")

    @model_validator(mode="after")
    def validate_paths(self) -> "SystemConfig":
        """Validate and create necessary paths."""
        # Create directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Cache directory ensured at: {self.cache_dir}")
        return self

    @model_validator(mode="after")
    def validate_gpu_settings(self) -> "SystemConfig":
        """Validate GPU settings."""
        if self.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available, falling back to CPU")
            self.use_gpu = False
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SystemConfig":
        """Create from dictionary with validation."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "SystemConfig":
        """Load configuration from JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        with open(path) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to: {json_path}")


# ============================================================================
# Entity Data Models
# ============================================================================


class EntityData(BaseModel):
    """Validation model for entity data."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    id: str = Field(..., min_length=1, description="Unique entity identifier")
    name: str = Field(..., min_length=1, description="Entity name")
    description: Optional[str] = Field(default="", description="Entity description")
    aliases: Optional[list[str]] = Field(default_factory=list, description="Entity aliases")
    entity_type: Optional[str] = Field(default="UNKNOWN", description="Entity type")

    @field_validator("id", "name")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure ID and name are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v.strip()

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, v: Optional[list[str]]) -> list[str]:
        """Validate aliases are non-empty strings."""
        if v is None:
            return []
        # Filter out empty aliases
        return [alias.strip() for alias in v if alias and alias.strip()]


class EntityCollection(BaseModel):
    """Collection of entities with validation."""

    model_config = ConfigDict(validate_assignment=True)

    entities: list[EntityData] = Field(default_factory=list)

    @field_validator("entities")
    @classmethod
    def validate_unique_ids(cls, v: list[EntityData]) -> list[EntityData]:
        """Ensure all entity IDs are unique."""
        ids = [entity.id for entity in v]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate entity IDs found: {set(duplicates)}")
        return v

    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> "EntityCollection":
        """Load entities from JSON file with validation."""
        path = Path(file_path)

        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"Entity file not found: {file_path}")

        # Check file size (max 100MB)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            raise ValueError(f"Entity file too large: {file_size_mb:.2f}MB (max 100MB)")

        # Load and parse JSON
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in entity file: {e}") from e

        # Handle different JSON formats
        if isinstance(data, list):
            entities_data = data
        elif isinstance(data, dict) and "entities" in data:
            entities_data = data["entities"]
        elif isinstance(data, dict):
            # Handle dict with entity IDs as keys: {"Q312": {...}, "Q19837": {...}}
            entities_data = list(data.values())
        else:
            raise ValueError("Entity file must be a list of entities or dict with 'entities' key")

        # Validate each entity
        entities = []
        errors = []

        for i, entity_dict in enumerate(entities_data):
            try:
                entity = EntityData(**entity_dict)
                entities.append(entity)
            except ValidationError as e:
                errors.append(f"Entity {i}: {e}")

        if errors:
            error_msg = "\n".join(errors[:5])  # Show first 5 errors
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            raise ValueError(f"Entity validation errors:\n{error_msg}")

        logger.info(f"Loaded {len(entities)} entities from {file_path}")
        return cls(entities=entities)

    @classmethod
    def from_csv_file(cls, file_path: Union[str, Path]) -> "EntityCollection":
        """Load entities from CSV file with validation."""
        path = Path(file_path)

        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"Entity file not found: {file_path}")

        # Check file size (max 100MB)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            raise ValueError(f"Entity file too large: {file_size_mb:.2f}MB (max 100MB)")

        # Load CSV
        entities = []
        errors = []

        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    try:
                        # Parse aliases if present
                        if "aliases" in row and row["aliases"]:
                            row["aliases"] = [a.strip() for a in row["aliases"].split(";")]

                        entity = EntityData(**row)
                        entities.append(entity)
                    except ValidationError as e:
                        errors.append(f"Row {i + 1}: {e}")

        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}") from e

        if errors:
            error_msg = "\n".join(errors[:5])  # Show first 5 errors
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            raise ValueError(f"CSV validation errors:\n{error_msg}")

        logger.info(f"Loaded {len(entities)} entities from {file_path}")
        return cls(entities=entities)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [entity.model_dump() for entity in self.entities]


# ============================================================================
# Input Validation Functions
# ============================================================================


class InputValidator:
    """Validator for various input types."""

    @staticmethod
    def validate_text_input(text: str, max_length: int = 10000, min_length: int = 1) -> str:
        """
        Validate text input.

        Args:
            text: Input text
            max_length: Maximum allowed length
            min_length: Minimum allowed length

        Returns:
            Validated and cleaned text

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise TypeError(f"Text must be a string, got {type(text)}")

        text = text.strip()

        if len(text) < min_length:
            raise ValueError(f"Text too short: {len(text)} chars (min {min_length})")

        if len(text) > max_length:
            raise ValueError(f"Text too long: {len(text)} chars (max {max_length})")

        return text

    @staticmethod
    def validate_batch_texts(texts: list[str], max_batch_size: int = 128) -> list[str]:
        """
        Validate a batch of texts.

        Args:
            texts: List of text strings
            max_batch_size: Maximum batch size

        Returns:
            Validated list of texts

        Raises:
            ValueError: If batch is invalid
        """
        if not isinstance(texts, list):
            raise TypeError(f"Texts must be a list, got {type(texts)}")

        if len(texts) == 0:
            raise ValueError("Empty batch provided")

        if len(texts) > max_batch_size:
            raise ValueError(f"Batch too large: {len(texts)} (max {max_batch_size})")

        # Validate each text
        validated = []
        for i, text in enumerate(texts):
            try:
                validated.append(InputValidator.validate_text_input(text))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid text at index {i}: {e}") from e

        return validated

    @staticmethod
    def validate_model_path(path: Union[str, Path], check_exists: bool = True) -> Path:
        """
        Validate model path.

        Args:
            path: Path to model directory or file
            check_exists: Whether to check if path exists

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is invalid
        """
        if not path:
            raise ValueError("Model path cannot be empty")

        path_obj = Path(path)

        if check_exists and not path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")

        return path_obj

    @staticmethod
    def validate_file_size(file_path: Union[str, Path], max_size_mb: float = 500) -> None:
        """
        Validate file size before loading.

        Args:
            file_path: Path to file
            max_size_mb: Maximum file size in MB

        Raises:
            ValueError: If file is too large
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        size_mb = path.stat().st_size / (1024 * 1024)

        if size_mb > max_size_mb:
            raise ValueError(f"File too large: {size_mb:.2f}MB (max {max_size_mb}MB)")

        logger.debug(f"File size OK: {size_mb:.2f}MB")

    @staticmethod
    def validate_entity_file(file_path: Union[str, Path]) -> EntityCollection:
        """
        Validate and load entity file (JSON or CSV).

        Args:
            file_path: Path to entity file

        Returns:
            Validated EntityCollection

        Raises:
            ValueError: If file is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Entity file not found: {file_path}")

        # Check file extension
        suffix = path.suffix.lower()

        if suffix == ".json":
            return EntityCollection.from_json_file(path)
        elif suffix == ".csv":
            return EntityCollection.from_csv_file(path)
        else:
            raise ValueError(f"Unsupported entity file format: {suffix}. Use .json or .csv")

    @staticmethod
    def validate_confidence_score(score: float) -> float:
        """
        Validate confidence score.

        Args:
            score: Confidence score

        Returns:
            Validated score

        Raises:
            ValueError: If score is invalid
        """
        if not isinstance(score, (int, float)):
            raise TypeError(f"Score must be numeric, got {type(score)}")

        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {score}")

        return float(score)


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_config(config: Union[dict[str, Any], SystemConfig]) -> SystemConfig:
    """
    Validate system configuration.

    Args:
        config: Configuration dictionary or SystemConfig object

    Returns:
        Validated SystemConfig object

    Raises:
        ValidationError: If configuration is invalid
    """
    if isinstance(config, SystemConfig):
        return config

    try:
        return SystemConfig.from_dict(config)
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def validate_and_load_entities(
    file_path: Union[str, Path], max_entities: Optional[int] = None
) -> list[dict[str, Any]]:
    """
    Validate and load entities from file.

    Args:
        file_path: Path to entity file
        max_entities: Maximum number of entities to load

    Returns:
        List of validated entity dictionaries

    Raises:
        ValueError: If entities are invalid
    """
    entity_collection = InputValidator.validate_entity_file(file_path)

    if max_entities and len(entity_collection.entities) > max_entities:
        logger.warning(
            f"Loaded {len(entity_collection.entities)} entities, "
            f"but max_entities is {max_entities}. Truncating."
        )
        entity_collection.entities = entity_collection.entities[:max_entities]

    return entity_collection.to_dict_list()


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "SystemConfig",
    "RetrieverConfig",
    "ReaderConfig",
    "ConsensusConfig",
    "EntityData",
    "EntityCollection",
    "InputValidator",
    "validate_config",
    "validate_and_load_entities",
]
