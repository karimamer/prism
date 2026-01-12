"""
Base adapter interface for entity resolution models.

This module provides the abstract base class that all model adapters must implement,
enabling pluggable model architecture without modifying core system code.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from entity_resolution.models.output import EntityPrediction, RelationPrediction


# ============================================================================
# Model Metadata
# ============================================================================


class ModelMetadata(BaseModel):
    """Metadata about a model adapter."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (e.g., 'entity_linking', 're', 'joint')")
    description: str = Field(default="", description="Model description")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Model capabilities (e.g., ['entity_linking', 'relation_extraction'])",
    )
    required_inputs: list[str] = Field(
        default_factory=list, description="Required input fields"
    )
    optional_inputs: list[str] = Field(
        default_factory=list, description="Optional input fields"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ModelPrediction(BaseModel):
    """Normalized prediction output from a model adapter."""

    entities: list[EntityPrediction] = Field(default_factory=list, description="Entity predictions")
    relations: list[RelationPrediction] = Field(
        default_factory=list, description="Relation predictions"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall prediction confidence")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific metadata"
    )


# ============================================================================
# Base Adapter Interface
# ============================================================================


class BaseModelAdapter(ABC, nn.Module):
    """
    Abstract base class for all model adapters.

    This interface enables pluggable models by defining a standard contract
    that all model adapters must implement. Adding a new model only requires:
    1. Creating a new adapter class inheriting from BaseModelAdapter
    2. Implementing the abstract methods
    3. Registering the adapter with the system

    No changes to core system code are required.
    """

    def __init__(self, config: Any, device: Optional[torch.device] = None):
        """
        Initialize the model adapter.

        Args:
            config: Model-specific configuration (can be any type)
            device: Torch device for computation
        """
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")
        self._metadata: Optional[ModelMetadata] = None

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            ModelMetadata describing this model adapter
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional model-specific inputs

        Returns:
            Dictionary with model-specific outputs
        """
        pass

    @abstractmethod
    def predict(
        self,
        text: str,
        candidates: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelPrediction:
        """
        Make predictions on input text.

        This is the main interface method that returns normalized predictions.

        Args:
            text: Input text to process
            candidates: Optional list of candidate entities
            **kwargs: Additional model-specific parameters

        Returns:
            ModelPrediction with normalized entities and relations
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        texts: list[str],
        candidates: Optional[list[list[dict[str, Any]]]] = None,
        **kwargs,
    ) -> list[ModelPrediction]:
        """
        Make predictions on batch of texts.

        Args:
            texts: List of input texts
            candidates: Optional list of candidate lists (one per text)
            **kwargs: Additional model-specific parameters

        Returns:
            List of ModelPrediction objects
        """
        pass

    def validate_input(self, text: str) -> bool:
        """
        Validate input text.

        Args:
            text: Input text to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(text, str):
            raise TypeError(f"Input must be string, got {type(text)}")

        if not text.strip():
            raise ValueError("Input text cannot be empty")

        return True

    def to_device(self, device: torch.device):
        """
        Move model to specified device.

        Args:
            device: Target device
        """
        self.device = device
        self.to(device)

    def get_model_name(self) -> str:
        """
        Get model name.

        Returns:
            Model name string
        """
        metadata = self.get_metadata()
        return metadata.name

    def supports_capability(self, capability: str) -> bool:
        """
        Check if model supports a capability.

        Args:
            capability: Capability name (e.g., 'entity_linking')

        Returns:
            True if supported, False otherwise
        """
        metadata = self.get_metadata()
        return capability in metadata.capabilities

    def __repr__(self) -> str:
        """String representation."""
        metadata = self.get_metadata()
        return f"{self.__class__.__name__}(name={metadata.name}, version={metadata.version})"


# ============================================================================
# Model Registry
# ============================================================================


class ModelRegistry:
    """
    Registry for model adapters.

    This allows dynamic model discovery and instantiation.
    """

    _registry: dict[str, type[BaseModelAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[BaseModelAdapter]):
        """
        Register a model adapter.

        Args:
            name: Unique name for the adapter
            adapter_class: Adapter class (must inherit from BaseModelAdapter)

        Raises:
            ValueError: If adapter is already registered or invalid
        """
        if not issubclass(adapter_class, BaseModelAdapter):
            raise ValueError(f"{adapter_class} must inherit from BaseModelAdapter")

        if name in cls._registry:
            raise ValueError(f"Adapter '{name}' is already registered")

        cls._registry[name] = adapter_class

    @classmethod
    def get(cls, name: str) -> Optional[type[BaseModelAdapter]]:
        """
        Get registered adapter by name.

        Args:
            name: Adapter name

        Returns:
            Adapter class or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def list_adapters(cls) -> list[str]:
        """
        List all registered adapters.

        Returns:
            List of adapter names
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, config: Any, device: Optional[torch.device] = None) -> BaseModelAdapter:
        """
        Create adapter instance by name.

        Args:
            name: Adapter name
            config: Adapter configuration
            device: Torch device

        Returns:
            Instantiated adapter

        Raises:
            ValueError: If adapter not found
        """
        adapter_class = cls.get(name)
        if adapter_class is None:
            raise ValueError(
                f"Adapter '{name}' not found. Available: {cls.list_adapters()}"
            )

        return adapter_class(config=config, device=device)


# ============================================================================
# Decorator for easy registration
# ============================================================================


def register_adapter(name: str):
    """
    Decorator to register a model adapter.

    Usage:
        @register_adapter("my_model")
        class MyModelAdapter(BaseModelAdapter):
            ...
    """
    def decorator(adapter_class: type[BaseModelAdapter]):
        ModelRegistry.register(name, adapter_class)
        return adapter_class

    return decorator


__all__ = [
    "BaseModelAdapter",
    "ModelMetadata",
    "ModelPrediction",
    "ModelRegistry",
    "register_adapter",
]
