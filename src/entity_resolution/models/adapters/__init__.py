"""
Model adapters for pluggable architecture.

This package contains adapters for different entity resolution models,
all implementing the BaseModelAdapter interface.
"""

from entity_resolution.models.base_adapter import (
    BaseModelAdapter,
    ModelMetadata,
    ModelPrediction,
    ModelRegistry,
    register_adapter,
)

__all__ = [
    "BaseModelAdapter",
    "ModelMetadata",
    "ModelPrediction",
    "ModelRegistry",
    "register_adapter",
]
