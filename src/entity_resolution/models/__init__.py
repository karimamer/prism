"""
Entity Resolution Models.

This package contains all model implementations including:
- ATG (Autoregressive Text-to-Graph)
- ReLiK (Retrieval-based Entity Linking)
- SPEL (Structured Prediction for Entity Linking)
- UniRel (Unified Representation Learning)
- Base components (retriever, reader, consensus)
"""

# Model-specific implementations
from entity_resolution.models.atg import ATGConfig, ImprovedATGModel
# Base configurations
from entity_resolution.models.base_config import (BaseEncoderConfig,
                                                  BaseModelConfig,
                                                  BaseRetrieverConfig,
                                                  BaseThresholdConfig)
from entity_resolution.models.candidate_generator import \
    EntityCandidateGenerator
from entity_resolution.models.consensus import ConsensusModule
from entity_resolution.models.entity_encoder import EntityFocusedEncoder
# Output formatting
from entity_resolution.models.output import (EntityOutputFormatter,
                                             EntityPrediction, EntityType,
                                             ModelStatus, RelationPrediction,
                                             UnifiedSystemOutput,
                                             create_unified_output)
from entity_resolution.models.reader import EntityReader
from entity_resolution.models.relik import ReLiKConfig, ReLiKModel
from entity_resolution.models.resolution_processor import \
    EntityResolutionProcessor
# Base components
from entity_resolution.models.retriever import EntityRetriever
from entity_resolution.models.spel import SPELConfig, SPELModel
from entity_resolution.models.unirel import UniRelConfig, UniRelModel

__all__ = [
    # Base components
    "EntityRetriever",
    "EntityReader",
    "ConsensusModule",
    "EntityFocusedEncoder",
    "EntityCandidateGenerator",
    "EntityResolutionProcessor",
    # ATG
    "ATGConfig",
    "ImprovedATGModel",
    # ReLiK
    "ReLiKConfig",
    "ReLiKModel",
    # SPEL
    "SPELConfig",
    "SPELModel",
    # UniRel
    "UniRelConfig",
    "UniRelModel",
    # Base configs
    "BaseModelConfig",
    "BaseEncoderConfig",
    "BaseRetrieverConfig",
    "BaseThresholdConfig",
    # Output
    "EntityOutputFormatter",
    "UnifiedSystemOutput",
    "EntityPrediction",
    "RelationPrediction",
    "EntityType",
    "ModelStatus",
    "create_unified_output",
]
