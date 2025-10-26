"""
Prism Entity Resolution System.

A unified entity resolution system combining state-of-the-art techniques
from ReLiK, SPEL, UniRel, ATG, and OneNet.
"""

__version__ = "0.1.0"

# Output models
from entity_resolution.models.output import (
                                             EntityPrediction,
                                             EntityType,
                                             ModelStatus,
                                             RelationPrediction,
                                             UnifiedSystemOutput,
                                             create_unified_output,
)

# Core system
from entity_resolution.unified_system import UnifiedEntityResolutionSystem

# Configuration and validation
from entity_resolution.validation import (
                                             EntityCollection,
                                             EntityData,
                                             InputValidator,
                                             SystemConfig,
                                             validate_and_load_entities,
                                             validate_config,
)

__all__ = [
    # Version
    "__version__",
    # Core system
    "UnifiedEntityResolutionSystem",
    # Configuration
    "SystemConfig",
    "validate_config",
    # Validation
    "EntityData",
    "EntityCollection",
    "InputValidator",
    "validate_and_load_entities",
    # Output
    "UnifiedSystemOutput",
    "EntityPrediction",
    "RelationPrediction",
    "EntityType",
    "ModelStatus",
    "create_unified_output",
]
