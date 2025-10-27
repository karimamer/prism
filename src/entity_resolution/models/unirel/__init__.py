"""
UniRel: Unified Representation and Interaction for Joint Relational Triple Extraction.

This module implements the UniRel model from:
"UniRel: Unified Representation and Interaction for Joint Relational Triple Extraction"

Key components:
- UniRelConfig: Configuration for the model
- InteractionMap: Core component for modeling entity-entity and entity-relation interactions
- UniRelModel: Complete model for joint triple extraction
"""

from .config import UniRelConfig
from .interaction_map import InteractionDecoder, InteractionMap
from .model import UniRelModel, create_unirel_model

__all__ = [
    "UniRelConfig",
    "InteractionMap",
    "InteractionDecoder",
    "UniRelModel",
    "create_unirel_model",
]
