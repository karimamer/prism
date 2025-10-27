"""
SPEL: Structured Prediction for Entity Linking

Based on the paper:
"SPEL: Structured Prediction for Entity Linking" by Shavarani and Sarkar (2023)

This submodule provides a structured prediction approach for entity linking that:
- Uses token-level classification for entity prediction
- Implements context-sensitive prediction aggregation
- Handles candidate sets efficiently
- Addresses tokenization mismatch between training and inference

Key Components:
- SPELConfig: Configuration for SPEL model
- SPELModel: Complete SPEL model for entity linking
- CandidateSetManager: Manages fixed and mention-specific candidate sets
- PredictionAggregator: Aggregates subword predictions into spans
"""

from .aggregation import PredictionAggregator
from .candidate_sets import CandidateSetManager
from .config import SPELConfig
from .model import SPELModel, create_spel_model

__all__ = [
    "SPELConfig",
    "CandidateSetManager",
    "PredictionAggregator",
    "SPELModel",
    "create_spel_model",
]

__version__ = "1.0.0"
__author__ = "PRISM Team (based on Shavarani and Sarkar 2023)"
