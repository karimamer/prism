"""
ReLiK: Retrieve and LinK
Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget

Based on the paper:
"ReLiK: Retrieve and LinK, Fast and Accurate Entity Linking and Relation Extraction
on an Academic Budget" by Orlando et al. (2024)

This submodule provides a complete implementation of ReLiK for entity linking and
relation extraction using a Retriever-Reader architecture.

Key Components:
- ReLiKRetriever: Dense retrieval for candidate entities/relations
- ReLiKReader: Single-pass linking/extraction with contextualization
- ReLiKModel: Complete ReLiK model for EL and RE
- ReLiKConfig: Configuration for ReLiK components
"""

from .confidence_calibration import (
    ConfidenceCalibrator,
    PlattScaler,
    TemperatureScaler,
)
from .config import ReLiKConfig
from .dynamic_index import DynamicIndexManager
from .hard_negative_mining import HardNegativeMiner
from .linker import CompleteEntityLinker, create_entity_linker
from .reader_improved import ImprovedReLiKReader
from .relation_extractor import ReLiKRelationExtractor, create_relation_extractor
from .retriever import ReLiKRetriever
from .tokenizer import ReLiKTokenizer
from .unified_integration import ReLiKSystem, create_enhanced_relik_integration

__all__ = [
    "ReLiKConfig",
    "ReLiKRetriever",
    "ReLiKTokenizer",
    "ImprovedReLiKReader",
    "CompleteEntityLinker",
    "create_entity_linker",
    "ReLiKRelationExtractor",
    "create_relation_extractor",
    "HardNegativeMiner",
    "ConfidenceCalibrator",
    "TemperatureScaler",
    "PlattScaler",
    "DynamicIndexManager",
    "ReLiKSystem",
    "create_enhanced_relik_integration",
]

__version__ = "1.0.0"
__author__ = "PRISM Team (based on Orlando et al. 2024)"
