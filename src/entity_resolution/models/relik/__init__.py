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

from .config import ReLiKConfig
from .model import ReLiKModel, create_relik_model
from .reader import ReLiKReader
from .retriever import ReLiKRetriever

__all__ = [
    "ReLiKConfig",
    "ReLiKRetriever",
    "ReLiKReader",
    "ReLiKModel",
    "create_relik_model",
]

__version__ = "1.0.0"
__author__ = "PRISM Team (based on Orlando et al. 2024)"
