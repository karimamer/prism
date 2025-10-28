"""
Enhanced ReLiK integration for UnifiedEntityResolutionSystem.

This module provides integration of all improved ReLiK features into
the unified system, including:
- Improved span detection with candidate encoding
- Dynamic knowledge base updates
- Relation extraction
- Confidence calibration
- Hard negative mining for training
"""

from typing import Any, Optional

import torch

from .confidence_calibration import ConfidenceCalibrator
from .dynamic_index import DynamicIndexManager
from .hard_negative_mining import HardNegativeMiner
from .linker import CompleteEntityLinker, create_entity_linker
from .reader_improved import ImprovedReLiKReader
from .relation_extractor import ReLiKRelationExtractor
from .retriever import ReLiKRetriever


class ReLiKSystem:
    """
    Enhanced ReLiK integration with all advanced features.

    Provides a unified interface for using improved ReLiK components
    in the entity resolution system.
    """

    def __init__(
        self,
        retriever_model: str = "microsoft/deberta-v3-small",
        reader_model: str = "microsoft/deberta-v3-base",
        enable_relation_extraction: bool = False,
        enable_calibration: bool = False,
        enable_dynamic_updates: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Initialize enhanced ReLiK integration.

        Args:
            retriever_model: Model for retrieval
            reader_model: Model for reading (uses ImprovedReLiKReader)
            enable_relation_extraction: Enable relation extraction
            enable_calibration: Enable confidence calibration
            enable_dynamic_updates: Enable dynamic KB updates
            device: Device to use
            **kwargs: Additional arguments
        """
        self.device = device
        self.enable_relation_extraction = enable_relation_extraction
        self.enable_calibration = enable_calibration
        self.enable_dynamic_updates = enable_dynamic_updates

        # Create retriever
        self.retriever = ReLiKRetriever(
            model_name=retriever_model,
            max_query_length=kwargs.get("max_query_length", 64),
            max_passage_length=kwargs.get("max_passage_length", 64),
            use_faiss=kwargs.get("use_faiss", True),
        )

        # Create improved reader (always use improved version)
        self.reader = ImprovedReLiKReader(
            model_name=reader_model,
            max_seq_length=kwargs.get("max_seq_length", 1024),
            num_entity_types=kwargs.get("num_entity_types", 4),
            dropout=kwargs.get("dropout", 0.1),
            use_entity_linking=True,
            use_relation_extraction=enable_relation_extraction,
            max_span_length=kwargs.get("max_span_length", 10),
        )

        # Create complete entity linker
        self.linker = CompleteEntityLinker(self.retriever, self.reader, device=device)

        # Optional: Relation extractor
        if enable_relation_extraction:
            self.relation_extractor = ReLiKRelationExtractor(self.reader)
        else:
            self.relation_extractor = None

        # Optional: Confidence calibrator
        if enable_calibration:
            self.calibrator = ConfidenceCalibrator(method="temperature")
        else:
            self.calibrator = None

        # Optional: Dynamic index manager
        if enable_dynamic_updates:
            self.dynamic_manager = DynamicIndexManager(
                self.retriever,
                rebuild_threshold=kwargs.get("rebuild_threshold", 1000),
                auto_rebuild=kwargs.get("auto_rebuild", True),
            )
        else:
            self.dynamic_manager = None

        # Hard negative miner (for training)
        self.hard_negative_miner: Optional[HardNegativeMiner] = None

        # Knowledge base reference
        self.knowledge_base: Optional[dict[str, dict[str, Any]]] = None

    def load_entities(
        self,
        entities: dict[str, dict[str, Any]],
        batch_size: int = 32,
    ):
        """
        Load entities into knowledge base with dynamic update support.

        Args:
            entities: Dictionary of entity_id -> entity_data
            batch_size: Batch size for encoding
        """
        self.knowledge_base = entities

        # Build retriever index
        self.retriever.build_index(entities, batch_size=batch_size)

        # Initialize hard negative miner if knowledge base is loaded
        if self.knowledge_base and len(self.knowledge_base) > 0:
            self.hard_negative_miner = HardNegativeMiner(
                self.retriever,
                self.knowledge_base,
                strategy="mixed",  # Balanced approach
                num_negatives=7,
            )

    def add_entity(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        immediate: bool = True,
    ):
        """
        Add entity with dynamic updates.

        Args:
            entity_id: Entity ID
            entity_data: Entity data
            immediate: Apply updates immediately
        """
        if not self.enable_dynamic_updates or self.dynamic_manager is None:
            raise RuntimeError("Dynamic updates not enabled")

        # Add to manager
        self.dynamic_manager.add_entity(entity_id, entity_data, immediate=immediate)

        # Update knowledge base
        if self.knowledge_base is not None:
            self.knowledge_base[entity_id] = entity_data

    def update_entity(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        immediate: bool = True,
    ):
        """Update existing entity."""
        if not self.enable_dynamic_updates or self.dynamic_manager is None:
            raise RuntimeError("Dynamic updates not enabled")

        self.dynamic_manager.update_entity(entity_id, entity_data, immediate=immediate)

        if self.knowledge_base is not None:
            self.knowledge_base[entity_id] = entity_data

    def remove_entity(
        self,
        entity_id: str,
        immediate: bool = True,
    ):
        """Remove entity."""
        if not self.enable_dynamic_updates or self.dynamic_manager is None:
            raise RuntimeError("Dynamic updates not enabled")

        self.dynamic_manager.remove_entity(entity_id, immediate=immediate)

        if self.knowledge_base is not None and entity_id in self.knowledge_base:
            del self.knowledge_base[entity_id]

    def process_text(
        self,
        text: str,
        top_k_retrieval: int = 100,
        top_k_linking: int = 10,
        span_threshold: float = 0.5,
        entity_threshold: float = 0.3,
        extract_relations: bool = False,
        relation_types: Optional[list[str]] = None,
        relation_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Process text with all enhanced features.

        Args:
            text: Input text
            top_k_retrieval: Top-k for retrieval
            top_k_linking: Top-k for linking
            span_threshold: Span detection threshold
            entity_threshold: Entity linking threshold
            extract_relations: Whether to extract relations
            relation_types: Relation types to extract
            relation_threshold: Relation extraction threshold

        Returns:
            Dictionary with entities and optionally relations
        """
        if self.knowledge_base is None:
            raise RuntimeError("Knowledge base not loaded")

        # Link entities
        entities = self.linker.link_entities_end_to_end(
            text,
            self.knowledge_base,
            top_k_retrieval=top_k_retrieval,
            top_k_linking=top_k_linking,
            span_threshold=span_threshold,
            entity_threshold=entity_threshold,
        )

        # Calibrate confidence scores if enabled
        if self.calibrator is not None and entities:
            entities = self._calibrate_entity_scores(entities)

        result = {
            "text": text,
            "entities": entities,
            "num_entities": len(entities),
        }

        # Extract relations if requested
        if extract_relations and self.relation_extractor is not None:
            if not relation_types:
                relation_types = ["founded by", "located in", "works at"]

            relations = self.relation_extractor.extract_relations(
                text,
                entities,
                relation_types,
                relation_threshold=relation_threshold,
            )

            # Calibrate relation scores if enabled
            if self.calibrator is not None and relations:
                relations = self._calibrate_relation_scores(relations)

            result["relations"] = relations
            result["num_relations"] = len(relations)

        return result

    def _calibrate_entity_scores(
        self,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Calibrate entity confidence scores."""
        if self.calibrator is None or self.calibrator.entity_calibrator is None:
            return entities

        for entity in entities:
            if "candidates" in entity:
                for cand in entity["candidates"]:
                    if "score" in cand:
                        original_score = torch.tensor([cand["score"]])
                        calibrated = self.calibrator.calibrate_entity_scores(original_score)
                        cand["score"] = calibrated.item()

            if "best_entity" in entity and "score" in entity["best_entity"]:
                original_score = torch.tensor([entity["best_entity"]["score"]])
                calibrated = self.calibrator.calibrate_entity_scores(original_score)
                entity["best_entity"]["score"] = calibrated.item()

        return entities

    def _calibrate_relation_scores(
        self,
        relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Calibrate relation confidence scores."""
        if self.calibrator is None or self.calibrator.relation_calibrator is None:
            return relations

        for relation in relations:
            if "confidence" in relation:
                original_score = torch.tensor([relation["confidence"]])
                calibrated = self.calibrator.calibrate_relation_scores(original_score)
                relation["confidence"] = calibrated.item()

        return relations

    def fit_calibrator(
        self,
        validation_data: dict[str, Any],
    ):
        """
        Fit confidence calibrator on validation data.

        Args:
            validation_data: Dictionary with 'entity_scores', 'entity_labels',
                           'relation_scores', 'relation_labels'
        """
        if not self.enable_calibration or self.calibrator is None:
            raise RuntimeError("Calibration not enabled")

        if "entity_scores" in validation_data and "entity_labels" in validation_data:
            self.calibrator.fit_entity_calibrator(
                validation_data["entity_scores"],
                validation_data["entity_labels"],
            )

        if "relation_scores" in validation_data and "relation_labels" in validation_data:
            self.calibrator.fit_relation_calibrator(
                validation_data["relation_scores"],
                validation_data["relation_labels"],
            )

    def get_training_batch(
        self,
        queries: list[str],
        positive_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Get training batch with hard negatives.

        Args:
            queries: Query texts
            positive_ids: Positive entity IDs

        Returns:
            Training batch dictionary
        """
        if self.hard_negative_miner is None:
            raise RuntimeError("Hard negative miner not initialized")

        return self.hard_negative_miner.prepare_training_batch(queries, positive_ids)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the system."""
        stats = {
            "knowledge_base_size": len(self.knowledge_base) if self.knowledge_base else 0,
            "use_improved_reader": self.use_improved_reader,
            "relation_extraction_enabled": self.enable_relation_extraction,
            "calibration_enabled": self.enable_calibration,
            "dynamic_updates_enabled": self.enable_dynamic_updates,
        }

        if self.dynamic_manager is not None:
            stats["dynamic_manager"] = self.dynamic_manager.get_statistics()

        return stats


def create_enhanced_relik_integration(
    config: dict[str, Any],
) -> ReLiKSystem:
    """
    Create enhanced ReLiK integration from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Enhanced ReLiK integration
    """
    # Set defaults for any missing parameters
    final_config = {
        "retriever_model": "microsoft/deberta-v3-small",
        "reader_model": "microsoft/deberta-v3-base",
        "enable_relation_extraction": False,
        "enable_calibration": False,
        "enable_dynamic_updates": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    # Update with provided config
    final_config.update(config)

    return ReLiKSystem(**final_config)


__all__ = ["ReLiKSystem", "create_enhanced_relik_integration"]
