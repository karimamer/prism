"""
Complete Entity Linking Pipeline for ReLiK.

Combines retriever and improved reader for end-to-end entity linking
with proper candidate ranking and confidence scores.
"""

from typing import Any

import torch
import torch.nn as nn

from .reader_improved import ImprovedReLiKReader
from .retriever import ReLiKRetriever


class CompleteEntityLinker(nn.Module):
    """
    Complete entity linking pipeline with retrieval and reading.

    Implements the full ReLiK pipeline:
    1. Retrieve candidate entities for the text (dense retrieval)
    2. Encode text with all candidates using special tokens
    3. Detect entity spans in the text
    4. Link spans to candidates using disambiguation
    5. Return top-k entities per span with confidence scores
    """

    def __init__(
        self,
        retriever: ReLiKRetriever,
        reader: ImprovedReLiKReader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize entity linker.

        Args:
            retriever: ReLiK retriever for candidate generation
            reader: Improved ReLiK reader for span detection and linking
            device: Device to run on
        """
        super().__init__()

        self.retriever = retriever
        self.reader = reader
        self.device = device

        # Move models to device
        self.retriever.to(device)
        self.reader.to(device)

    def link_entities_end_to_end(
        self,
        text: str,
        knowledge_base: dict[str, dict[str, Any]],
        top_k_retrieval: int = 100,
        top_k_linking: int = 10,
        span_threshold: float = 0.5,
        entity_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Complete entity linking pipeline.

        Args:
            text: Input text
            knowledge_base: Entity knowledge base (id -> entity data)
            top_k_retrieval: Number of candidates to retrieve
            top_k_linking: Number of top entities to return per span
            span_threshold: Threshold for span detection
            entity_threshold: Threshold for entity linking

        Returns:
            List of detected spans with linked entities:
            [
                {
                    "start": char_offset,
                    "end": char_offset,
                    "text": "span text",
                    "span_score": confidence,
                    "candidates": [
                        {
                            "entity_id": "Q123",
                            "entity_name": "Entity Name",
                            "score": confidence
                        },
                        ...
                    ],
                    "best_entity": {...}  # Highest scoring entity
                }
            ]
        """
        # Step 1: Retrieve candidates using retriever
        if not self.retriever.passage_index:
            raise RuntimeError("Retriever index not built. Call retriever.build_index() first.")

        retrieved = self.retriever.retrieve([text], top_k=top_k_retrieval)
        candidates_list = retrieved[0]  # Get results for first (only) query

        # Format candidates for reader
        candidates = []
        for cand_id, cand_data, retrieval_score in candidates_list:
            candidates.append(
                {
                    "id": cand_id,
                    "text": cand_data.get("text", ""),
                    "name": cand_data.get("name", cand_id),
                    "type": cand_data.get("type", ""),
                    "retrieval_score": retrieval_score,
                }
            )

        # Step 2: Use reader to detect spans and link to candidates
        linked_entities = self.reader.predict_spans_with_linking(
            text,
            candidates,
            span_threshold=span_threshold,
            entity_threshold=entity_threshold,
            top_k=top_k_linking,
        )

        return linked_entities

    def link_entities_batch(
        self,
        texts: list[str],
        knowledge_base: dict[str, dict[str, Any]],
        batch_size: int = 8,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """
        Link entities in a batch of texts.

        Args:
            texts: List of input texts
            knowledge_base: Entity knowledge base
            batch_size: Batch size for retrieval
            **kwargs: Additional arguments for link_entities_end_to_end

        Returns:
            List of results, one per input text
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Process each text (retrieval can be batched, but reading is sequential for now)
            batch_results = []
            for text in batch_texts:
                result = self.link_entities_end_to_end(text, knowledge_base, **kwargs)
                batch_results.append(result)

            all_results.extend(batch_results)

        return all_results

    def forward(
        self,
        text: str,
        knowledge_base: dict[str, dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Forward pass (alias for link_entities_end_to_end).

        Args:
            text: Input text
            knowledge_base: Entity knowledge base
            **kwargs: Additional arguments

        Returns:
            Linked entities
        """
        return self.link_entities_end_to_end(text, knowledge_base, **kwargs)

    def eval(self):
        """Set to evaluation mode."""
        self.retriever.eval()
        self.reader.eval()
        return super().eval()

    def train(self, mode: bool = True):
        """Set to training mode."""
        self.retriever.train(mode)
        self.reader.train(mode)
        return super().train(mode)


def create_entity_linker(
    retriever_model: str = "microsoft/deberta-v3-small",
    reader_model: str = "microsoft/deberta-v3-base",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> CompleteEntityLinker:
    """
    Convenience function to create entity linker.

    Args:
        retriever_model: Model name for retriever
        reader_model: Model name for reader
        device: Device to run on
        **kwargs: Additional arguments for components

    Returns:
        Complete entity linker
    """
    # Create retriever
    retriever = ReLiKRetriever(
        model_name=retriever_model,
        max_query_length=kwargs.get("max_query_length", 64),
        max_passage_length=kwargs.get("max_passage_length", 64),
        use_faiss=kwargs.get("use_faiss", True),
    )

    # Create reader
    reader = ImprovedReLiKReader(
        model_name=reader_model,
        max_seq_length=kwargs.get("max_seq_length", 1024),
        num_entity_types=kwargs.get("num_entity_types", 4),
        dropout=kwargs.get("dropout", 0.1),
        use_entity_linking=True,
        use_relation_extraction=False,
    )

    # Create linker
    linker = CompleteEntityLinker(retriever, reader, device=device)

    return linker


__all__ = ["CompleteEntityLinker", "create_entity_linker"]
