"""
ReLiK Model: Complete Retrieve and LinK system.

Combines Retriever and Reader for end-to-end entity linking and relation extraction.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from .config import ReLiKConfig
from .reader import ReLiKReader
from .retriever import ReLiKRetriever


class ReLiKModel(nn.Module):
    """
    Complete ReLiK model for entity linking and relation extraction.

    Combines a Retriever (for candidate retrieval) and Reader (for linking/extraction)
    to perform end-to-end IE tasks.
    """

    def __init__(self, config: ReLiKConfig):
        """
        Initialize ReLiK model.

        Args:
            config: ReLiK configuration
        """
        super().__init__()

        self.config = config

        # Initialize Retriever
        self.retriever = ReLiKRetriever(
            model_name=config.retriever_model,
            max_query_length=config.max_query_length,
            max_passage_length=config.max_passage_length,
            use_faiss=True,
        )

        # Initialize Reader
        self.reader = ReLiKReader(
            model_name=config.reader_model,
            max_seq_length=config.max_seq_length,
            num_entity_types=len(config.entity_types),
            num_relation_types=len(config.relation_types),
            dropout=config.dropout,
            use_entity_linking=config.use_entity_linking,
            use_relation_extraction=config.use_relation_extraction,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        # Knowledge base (populated via load_entities)
        self.entity_kb = None
        self.relation_kb = None

    def load_entities(
        self,
        entities: dict[str, dict[str, Any]],
        batch_size: int = 32,
    ):
        """
        Load entity knowledge base and build retrieval index.

        Args:
            entities: Dictionary mapping entity IDs to entity data
                     Each entity should have 'text' (description) and optionally
                     'name', 'type', etc.
            batch_size: Batch size for encoding
        """
        self.entity_kb = entities

        # Build retriever index
        self.retriever.build_index(entities, batch_size=batch_size)

    def load_relations(
        self,
        relations: dict[str, dict[str, Any]],
    ):
        """
        Load relation types.

        Args:
            relations: Dictionary mapping relation IDs to relation data
        """
        self.relation_kb = relations

    def process_text(
        self,
        text: str,
        top_k: Optional[int] = None,
        return_candidates: bool = False,
    ) -> dict[str, Any]:
        """
        Process text for entity linking.

        Args:
            text: Input text
            top_k: Number of candidates to retrieve (default: from config)
            return_candidates: Whether to return retrieved candidates

        Returns:
            Dictionary with entities and optionally candidates
        """
        if self.entity_kb is None:
            raise RuntimeError("Entity KB not loaded. Call load_entities() first.")

        top_k = top_k or self.config.num_el_passages

        # Step 1: Retrieve candidates
        candidates = self.retriever.retrieve([text], top_k=top_k)[0]

        # Step 2: Link entities
        candidate_data = [c[1] for c in candidates]
        entities = self.reader.link_entities(
            text,
            candidate_data,
            entity_threshold=self.config.entity_threshold,
        )

        result = {
            "text": text,
            "entities": entities,
            "num_entities": len(entities),
        }

        if return_candidates:
            result["candidates"] = [
                {
                    "id": cid,
                    "data": cdata,
                    "score": score,
                }
                for cid, cdata, score in candidates
            ]

        return result

    def process_batch(
        self,
        texts: list[str],
        top_k: Optional[int] = None,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Process a batch of texts.

        Args:
            texts: List of input texts
            top_k: Number of candidates to retrieve
            batch_size: Batch size for processing

        Returns:
            List of results for each text
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            for text in batch:
                result = self.process_text(text, top_k=top_k)
                results.append(result)

        return results

    def extract_relations(
        self,
        text: str,
        top_k: Optional[int] = None,
        return_candidates: bool = False,
    ) -> dict[str, Any]:
        """
        Extract relations from text.

        Args:
            text: Input text
            top_k: Number of relation candidates to retrieve
            return_candidates: Whether to return retrieved candidates

        Returns:
            Dictionary with relations and optionally candidates
        """
        if not self.config.use_relation_extraction:
            raise RuntimeError("Relation extraction not enabled in config.")

        top_k = top_k or self.config.num_re_passages

        # Extract relations
        relations = self.reader.extract_relations(
            text,
            self.config.relation_types,
            relation_threshold=self.config.relation_threshold,
        )

        result = {
            "text": text,
            "relations": relations,
            "num_relations": len(relations),
        }

        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass (primarily for training).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments for reader

        Returns:
            Dictionary with model outputs
        """
        return self.reader(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save directory
        """
        import os

        os.makedirs(path, exist_ok=True)

        # Save retriever
        retriever_path = os.path.join(path, "retriever")
        os.makedirs(retriever_path, exist_ok=True)
        self.retriever.encoder.save_pretrained(retriever_path)
        self.retriever.tokenizer.save_pretrained(retriever_path)

        # Save reader
        reader_path = os.path.join(path, "reader")
        os.makedirs(reader_path, exist_ok=True)
        self.reader.encoder.save_pretrained(reader_path)
        self.reader.tokenizer.save_pretrained(reader_path)
        torch.save(self.reader.state_dict(), os.path.join(reader_path, "reader_weights.pt"))

        # Save config
        import json

        config_dict = {
            "retriever_model": self.config.retriever_model,
            "reader_model": self.config.reader_model,
            "max_seq_length": self.config.max_seq_length,
            "entity_types": self.config.entity_types,
            "relation_types": self.config.relation_types,
            "use_entity_linking": self.config.use_entity_linking,
            "use_relation_extraction": self.config.use_relation_extraction,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ReLiKModel":
        """
        Load model from path.

        Args:
            path: Path to model directory

        Returns:
            Loaded ReLiK model
        """
        import json
        import os

        # Load config
        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)

        config = ReLiKConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights (retriever and reader models are already loaded from pretrained)
        reader_weights_path = os.path.join(path, "reader", "reader_weights.pt")
        if os.path.exists(reader_weights_path):
            model.reader.load_state_dict(torch.load(reader_weights_path))

        return model


def create_relik_model(
    retriever_model: str = "microsoft/deberta-v3-small",
    reader_model: str = "microsoft/deberta-v3-base",
    entity_types: Optional[list[str]] = None,
    relation_types: Optional[list[str]] = None,
    use_entity_linking: bool = True,
    use_relation_extraction: bool = False,
    **kwargs,
) -> ReLiKModel:
    """
    Convenience function to create a ReLiK model.

    Args:
        retriever_model: Model name for retriever
        reader_model: Model name for reader
        entity_types: List of entity types
        relation_types: List of relation types
        use_entity_linking: Whether to enable entity linking
        use_relation_extraction: Whether to enable relation extraction
        **kwargs: Additional config parameters

    Returns:
        ReLiK model
    """
    config = ReLiKConfig(
        retriever_model=retriever_model,
        reader_model=reader_model,
        entity_types=entity_types or ["PER", "ORG", "LOC", "MISC"],
        relation_types=relation_types or ["Work_For", "Based_In", "Located_In"],
        use_entity_linking=use_entity_linking,
        use_relation_extraction=use_relation_extraction,
        **kwargs,
    )

    return ReLiKModel(config)
