"""
Relation Extraction for ReLiK.

Implements relation extraction between detected entities using
relation type verbalization and triplet scoring.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from .reader_improved import ImprovedReLiKReader
from .tokenizer import ReLiKTokenizer


class ReLiKRelationExtractor(nn.Module):
    """
    Relation extraction between detected entities.

    Uses the ReLiK approach of encoding relation types as verbalizations
    and scoring (subject, relation, object) triplets.
    """

    def __init__(
        self,
        reader: ImprovedReLiKReader,
        relation_scorer: Optional[nn.Module] = None,
    ):
        """
        Initialize relation extractor.

        Args:
            reader: ReLiK reader for encoding
            relation_scorer: Optional custom scorer (creates default if None)
        """
        super().__init__()

        self.reader = reader
        self.tokenizer = reader.tokenizer

        # Create relation scorer if not provided
        if relation_scorer is None:
            self.relation_scorer = nn.Sequential(
                nn.Linear(reader.hidden_size * 3, reader.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(reader.hidden_size, 1),
            )
        else:
            self.relation_scorer = relation_scorer

    def extract_relations(
        self,
        text: str,
        entities: list[dict[str, Any]],
        relation_types: list[str],
        relation_threshold: float = 0.5,
        max_distance: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Extract relations between detected entities.

        Args:
            text: Input text
            entities: Detected entities with 'start', 'end', 'text' fields
            relation_types: List of possible relation type descriptions
            relation_threshold: Minimum confidence threshold
            max_distance: Maximum token distance between entities

        Returns:
            List of (subject, relation, object, score) triplets
        """
        if len(entities) < 2:
            return []

        # Encode text with relation types
        encoded = self.tokenizer.encode_with_relations(
            text,
            relation_types,
            max_length=self.reader.max_seq_length,
        )

        # Move to device
        device = next(self.reader.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        marker_positions = encoded["marker_positions"]
        text_end = encoded["text_end"]

        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = self.reader.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = outputs.last_hidden_state[0]  # [seq_len, hidden]

        # Map entities to token positions
        entity_embeddings = []
        for entity in entities:
            # Find tokens corresponding to entity span
            start_char = entity["start"]
            end_char = entity["end"]

            # Approximate character-to-token mapping
            # In practice, use offset_mapping for exact mapping
            approx_start_token = min(start_char // 5, text_end - 1)
            approx_end_token = min(end_char // 5, text_end - 1)

            # Get entity embedding (mean of span)
            if approx_start_token < text_end and approx_end_token < text_end:
                entity_emb = hidden_states[approx_start_token : approx_end_token + 1].mean(dim=0)
            else:
                entity_emb = hidden_states[0]  # Fallback to first token

            entity_embeddings.append(entity_emb)

        entity_embeddings = torch.stack(entity_embeddings)  # [num_entities, hidden]

        # Extract relation type embeddings
        relation_embeddings = []
        for i in range(len(relation_types)):
            marker_pos = marker_positions[i]
            if marker_pos >= 0 and marker_pos < len(hidden_states):
                # Get embedding at <Ri> marker position
                rel_emb = hidden_states[marker_pos]
                relation_embeddings.append(rel_emb)

        if not relation_embeddings:
            return []

        relation_embeddings = torch.stack(relation_embeddings)  # [num_relations, hidden]

        # Enumerate entity pairs and score relations
        extracted_relations = []

        for i, subj_entity in enumerate(entities):
            for j, obj_entity in enumerate(entities):
                if i == j:
                    continue

                # Check distance constraint
                distance = abs(subj_entity["start"] - obj_entity["start"])
                if distance > max_distance:
                    continue

                subj_emb = entity_embeddings[i]
                obj_emb = entity_embeddings[j]

                # Score each relation type
                for k, rel_type in enumerate(relation_types):
                    rel_emb = relation_embeddings[k]

                    # Combine representations: [subj, rel, obj]
                    combined = torch.cat([subj_emb, rel_emb, obj_emb])

                    # Score triplet
                    score = torch.sigmoid(self.relation_scorer(combined)).item()

                    if score >= relation_threshold:
                        extracted_relations.append(
                            {
                                "subject": {
                                    "text": subj_entity["text"],
                                    "start": subj_entity["start"],
                                    "end": subj_entity["end"],
                                    "entity_id": subj_entity.get("entity_id"),
                                },
                                "relation": rel_type,
                                "relation_score": score,
                                "object": {
                                    "text": obj_entity["text"],
                                    "start": obj_entity["start"],
                                    "end": obj_entity["end"],
                                    "entity_id": obj_entity.get("entity_id"),
                                },
                                "confidence": score,
                            }
                        )

        # Sort by confidence
        extracted_relations.sort(key=lambda x: x["confidence"], reverse=True)

        return extracted_relations

    def forward(
        self,
        text: str,
        entities: list[dict[str, Any]],
        relation_types: list[str],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Forward pass (alias for extract_relations)."""
        return self.extract_relations(text, entities, relation_types, **kwargs)


def create_relation_extractor(
    reader: ImprovedReLiKReader,
) -> ReLiKRelationExtractor:
    """
    Convenience function to create relation extractor.

    Args:
        reader: ReLiK reader to use

    Returns:
        Relation extractor
    """
    return ReLiKRelationExtractor(reader)


__all__ = ["ReLiKRelationExtractor", "create_relation_extractor"]
