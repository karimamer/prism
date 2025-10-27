"""
Candidate set management for SPEL.

Manages fixed and mention-specific candidate sets for entity linking.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CandidateSetManager:
    """
    Manages candidate sets for entity linking.

    Supports three types of candidate sets:
    1. Fixed candidate set: K most frequent entities
    2. Context-agnostic mention-specific: Pre-computed candidates per mention surface form
    3. Context-aware mention-specific: Candidates depend on context
    """

    def __init__(
        self,
        fixed_candidates: Optional[List[str]] = None,
        mention_candidates: Optional[Dict[str, List[str]]] = None,
        entity_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize candidate set manager.

        Args:
            fixed_candidates: List of entity IDs in fixed candidate set
            mention_candidates: Dict mapping mention text to list of candidate entity IDs
            entity_to_idx: Dict mapping entity ID to vocabulary index
        """
        self.fixed_candidates = set(fixed_candidates) if fixed_candidates else set()
        self.mention_candidates = mention_candidates or {}
        self.entity_to_idx = entity_to_idx or {}
        self.idx_to_entity = {idx: ent for ent, idx in self.entity_to_idx.items()}

        # Add special O (non-entity) token
        if "O" not in self.entity_to_idx:
            self.entity_to_idx["O"] = len(self.entity_to_idx)
            self.idx_to_entity[len(self.idx_to_entity)] = "O"

    def get_candidates_for_mention(
        self,
        mention_text: str,
        use_mention_specific: bool = False,
    ) -> Set[str]:
        """
        Get candidate entities for a mention.

        Args:
            mention_text: Surface form of the mention
            use_mention_specific: Whether to use mention-specific candidates

        Returns:
            Set of candidate entity IDs
        """
        if use_mention_specific and mention_text in self.mention_candidates:
            # Return mention-specific candidates
            mention_cands = set(self.mention_candidates[mention_text])
            # Intersect with fixed candidates for safety
            return mention_cands & self.fixed_candidates if self.fixed_candidates else mention_cands
        else:
            # Return all fixed candidates
            return self.fixed_candidates

    def filter_predictions(
        self,
        mention_text: str,
        predicted_entities: List[Tuple[str, float]],
        use_mention_specific: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Filter predicted entities using candidate set.

        Args:
            mention_text: Surface form of the mention
            predicted_entities: List of (entity_id, probability) tuples
            use_mention_specific: Whether to use mention-specific candidates

        Returns:
            Filtered list of (entity_id, probability) tuples
        """
        if not use_mention_specific:
            # No filtering needed with fixed candidate set
            return predicted_entities

        # Get valid candidates for this mention
        valid_candidates = self.get_candidates_for_mention(mention_text, use_mention_specific=True)

        if not valid_candidates:
            # No candidates found, return original predictions
            return predicted_entities

        # Filter predictions
        filtered = [
            (ent, prob) for ent, prob in predicted_entities if ent in valid_candidates or ent == "O"
        ]

        return filtered if filtered else predicted_entities

    def add_fixed_candidates(self, entity_ids: List[str]):
        """
        Add entities to fixed candidate set.

        Args:
            entity_ids: List of entity IDs to add
        """
        for ent_id in entity_ids:
            self.fixed_candidates.add(ent_id)
            if ent_id not in self.entity_to_idx:
                idx = len(self.entity_to_idx)
                self.entity_to_idx[ent_id] = idx
                self.idx_to_entity[idx] = ent_id

    def add_mention_candidates(self, mention_text: str, entity_ids: List[str]):
        """
        Add mention-specific candidates.

        Args:
            mention_text: Mention surface form
            entity_ids: List of candidate entity IDs
        """
        if mention_text not in self.mention_candidates:
            self.mention_candidates[mention_text] = []

        self.mention_candidates[mention_text].extend(entity_ids)

        # Also add to fixed candidates to ensure they're in vocabulary
        self.add_fixed_candidates(entity_ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size (number of entities + O token)."""
        return len(self.entity_to_idx)

    def get_entity_idx(self, entity_id: str) -> Optional[int]:
        """Get vocabulary index for entity ID."""
        return self.entity_to_idx.get(entity_id)

    def get_entity_from_idx(self, idx: int) -> Optional[str]:
        """Get entity ID from vocabulary index."""
        return self.idx_to_entity.get(idx)

    def build_from_frequency(
        self,
        entity_frequencies: Dict[str, int],
        top_k: int = 500000,
    ):
        """
        Build fixed candidate set from entity frequencies.

        Args:
            entity_frequencies: Dict mapping entity ID to frequency count
            top_k: Number of top entities to include
        """
        # Sort by frequency and take top-k
        sorted_entities = sorted(entity_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_entities = [ent for ent, freq in sorted_entities[:top_k]]

        self.add_fixed_candidates(top_entities)

        logger.info(f"Built fixed candidate set with {len(top_entities)} entities")

    def save(self, path: str):
        """Save candidate sets to file."""
        import json
        import os

        os.makedirs(path, exist_ok=True)

        # Save fixed candidates
        with open(os.path.join(path, "fixed_candidates.json"), "w") as f:
            json.dump(list(self.fixed_candidates), f)

        # Save mention candidates
        with open(os.path.join(path, "mention_candidates.json"), "w") as f:
            json.dump(self.mention_candidates, f)

        # Save entity vocabulary
        with open(os.path.join(path, "entity_vocab.json"), "w") as f:
            json.dump(self.entity_to_idx, f)

        logger.info(f"Saved candidate sets to {path}")

    @classmethod
    def load(cls, path: str) -> "CandidateSetManager":
        """Load candidate sets from file."""
        import json
        import os

        # Load fixed candidates
        with open(os.path.join(path, "fixed_candidates.json"), "r") as f:
            fixed_candidates = json.load(f)

        # Load mention candidates
        mention_path = os.path.join(path, "mention_candidates.json")
        if os.path.exists(mention_path):
            with open(mention_path, "r") as f:
                mention_candidates = json.load(f)
        else:
            mention_candidates = {}

        # Load entity vocabulary
        with open(os.path.join(path, "entity_vocab.json"), "r") as f:
            entity_to_idx = json.load(f)

        logger.info(f"Loaded candidate sets from {path}")

        return cls(
            fixed_candidates=fixed_candidates,
            mention_candidates=mention_candidates,
            entity_to_idx=entity_to_idx,
        )
