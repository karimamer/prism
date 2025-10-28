"""
Dynamic Index Updates for ReLiK Retriever.

Supports incremental updates to the retrieval index without
full rebuilding, enabling efficient handling of evolving knowledge bases.
"""

from typing import Any, Optional

import torch
import numpy as np

from .retriever import ReLiKRetriever


class DynamicIndexManager:
    """
    Manages dynamic updates to the retriever index.

    Supports:
    1. Adding new entities
    2. Removing entities
    3. Updating entity descriptions
    4. Batch updates
    5. Periodic reindexing for optimization
    """

    def __init__(
        self,
        retriever: ReLiKRetriever,
        rebuild_threshold: int = 1000,
        auto_rebuild: bool = True,
    ):
        """
        Initialize dynamic index manager.

        Args:
            retriever: ReLiK retriever to manage
            rebuild_threshold: Number of updates before triggering rebuild
            auto_rebuild: Whether to automatically rebuild index
        """
        self.retriever = retriever
        self.rebuild_threshold = rebuild_threshold
        self.auto_rebuild = auto_rebuild

        # Track updates
        self.num_updates = 0
        self.pending_additions = {}
        self.pending_removals = set()
        self.pending_updates = {}

    def add_entity(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        immediate: bool = False,
    ):
        """
        Add a new entity to the index.

        Args:
            entity_id: Entity ID
            entity_data: Entity data with 'text' field
            immediate: Whether to update index immediately
        """
        if entity_id in self.pending_removals:
            self.pending_removals.remove(entity_id)

        self.pending_additions[entity_id] = entity_data

        if immediate:
            self._apply_pending_updates()
        else:
            self.num_updates += 1
            self._check_rebuild()

    def remove_entity(
        self,
        entity_id: str,
        immediate: bool = False,
    ):
        """
        Remove an entity from the index.

        Args:
            entity_id: Entity ID to remove
            immediate: Whether to update index immediately
        """
        if entity_id in self.pending_additions:
            del self.pending_additions[entity_id]
        else:
            self.pending_removals.add(entity_id)

        if immediate:
            self._apply_pending_updates()
        else:
            self.num_updates += 1
            self._check_rebuild()

    def update_entity(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        immediate: bool = False,
    ):
        """
        Update an existing entity's data.

        Args:
            entity_id: Entity ID
            entity_data: New entity data
            immediate: Whether to update index immediately
        """
        self.pending_updates[entity_id] = entity_data

        if immediate:
            self._apply_pending_updates()
        else:
            self.num_updates += 1
            self._check_rebuild()

    def batch_add(
        self,
        entities: dict[str, dict[str, Any]],
        immediate: bool = False,
    ):
        """
        Add multiple entities at once.

        Args:
            entities: Dictionary of entity_id -> entity_data
            immediate: Whether to update index immediately
        """
        self.pending_additions.update(entities)
        self.num_updates += len(entities)

        if immediate:
            self._apply_pending_updates()
        else:
            self._check_rebuild()

    def batch_remove(
        self,
        entity_ids: list[str],
        immediate: bool = False,
    ):
        """
        Remove multiple entities at once.

        Args:
            entity_ids: List of entity IDs to remove
            immediate: Whether to update index immediately
        """
        self.pending_removals.update(entity_ids)
        self.num_updates += len(entity_ids)

        if immediate:
            self._apply_pending_updates()
        else:
            self._check_rebuild()

    def apply_updates(self):
        """Apply all pending updates to the index."""
        self._apply_pending_updates()

    def _apply_pending_updates(self):
        """Internal method to apply pending updates."""
        if not self.pending_additions and not self.pending_removals and not self.pending_updates:
            return

        # Check if index exists
        if self.retriever.passage_index is None:
            # No index yet, build from scratch
            all_data = {**self.pending_additions}
            self.retriever.build_index(all_data)
            self.pending_additions.clear()
            self.num_updates = 0
            return

        # Apply removals
        if self.pending_removals:
            self._apply_removals()

        # Apply updates (treated as remove + add)
        if self.pending_updates:
            self._apply_entity_updates()

        # Apply additions
        if self.pending_additions:
            self._apply_additions()

        # Clear pending
        self.pending_additions.clear()
        self.pending_removals.clear()
        self.pending_updates.clear()
        self.num_updates = 0

    def _apply_additions(self):
        """Add new entities to index."""
        # Encode new entities
        new_texts = [data["text"] for data in self.pending_additions.values()]
        new_ids = list(self.pending_additions.keys())

        new_embeddings = self.retriever.encode_passage(new_texts)

        if self.retriever.use_faiss:
            try:
                import faiss

                # Add to FAISS index
                embeddings_np = new_embeddings.cpu().numpy()
                self.retriever.passage_index.add(embeddings_np)

                # Update metadata
                self.retriever.passage_ids.extend(new_ids)
                self.retriever.passage_data.update(self.pending_additions)

            except ImportError:
                # Fallback to PyTorch
                self._apply_additions_pytorch(new_embeddings, new_ids)
        else:
            self._apply_additions_pytorch(new_embeddings, new_ids)

    def _apply_additions_pytorch(self, new_embeddings, new_ids):
        """Add entities using PyTorch tensor concatenation."""
        # Concatenate embeddings
        self.retriever.passage_index = torch.cat(
            [self.retriever.passage_index, new_embeddings], dim=0
        )

        # Update metadata
        self.retriever.passage_ids.extend(new_ids)
        self.retriever.passage_data.update(self.pending_additions)

    def _apply_removals(self):
        """Remove entities from index."""
        # Find indices to remove
        indices_to_remove = []
        for i, pid in enumerate(self.retriever.passage_ids):
            if pid in self.pending_removals:
                indices_to_remove.append(i)

        if not indices_to_remove:
            return

        # Remove from metadata
        new_passage_ids = []
        for i, pid in enumerate(self.retriever.passage_ids):
            if i not in indices_to_remove:
                new_passage_ids.append(pid)
            else:
                # Remove from passage_data
                if pid in self.retriever.passage_data:
                    del self.retriever.passage_data[pid]

        self.retriever.passage_ids = new_passage_ids

        # Remove from index
        if self.retriever.use_faiss:
            # FAISS doesn't support efficient removal, rebuild
            self._rebuild_index()
        else:
            # PyTorch: create mask and filter
            keep_mask = torch.ones(len(self.retriever.passage_index), dtype=torch.bool)
            keep_mask[indices_to_remove] = False
            self.retriever.passage_index = self.retriever.passage_index[keep_mask]

    def _apply_entity_updates(self):
        """Update existing entities."""
        # Treat as remove + add
        for entity_id in self.pending_updates:
            if entity_id in self.retriever.passage_data:
                self.pending_removals.add(entity_id)

        self._apply_removals()

        # Add updated versions
        self.pending_additions.update(self.pending_updates)
        self._apply_additions()

    def _check_rebuild(self):
        """Check if index should be rebuilt."""
        if self.auto_rebuild and self.num_updates >= self.rebuild_threshold:
            self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the entire index from scratch."""
        # Collect all current data
        all_data = dict(self.retriever.passage_data)

        # Apply pending operations
        for pid in self.pending_removals:
            if pid in all_data:
                del all_data[pid]

        all_data.update(self.pending_additions)
        all_data.update(self.pending_updates)

        # Rebuild
        self.retriever.build_index(all_data)

        # Clear pending
        self.pending_additions.clear()
        self.pending_removals.clear()
        self.pending_updates.clear()
        self.num_updates = 0

    def force_rebuild(self):
        """Force a full index rebuild."""
        self._rebuild_index()

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the index and pending updates.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_entities": len(self.retriever.passage_ids) if self.retriever.passage_ids else 0,
            "pending_additions": len(self.pending_additions),
            "pending_removals": len(self.pending_removals),
            "pending_updates": len(self.pending_updates),
            "total_pending": len(self.pending_additions)
            + len(self.pending_removals)
            + len(self.pending_updates),
            "num_updates_since_rebuild": self.num_updates,
            "rebuild_threshold": self.rebuild_threshold,
            "index_type": "FAISS" if self.retriever.use_faiss else "PyTorch",
        }


__all__ = ["DynamicIndexManager"]
