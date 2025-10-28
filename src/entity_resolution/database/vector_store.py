import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EntityKnowledgeBase:
    """
    Vector database for efficient entity storage and retrieval.

    This class provides functionality to:
    1. Load entities from various sources
    2. Create and manage entity embeddings
    3. Perform efficient nearest-neighbor search
    """

    def __init__(
        self,
        index_path: str = "./entity_index",
        cache_dir: str = "./cache",
        dimension: int = 256,
        use_gpu: bool = torch.cuda.is_available(),
    ):
        self.index_path = index_path
        self.cache_dir = cache_dir
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Create directories if they don't exist
        os.makedirs(index_path, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize entity storage
        self.entities = {}
        self.index = None

        # Initialize FAISS index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index for vector search"""
        logger.info(f"Initializing FAISS index with dimension {self.dimension}")

        # Create index
        if self.use_gpu:
            # GPU index
            try:
                # Get GPU resources
                res = faiss.StandardGpuResources()

                # Create CPU index first
                cpu_index = faiss.IndexFlatIP(self.dimension)

                # Transfer to GPU
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

                logger.info("Using GPU FAISS index")
            except Exception as e:
                logger.warning(f"Failed to create GPU index: {e}")
                logger.info("Falling back to CPU index")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.use_gpu = False
        else:
            # CPU index
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Using CPU FAISS index")

    def load_entities(self, entity_file: str) -> dict[str, dict[str, Any]]:
        """
        Load entities from a file.

        Args:
            entity_file: Path to entity file (JSON or CSV)

        Returns:
            Dictionary of entities
        """
        logger.info(f"Loading entities from {entity_file}")

        # Determine file type
        file_path = Path(entity_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Entity file not found: {entity_file}")

        # Load entities based on file type
        if file_path.suffix.lower() == ".json":
            entities = self._load_entities_from_json(file_path)
        elif file_path.suffix.lower() == ".csv":
            entities = self._load_entities_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Store entities
        self.entities = entities

        # Save entities to cache
        self._save_entities_to_cache()

        logger.info(f"Loaded {len(entities)} entities")

        return entities

    def _load_entities_from_json(self, file_path: Path) -> dict[str, dict[str, Any]]:
        """Load entities from JSON file"""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Check data format
        if isinstance(data, list):
            # Convert list to dictionary
            entities = {}
            for entity in data:
                if "id" in entity:
                    entity_id = entity["id"]
                    entities[entity_id] = entity
                else:
                    logger.warning(f"Skipping entity without ID: {entity}")
        elif isinstance(data, dict):
            # Use dictionary directly
            entities = data
        else:
            raise ValueError(f"Unsupported JSON format: {type(data)}")

        return entities

    def _load_entities_from_csv(self, file_path: Path) -> dict[str, dict[str, Any]]:
        """Load entities from CSV file"""
        entities = {}

        with open(file_path, encoding="utf-8") as f:
            # Read CSV file
            reader = csv.DictReader(f)

            # Process each row
            for row in reader:
                if "id" in row:
                    entity_id = row["id"]
                    entities[entity_id] = row
                else:
                    # Generate ID if not present
                    entity_id = f"entity_{len(entities)}"
                    entities[entity_id] = row

        return entities

    def _save_entities_to_cache(self):
        """Save entities to cache"""
        cache_file = os.path.join(self.cache_dir, "entities.json")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self.entities, f)

        logger.info(f"Saved {len(self.entities)} entities to cache")

    def build_index(self, entity_embeddings: dict[str, np.ndarray]):
        """
        Build FAISS index from entity embeddings.

        Args:
            entity_embeddings: Dictionary mapping entity IDs to embeddings
        """
        logger.info(f"Building index with {len(entity_embeddings)} entities")

        # Extract entity IDs and embeddings
        entity_ids = list(entity_embeddings.keys())
        embeddings = np.array([entity_embeddings[eid] for eid in entity_ids]).astype(np.float32)

        # Reset index
        self._initialize_index()

        # Add embeddings to index
        self.index.add(embeddings)

        # Store entity IDs
        self.entity_ids = entity_ids

        # Save index
        index_file = os.path.join(self.index_path, "entity_index.faiss")
        if not self.use_gpu:
            faiss.write_index(self.index, index_file)
        else:
            # Convert GPU index to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_file)

        # Save entity IDs
        ids_file = os.path.join(self.index_path, "entity_ids.json")
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(entity_ids, f)

        logger.info(f"Built index with {len(entity_ids)} entities")

    def load_index(self):
        """Load FAISS index from disk"""
        index_file = os.path.join(self.index_path, "entity_index.faiss")
        ids_file = os.path.join(self.index_path, "entity_ids.json")

        if not os.path.exists(index_file) or not os.path.exists(ids_file):
            logger.warning("Index files not found")
            return False

        # Load entity IDs
        with open(ids_file, encoding="utf-8") as f:
            self.entity_ids = json.load(f)

        # Load index
        cpu_index = faiss.read_index(index_file)

        if self.use_gpu:
            # Convert to GPU index
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception as e:
                logger.warning(f"Failed to load GPU index: {e}")
                logger.info("Falling back to CPU index")
                self.index = cpu_index
                self.use_gpu = False
        else:
            self.index = cpu_index

        logger.info(f"Loaded index with {len(self.entity_ids)} entities")

        return True

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """
        Search for most similar entities.

        Args:
            query_embedding: Query embedding
            k: Number of results to return

        Returns:
            List of (entity_id, score) tuples
        """
        # Ensure index is loaded
        if self.index is None or not hasattr(self, "entity_ids"):
            if not self.load_index():
                logger.error("No index available for search")
                return []

        # Ensure query embedding is in correct format
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()

        # Reshape if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Convert to float32
        query_embedding = query_embedding.astype(np.float32)

        # Search index
        scores, indices = self.index.search(query_embedding, k)

        # Format results
        results = []
        for _i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.entity_ids):  # Valid index
                entity_id = self.entity_ids[idx]
                results.append((entity_id, float(score)))

        return results

    def get_entity(self, entity_id: str) -> Optional[dict[str, Any]]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity dictionary or None if not found
        """
        return self.entities.get(entity_id)

    def get_entities(self, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Get multiple entities by ID.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Dictionary mapping entity IDs to entity dictionaries
        """
        return {eid: self.entities[eid] for eid in entity_ids if eid in self.entities}

    def add_entity(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        entity_embedding: Optional[np.ndarray] = None,
        save_to_disk: bool = True,
    ):
        """
        Add a new entity to the knowledge base.

        Args:
            entity_id: Entity ID
            entity_data: Entity data
            entity_embedding: Optional entity embedding
            save_to_disk: Whether to save changes to disk
        """
        # Check if entity already exists
        if entity_id in self.entities:
            logger.warning(f"Entity {entity_id} already exists, use update_entity instead")
            return

        # Add entity to storage
        self.entities[entity_id] = entity_data

        # Add entity to index if embedding is provided
        if entity_embedding is not None:
            self._add_embedding_to_index(entity_id, entity_embedding)

        # Save entity to cache
        if save_to_disk:
            self._save_entities_to_cache()
            self._save_index()

        logger.debug(f"Added entity {entity_id}")

    def update_entity(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        entity_embedding: Optional[np.ndarray] = None,
        save_to_disk: bool = True,
    ):
        """
        Update an existing entity in the knowledge base.

        Args:
            entity_id: Entity ID
            entity_data: New entity data
            entity_embedding: Optional new entity embedding
            save_to_disk: Whether to save changes to disk
        """
        if entity_id not in self.entities:
            logger.warning(f"Entity {entity_id} not found, use add_entity instead")
            return

        # Update entity data
        self.entities[entity_id] = entity_data

        # Update embedding if provided
        if entity_embedding is not None:
            # For FAISS, we need to rebuild the index (no in-place updates)
            # Mark for rebuild or remove and re-add
            if hasattr(self, "entity_ids") and entity_id in self.entity_ids:
                # Remove old and add new
                self._remove_embedding_from_index(entity_id)
                self._add_embedding_to_index(entity_id, entity_embedding)

        # Save changes
        if save_to_disk:
            self._save_entities_to_cache()
            self._save_index()

        logger.debug(f"Updated entity {entity_id}")

    def remove_entity(
        self,
        entity_id: str,
        save_to_disk: bool = True,
    ):
        """
        Remove an entity from the knowledge base.

        Args:
            entity_id: Entity ID
            save_to_disk: Whether to save changes to disk
        """
        if entity_id not in self.entities:
            logger.warning(f"Entity {entity_id} not found")
            return

        # Remove from entity storage
        del self.entities[entity_id]

        # Remove from index
        if hasattr(self, "entity_ids") and entity_id in self.entity_ids:
            self._remove_embedding_from_index(entity_id)

        # Save changes
        if save_to_disk:
            self._save_entities_to_cache()
            self._save_index()

        logger.debug(f"Removed entity {entity_id}")

    def batch_add_entities(
        self,
        entities: dict[str, dict[str, Any]],
        embeddings: Optional[dict[str, np.ndarray]] = None,
        save_to_disk: bool = True,
    ):
        """
        Add multiple entities at once (more efficient).

        Args:
            entities: Dictionary of entity_id -> entity_data
            embeddings: Optional dictionary of entity_id -> embedding
            save_to_disk: Whether to save changes to disk
        """
        for entity_id, entity_data in entities.items():
            embedding = embeddings.get(entity_id) if embeddings else None
            self.add_entity(entity_id, entity_data, embedding, save_to_disk=False)

        # Save once after all additions
        if save_to_disk:
            self._save_entities_to_cache()
            self._save_index()

        logger.info(f"Batch added {len(entities)} entities")

    def batch_remove_entities(
        self,
        entity_ids: list[str],
        save_to_disk: bool = True,
    ):
        """
        Remove multiple entities at once (more efficient).

        Args:
            entity_ids: List of entity IDs to remove
            save_to_disk: Whether to save changes to disk
        """
        for entity_id in entity_ids:
            self.remove_entity(entity_id, save_to_disk=False)

        # Save once after all removals
        if save_to_disk:
            self._save_entities_to_cache()
            self._save_index()

        logger.info(f"Batch removed {len(entity_ids)} entities")

    def _add_embedding_to_index(self, entity_id: str, entity_embedding: np.ndarray):
        """
        Add embedding to FAISS index.

        Args:
            entity_id: Entity ID
            entity_embedding: Entity embedding
        """
        # Ensure embedding is in correct format
        if isinstance(entity_embedding, torch.Tensor):
            entity_embedding = entity_embedding.detach().cpu().numpy()

        # Reshape if needed
        if len(entity_embedding.shape) == 1:
            entity_embedding = entity_embedding.reshape(1, -1)

        # Convert to float32
        entity_embedding = entity_embedding.astype(np.float32)

        # Verify dimension
        if entity_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {entity_embedding.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        # Add to index
        self.index.add(entity_embedding)

        # Update entity IDs
        if hasattr(self, "entity_ids"):
            self.entity_ids.append(entity_id)
        else:
            self.entity_ids = [entity_id]

    def _remove_embedding_from_index(self, entity_id: str):
        """
        Remove embedding from FAISS index.

        Note: FAISS doesn't support efficient removal, so we mark for rebuild.
        """
        if not hasattr(self, "entity_ids"):
            return

        if entity_id in self.entity_ids:
            # Mark that index needs rebuilding
            if not hasattr(self, "_needs_rebuild"):
                self._needs_rebuild = True

            # Remove from entity_ids list
            self.entity_ids.remove(entity_id)

            logger.debug(f"Marked entity {entity_id} for removal from index")

    def rebuild_index(self, entity_embeddings: dict[str, np.ndarray]):
        """
        Rebuild the entire FAISS index.

        Use this after many updates/removals for optimal performance.

        Args:
            entity_embeddings: Dictionary mapping entity IDs to embeddings
        """
        logger.info("Rebuilding index...")
        self.build_index(entity_embeddings)
        self._needs_rebuild = False
        logger.info("Index rebuilt successfully")

    def needs_rebuild(self) -> bool:
        """
        Check if index needs rebuilding.

        Returns:
            True if rebuild is needed
        """
        return getattr(self, "_needs_rebuild", False)

    def _save_index(self):
        """Save FAISS index to disk."""
        if self.index is None:
            return

        index_file = os.path.join(self.index_path, "entity_index.faiss")

        try:
            if not self.use_gpu:
                faiss.write_index(self.index, index_file)
            else:
                # Convert GPU index to CPU for saving
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_file)

            # Save entity IDs
            if hasattr(self, "entity_ids"):
                ids_file = os.path.join(self.index_path, "entity_ids.json")
                with open(ids_file, "w", encoding="utf-8") as f:
                    json.dump(self.entity_ids, f)

            logger.debug("Saved index to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_entities": len(self.entities),
            "indexed_entities": len(self.entity_ids) if hasattr(self, "entity_ids") else 0,
            "dimension": self.dimension,
            "use_gpu": self.use_gpu,
            "needs_rebuild": self.needs_rebuild(),
            "index_path": self.index_path,
            "cache_dir": self.cache_dir,
        }

        return stats

    def compact(self):
        """
        Compact the knowledge base by removing orphaned data.

        Removes entities from storage that are not in the index.
        """
        if not hasattr(self, "entity_ids"):
            logger.warning("No index loaded, cannot compact")
            return

        # Find entities in storage but not in index
        orphaned = set(self.entities.keys()) - set(self.entity_ids)

        if orphaned:
            logger.info(f"Found {len(orphaned)} orphaned entities, removing...")
            for entity_id in orphaned:
                del self.entities[entity_id]

            self._save_entities_to_cache()
            logger.info("Compaction complete")
        else:
            logger.info("No orphaned entities found")

    def clear(self):
        """Clear all entities and reset the index."""
        self.entities = {}
        self._initialize_index()
        self.entity_ids = []
        self._needs_rebuild = False

        # Clear cache
        self._save_entities_to_cache()
        self._save_index()

        logger.info("Knowledge base cleared")
