import os
import json
import csv
import torch
import faiss
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        use_gpu: bool = torch.cuda.is_available()
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

    def load_entities(self, entity_file: str) -> Dict[str, Dict[str, Any]]:
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
        if file_path.suffix.lower() == '.json':
            entities = self._load_entities_from_json(file_path)
        elif file_path.suffix.lower() == '.csv':
            entities = self._load_entities_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Store entities
        self.entities = entities

        # Save entities to cache
        self._save_entities_to_cache()

        logger.info(f"Loaded {len(entities)} entities")

        return entities

    def _load_entities_from_json(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load entities from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check data format
        if isinstance(data, list):
            # Convert list to dictionary
            entities = {}
            for entity in data:
                if 'id' in entity:
                    entity_id = entity['id']
                    entities[entity_id] = entity
                else:
                    logger.warning(f"Skipping entity without ID: {entity}")
        elif isinstance(data, dict):
            # Use dictionary directly
            entities = data
        else:
            raise ValueError(f"Unsupported JSON format: {type(data)}")

        return entities

    def _load_entities_from_csv(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load entities from CSV file"""
        entities = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            # Read CSV file
            reader = csv.DictReader(f)

            # Process each row
            for row in reader:
                if 'id' in row:
                    entity_id = row['id']
                    entities[entity_id] = row
                else:
                    # Generate ID if not present
                    entity_id = f"entity_{len(entities)}"
                    entities[entity_id] = row

        return entities

    def _save_entities_to_cache(self):
        """Save entities to cache"""
        cache_file = os.path.join(self.cache_dir, "entities.json")

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.entities, f)

        logger.info(f"Saved {len(self.entities)} entities to cache")

    def build_index(self, entity_embeddings: Dict[str, np.ndarray]):
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
        with open(ids_file, 'w', encoding='utf-8') as f:
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
        with open(ids_file, 'r', encoding='utf-8') as f:
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

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for most similar entities.

        Args:
            query_embedding: Query embedding
            k: Number of results to return

        Returns:
            List of (entity_id, score) tuples
        """
        # Ensure index is loaded
        if self.index is None or not hasattr(self, 'entity_ids'):
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
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.entity_ids):  # Valid index
                entity_id = self.entity_ids[idx]
                results.append((entity_id, float(score)))

        return results

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity dictionary or None if not found
        """
        return self.entities.get(entity_id)

    def get_entities(self, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple entities by ID.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Dictionary mapping entity IDs to entity dictionaries
        """
        return {eid: self.entities[eid] for eid in entity_ids if eid in self.entities}

    def add_entity(self, entity_id: str, entity_data: Dict[str, Any], entity_embedding: Optional[np.ndarray] = None):
        """
        Add a new entity to the knowledge base.

        Args:
            entity_id: Entity ID
            entity_data: Entity data
            entity_embedding: Optional entity embedding
        """
        # Add entity to storage
        self.entities[entity_id] = entity_data

        # Add entity to index if embedding is provided
        if entity_embedding is not None:
            # Ensure embedding is in correct format
            if isinstance(entity_embedding, torch.Tensor):
                entity_embedding = entity_embedding.detach().cpu().numpy()

            # Reshape if needed
            if len(entity_embedding.shape) == 1:
                entity_embedding = entity_embedding.reshape(1, -1)

            # Convert to float32
            entity_embedding = entity_embedding.astype(np.float32)

            # Add to index
            self.index.add(entity_embedding)

            # Update entity IDs
            if hasattr(self, 'entity_ids'):
                self.entity_ids.append(entity_id)
            else:
                self.entity_ids = [entity_id]

        # Save entity to cache
        self._save_entities_to_cache()
