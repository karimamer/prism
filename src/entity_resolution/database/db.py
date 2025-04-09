import duckdb
import torch
from typing import List, Tuple, Dict, Any

class DuckDBKnowledgeBase:
    def __init__(self, db_path=":memory:"):
        """
        Initialize knowledge base with DuckDB

        Args:
            db_path: Path to database file or ":memory:" for in-memory database
        """
        self.conn = duckdb.connect(db_path)
        self._setup_database()

    def _setup_database(self):
        """Set up the database schema for entities and embeddings"""
        # Create entity table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                description TEXT,
                type VARCHAR
            )
        """)

        # Create embeddings table with vector support
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                entity_id VARCHAR PRIMARY KEY,
                embedding FLOAT[]
            )
        """)

    def load_entities(self, entities: List[Dict[str, Any]]):
        """
        Load entities into the database

        Args:
            entities: List of entity dictionaries
        """
        # Prepare data for batch insert
        entity_data = [(
            entity['id'],
            entity['name'],
            entity.get('description', ''),
            entity.get('type', 'unknown')
        ) for entity in entities]

        # Batch insert entities
        self.conn.executemany("""
            INSERT INTO entities (entity_id, name, description, type)
            VALUES (?, ?, ?, ?)
        """, entity_data)

    def add_embeddings(self, entity_embeddings: Dict[str, torch.Tensor]):
        """
        Add embeddings for entities

        Args:
            entity_embeddings: Dictionary mapping entity IDs to embeddings
        """
        # Convert torch tensors to numpy arrays for DuckDB
        embedding_data = [(
            entity_id,
            embedding.cpu().numpy().tolist()
        ) for entity_id, embedding in entity_embeddings.items()]

        # Batch insert embeddings
        self.conn.executemany("""
            INSERT INTO embeddings (entity_id, embedding)
            VALUES (?, ?)
        """, embedding_data)

    def retrieve(self, query_vector: torch.Tensor, top_k: int = 100) -> Tuple[List[Dict], List[torch.Tensor]]:
        """
        Retrieve most similar entities to query vector

        Args:
            query_vector: Query embedding tensor
            top_k: Number of results to return

        Returns:
            Tuple of (entities, embeddings)
        """
        # Convert query vector to list for DuckDB
        query_list = query_vector.cpu().numpy().tolist()

        # Perform cosine similarity search using DuckDB vector operations
        results = self.conn.execute("""
            SELECT e.entity_id, e.name, e.description, e.type,
                   cosine_similarity(emb.embedding, ?) AS similarity
            FROM entities e
            JOIN embeddings emb ON e.entity_id = emb.entity_id
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_list, top_k]).fetchall()

        # Parse results
        entities = []
        embeddings = []

        for entity_id, name, description, entity_type, similarity in results:
            # Get the entity information
            entity = {
                'id': entity_id,
                'name': name,
                'description': description,
                'type': entity_type,
                'similarity': similarity
            }
            entities.append(entity)

            # Get the corresponding embedding
            embedding_result = self.conn.execute("""
                SELECT embedding FROM embeddings WHERE entity_id = ?
            """, [entity_id]).fetchone()

            if embedding_result:
                embedding = torch.tensor(embedding_result[0])
                embeddings.append(embedding)

        return entities, embeddings

    def get_entity_by_id(self, entity_id: str) -> Dict[str, Any]:
        """
        Get entity by ID

        Args:
            entity_id: Entity identifier

        Returns:
            Entity information or None if not found
        """
        result = self.conn.execute("""
            SELECT entity_id, name, description, type
            FROM entities
            WHERE entity_id = ?
        """, [entity_id]).fetchone()

        if result:
            entity_id, name, description, entity_type = result
            return {
                'id': entity_id,
                'name': name,
                'description': description,
                'type': entity_type
            }
        return None

    def close(self):
        """Close the database connection"""
        self.conn.close()
