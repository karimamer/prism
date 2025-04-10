import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np

class EntityRetriever(nn.Module):
    """
    ReLiK-style dense retrieval component with bi-encoder architecture.

    This retriever uses two encoders: one for query (input text) and one for entities.
    It efficiently retrieves the most relevant entities for a given text using FAISS.
    """
    def __init__(
        self,
        model_name="microsoft/deberta-v3-small",
        entity_dim=256,
        shared_encoder=True,
        use_faiss=True,
        top_k=100
    ):
        super().__init__()

        # Initialize text encoder
        self.text_encoder = AutoModel.from_pretrained(model_name)

        # Entity encoder can be shared with text encoder (parameter efficient)
        # or separate (more expressive but more parameters)
        if shared_encoder:
            self.entity_encoder = self.text_encoder
        else:
            self.entity_encoder = AutoModel.from_pretrained(model_name)

        # Projection layers for dimension reduction and alignment
        hidden_size = self.text_encoder.config.hidden_size
        self.text_projection = nn.Linear(hidden_size, entity_dim)
        self.entity_projection = nn.Linear(hidden_size, entity_dim)

        # Tokenizer for processing text
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FAISS index for efficient retrieval
        self.use_faiss = use_faiss
        self.index = None
        self.entity_ids = []
        self.entity_data = {}
        self.top_k = top_k

    def encode_text(self, text_ids, attention_mask):
        """Encode input text into a dense vector"""
        # Get text embeddings from encoder
        outputs = self.text_encoder(
            input_ids=text_ids,
            attention_mask=attention_mask
        )

        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output'):
            text_emb = outputs.pooler_output
        else:
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            text_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Project to common space
        text_emb = self.text_projection(text_emb)

        # Normalize embeddings (important for cosine similarity)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        return text_emb

    def encode_entity(self, entity_text):
        """Encode entity text into a dense vector"""
        # Tokenize entity text
        entity_inputs = self.tokenizer(
            entity_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )

        # Get entity embeddings from encoder
        outputs = self.entity_encoder(
            input_ids=entity_inputs["input_ids"].to(self.text_encoder.device),
            attention_mask=entity_inputs["attention_mask"].to(self.text_encoder.device)
        )

        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output'):
            entity_emb = outputs.pooler_output
        else:
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = entity_inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            entity_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Project to common space
        entity_emb = self.entity_projection(entity_emb)

        # Normalize embeddings
        entity_emb = F.normalize(entity_emb, p=2, dim=1)
        return entity_emb

    def build_index(self, entity_dict):
        """
        Build FAISS index from entity dictionary.

        Args:
            entity_dict: Dictionary mapping entity_id to {"name": name, "description": description}
        """
        self.entity_data = entity_dict
        self.entity_ids = list(entity_dict.keys())

        # Encode all entities
        entity_texts = [
            f"{entity_dict[eid]['name']} - {entity_dict[eid]['description'][:200]}"
            for eid in self.entity_ids
        ]

        # Process in batches to avoid OOM
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(entity_texts), batch_size):
            batch_texts = entity_texts[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = self.encode_entity(batch_texts).detach().cpu().numpy()
            all_embeddings.append(batch_embeddings)

        entity_embeddings = np.vstack(all_embeddings)

        # Build FAISS index
        dimension = entity_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product is equivalent to cosine similarity for normalized vectors
        self.index.add(entity_embeddings)

        # For very large entity sets, use approximate nearest neighbors
        if len(self.entity_ids) > 100000:
            nlist = min(4096, int(np.sqrt(len(self.entity_ids))))
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(entity_embeddings)
            self.index.add(entity_embeddings)

        print(f"Built index with {len(self.entity_ids)} entities.")

    def retrieve(self, text_ids, attention_mask, top_k=None):
        """
        Retrieve the most relevant entities for a given text.

        Args:
            text_ids: Tokenized input text
            attention_mask: Attention mask for input text
            top_k: Number of entities to retrieve (defaults to self.top_k)

        Returns:
            List of (entity_id, entity_data, score) tuples
        """
        if top_k is None:
            top_k = self.top_k

        # Ensure we have an index
        if self.index is None:
            print("No entity index built yet!")
            return []

        # Encode query
        with torch.no_grad():
            query_emb = self.encode_text(text_ids, attention_mask).detach().cpu().numpy()

        # Search index
        scores, indices = self.index.search(query_emb, top_k)

        # Format results
        results = []
        for i, (score_array, idx_array) in enumerate(zip(scores, indices)):
            for score, idx in zip(score_array, idx_array):
                if idx < len(self.entity_ids):  # Guard against invalid indices
                    entity_id = self.entity_ids[idx]
                    entity_data = self.entity_data[entity_id]
                    results.append((entity_id, entity_data, float(score)))

        return results

    def forward(self, text_ids, attention_mask, entity_ids=None, entity_texts=None):
        """
        Forward pass for training the retriever.

        For contrastive learning, we need:
        - Encoded text
        - Encoded positive entities (that match the text)
        - Encoded negative entities (that don't match)

        Args:
            text_ids: Tokenized input texts
            attention_mask: Attention masks for input texts
            entity_ids: IDs of positive entities for each text (Optional, only used if entity_texts not provided)
            entity_texts: Text representations of positive entities for each text

        Returns:
            Dictionary with text embeddings and entity embeddings
        """
        # Encode text
        text_emb = self.encode_text(text_ids, attention_mask)

        # Encode entities if provided
        entity_emb = None
        if entity_texts is not None:
            entity_emb = self.encode_entity(entity_texts)
        elif entity_ids is not None:
            # Look up entity texts from IDs
            entity_texts = [
                f"{self.entity_data[eid]['name']} - {self.entity_data[eid]['description'][:200]}"
                for eid in entity_ids
            ]
            entity_emb = self.encode_entity(entity_texts)

        return {
            "text_embeddings": text_emb,
            "entity_embeddings": entity_emb
        }

    def compute_similarity(self, text_emb, entity_emb):
        """Compute similarity matrix between text and entity embeddings"""
        # Cosine similarity
        sim = torch.matmul(text_emb, entity_emb.transpose(0, 1))
        return sim

    def contrastive_loss(self, text_emb, entity_emb, positive_pairs, temperature=0.07):
        """
        Compute contrastive loss for retriever training.

        Args:
            text_emb: Text embeddings [batch_size, embed_dim]
            entity_emb: Entity embeddings [batch_size, embed_dim]
            positive_pairs: Indices of positive text-entity pairs
            temperature: Temperature for softmax

        Returns:
            Contrastive loss
        """
        # Compute similarity matrix
        similarity = self.compute_similarity(text_emb, entity_emb) / temperature

        # Create labels (diagonal is positive pairs)
        labels = torch.zeros(similarity.size(0), dtype=torch.long, device=similarity.device)
        for i, pos_idx in enumerate(positive_pairs):
            labels[i] = pos_idx

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        return loss

    def save(self, path):
        """Save retriever model and index"""
        # Save model state
        torch.save({
            "text_encoder": self.text_encoder.state_dict(),
            "entity_encoder": self.entity_encoder.state_dict() if not self.shared_encoder else None,
            "text_projection": self.text_projection.state_dict(),
            "entity_projection": self.entity_projection.state_dict(),
            "entity_ids": self.entity_ids,
            "config": {
                "model_name": self.text_encoder.config.name_or_path,
                "entity_dim": self.text_projection.out_features,
                "shared_encoder": self.shared_encoder,
                "top_k": self.top_k
            }
        }, f"{path}/retriever_model.pt")

        # Save FAISS index
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, f"{path}/entity_index.faiss")

        # Save entity data
        torch.save(self.entity_data, f"{path}/entity_data.pt")

    @classmethod
    def load(cls, path):
        """Load retriever model and index"""
        # Load model state
        state_dict = torch.load(f"{path}/retriever_model.pt")
        config = state_dict["config"]

        # Create model instance
        retriever = cls(
            model_name=config["model_name"],
            entity_dim=config["entity_dim"],
            shared_encoder=config["shared_encoder"],
            top_k=config["top_k"]
        )

        # Load model weights
        retriever.text_encoder.load_state_dict(state_dict["text_encoder"])
        if not config["shared_encoder"] and state_dict["entity_encoder"] is not None:
            retriever.entity_encoder.load_state_dict(state_dict["entity_encoder"])
        retriever.text_projection.load_state_dict(state_dict["text_projection"])
        retriever.entity_projection.load_state_dict(state_dict["entity_projection"])

        # Load entity data
        retriever.entity_ids = state_dict["entity_ids"]
        retriever.entity_data = torch.load(f"{path}/entity_data.pt")

        # Load FAISS index
        try:
            retriever.index = faiss.read_index(f"{path}/entity_index.faiss")
        except:
            print("No FAISS index found. You'll need to rebuild the index.")

        return retriever
