import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np

class EntityRetriever(nn.Module):
    """
    Dense Passage Retrieval (DPR) style retriever following ReLiK paradigm.

    Uses a single shared Transformer encoder to produce dense representations
    for both queries and passages (entities/relations), following the DPR approach
    where EQ(q) = Retriever(q) and EP(p) = Retriever(p) use the same encoder.
    """
    def __init__(
        self,
        model_name="microsoft/deberta-v3-small",
        use_faiss=True,
        top_k=100
    ):
        super().__init__()

        # Single shared encoder for both queries and passages (DPR paradigm)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

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

    def encode(self, input_ids, attention_mask):
        """Encode input text/passage into dense vector using shared encoder (DPR style)"""
        # Get embeddings from shared encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use average of token encodings as specified in the paper
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Average pooling over all tokens (as specified: "average of encodings for tokens")
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return embeddings

    def encode_passages(self, passage_texts):
        """Encode passage texts (entities/relations) using shared encoder"""
        # Tokenize passage texts
        inputs = self.tokenizer(
            passage_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )

        # Move to same device as encoder
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}

        # Encode using shared encoder
        with torch.no_grad():
            embeddings = self.encode(inputs["input_ids"], inputs["attention_mask"])

        return embeddings

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
            batch_embeddings = self.encode_passages(batch_texts).detach().cpu().numpy()
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

        # Encode query using shared encoder
        with torch.no_grad():
            query_emb = self.encode(text_ids, attention_mask).detach().cpu().numpy()

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
        # Encode query using shared encoder
        query_emb = self.encode(text_ids, attention_mask)

        # Encode passages if provided
        passage_emb = None
        if entity_texts is not None:
            # Tokenize and encode passages
            entity_inputs = self.tokenizer(
                entity_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            )
            entity_inputs = {k: v.to(self.encoder.device) for k, v in entity_inputs.items()}
            passage_emb = self.encode(entity_inputs["input_ids"], entity_inputs["attention_mask"])
        elif entity_ids is not None:
            # Look up entity texts from IDs
            entity_texts = [
                f"{self.entity_data[eid]['name']} - {self.entity_data[eid]['description'][:200]}"
                for eid in entity_ids
            ]
            passage_emb = self.encode_passages(entity_texts)

        return {
            "query_embeddings": query_emb,
            "passage_embeddings": passage_emb
        }

    def compute_similarity(self, query_emb, passage_emb):
        """Compute similarity using dot product of contextualized representations (DPR style)"""
        # sim(q, p) = EQ(q)^T * EP(p) where both use same encoder
        sim = torch.matmul(query_emb, passage_emb.transpose(0, 1))
        return sim

    def multi_label_nce_loss(self, query_emb, passage_emb, positive_passage_mask, hard_negatives=None):
        """
        Multi-label Noise Contrastive Estimation loss as described in the paper.

        LRetriever = -log(sum over p+ in Dp(q) of exp(sim(q,p+)) /
                         (exp(sim(q,p+)) + sum over p- in P-q of exp(sim(q,p-))))

        Args:
            query_emb: Query embeddings [batch_size, embed_dim]
            passage_emb: Passage embeddings [num_passages, embed_dim]
            positive_passage_mask: Binary mask [batch_size, num_passages] indicating positive passages
            hard_negatives: Optional hard negative passages for mining

        Returns:
            Multi-label NCE loss
        """
        # Compute similarity matrix sim(q,p) = EQ(q)^T * EP(p)
        similarity = self.compute_similarity(query_emb, passage_emb)

        batch_size = query_emb.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            # Get positive passages for query i
            pos_mask = positive_passage_mask[i]
            pos_indices = torch.where(pos_mask)[0]

            if len(pos_indices) == 0:
                continue

            # Get similarity scores for query i
            query_scores = similarity[i]  # [num_passages]

            # Positive scores
            pos_scores = query_scores[pos_indices]  # [num_positives]

            # Negative scores (all other passages + in-batch negatives)
            neg_mask = ~pos_mask
            neg_scores = query_scores[neg_mask]  # [num_negatives]

            # Add hard negatives if provided
            if hard_negatives is not None and i < len(hard_negatives):
                hard_neg_scores = hard_negatives[i]
                neg_scores = torch.cat([neg_scores, hard_neg_scores])

            # Compute multi-label NCE loss for this query
            # Numerator: sum of exp(positive scores)
            pos_exp = torch.exp(pos_scores)
            numerator = torch.sum(pos_exp)

            # Denominator: numerator + sum of exp(negative scores)
            neg_exp = torch.exp(neg_scores)
            denominator = numerator + torch.sum(neg_exp)

            # Loss for this query: -log(numerator / denominator)
            query_loss = -torch.log(numerator / denominator + 1e-8)
            total_loss += query_loss

        return total_loss / batch_size

    def mine_hard_negatives(self, query_emb, passage_emb, positive_mask, top_k_negatives=10):
        """
        Mine hard negatives by retrieving highest-scoring incorrect passages.

        Args:
            query_emb: Query embeddings [batch_size, embed_dim]
            passage_emb: Passage embeddings [num_passages, embed_dim]
            positive_mask: Binary mask [batch_size, num_passages] indicating positive passages
            top_k_negatives: Number of hard negatives to mine per query

        Returns:
            List of hard negative scores for each query
        """
        # Compute similarity scores
        similarity = self.compute_similarity(query_emb, passage_emb)

        hard_negatives = []

        for i in range(query_emb.size(0)):
            # Get negative passages (non-positive)
            neg_mask = ~positive_mask[i]

            if neg_mask.sum() == 0:
                hard_negatives.append(torch.tensor([]))
                continue

            # Get scores for negative passages
            neg_scores = similarity[i][neg_mask]

            # Get top-k highest scoring negatives (hardest negatives)
            top_k = min(top_k_negatives, len(neg_scores))
            hard_neg_scores, _ = torch.topk(neg_scores, top_k)

            hard_negatives.append(hard_neg_scores)

        return hard_negatives

    def save(self, path):
        """Save retriever model and index"""
        # Save model state
        torch.save({
            "encoder": self.encoder.state_dict(),
            "entity_ids": self.entity_ids,
            "config": {
                "model_name": self.encoder.config.name_or_path,
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
            top_k=config["top_k"]
        )

        # Load model weights
        retriever.encoder.load_state_dict(state_dict["encoder"])

        # Load entity data
        retriever.entity_ids = state_dict["entity_ids"]
        retriever.entity_data = torch.load(f"{path}/entity_data.pt")

        # Load FAISS index
        try:
            retriever.index = faiss.read_index(f"{path}/entity_index.faiss")
        except:
            print("No FAISS index found. You'll need to rebuild the index.")

        return retriever
