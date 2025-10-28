"""
ReLiK Retriever: Dense retrieval for candidate entities and relations.

Implements a bi-encoder architecture for retrieving candidate entities or relations
based on Dense Passage Retrieval (DPR) with contrastive learning support.
"""

from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ReLiKRetriever(nn.Module):
    """
    ReLiK Retriever for dense entity/relation retrieval.

    Uses a bi-encoder architecture to compute dense representations of queries
    and passages (entity/relation descriptions) for efficient retrieval.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        max_query_length: int = 64,
        max_passage_length: int = 64,
        use_faiss: bool = True,
        use_separate_encoders: bool = False,
        loss_type: Literal["nll", "bce"] = "nll",
        temperature: float = 1.0,
    ):
        """
        Initialize the ReLiK Retriever.

        Args:
            model_name: Pre-trained model name for encoding
            max_query_length: Maximum query sequence length
            max_passage_length: Maximum passage sequence length
            use_faiss: Whether to use FAISS for efficient retrieval
            use_separate_encoders: Use separate encoders for query and passage
            loss_type: Loss function type ('nll' or 'bce') for training
            temperature: Temperature scaling for logits
        """
        super().__init__()

        self.model_name = model_name
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.use_faiss = use_faiss
        self.use_separate_encoders = use_separate_encoders
        self.loss_type = loss_type
        self.temperature = temperature

        # Load encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Separate passage encoder (optional for asymmetric encoding)
        if use_separate_encoders:
            self.passage_encoder = AutoModel.from_pretrained(model_name)
        else:
            self.passage_encoder = self.encoder  # Shared weights

        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size

        # Loss functions for training
        if loss_type == "nll":
            self.loss_fn = nn.NLLLoss()
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Passage index (populated during build_index)
        self.passage_index = None
        self.passage_ids = None
        self.passage_data = None

    def encode_query(
        self,
        queries: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode queries into dense vectors.

        Args:
            queries: List of query strings
            batch_size: Batch size for encoding

        Returns:
            Tensor of shape (num_queries, hidden_size)
        """
        all_embeddings = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_query_length,
                return_tensors="pt",
            )

            # Move to device
            encoded = {k: v.to(self.encoder.device) for k, v in encoded.items()}

            # Encode
            with torch.no_grad():
                outputs = self.encoder(**encoded)
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, encoded["attention_mask"]
                )

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encode_passage(
        self,
        passages: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode passages into dense vectors.

        Args:
            passages: List of passage strings
            batch_size: Batch size for encoding

        Returns:
            Tensor of shape (num_passages, hidden_size)
        """
        all_embeddings = []

        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_passage_length,
                return_tensors="pt",
            )

            # Move to device
            encoded = {k: v.to(self.encoder.device) for k, v in encoded.items()}

            # Encode
            with torch.no_grad():
                outputs = self.encoder(**encoded)
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, encoded["attention_mask"]
                )

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling over token embeddings.

        Args:
            token_embeddings: Token embeddings (batch, seq_len, hidden)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Pooled embeddings (batch, hidden)
        """
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Sum mask
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Mean pooling
        return sum_embeddings / sum_mask

    def build_index(
        self,
        passages: dict[str, dict[str, Any]],
        batch_size: int = 32,
    ):
        """
        Build passage index for retrieval.

        Args:
            passages: Dictionary mapping passage IDs to passage data
                     Each passage should have a 'text' field for encoding
            batch_size: Batch size for encoding
        """
        # Extract passage texts and IDs
        passage_ids = list(passages.keys())
        passage_texts = [passages[pid]["text"] for pid in passage_ids]

        # Encode passages
        passage_embeddings = self.encode_passage(passage_texts, batch_size=batch_size)

        # Store passage data
        self.passage_ids = passage_ids
        self.passage_data = passages

        # Build index
        if self.use_faiss:
            try:
                import faiss

                # Convert to numpy for FAISS
                embeddings_np = passage_embeddings.cpu().numpy()

                # Create FAISS index
                self.passage_index = faiss.IndexFlatIP(self.hidden_size)
                self.passage_index.add(embeddings_np)
            except ImportError:
                # Fallback to PyTorch
                self.use_faiss = False
                self.passage_index = passage_embeddings
        else:
            self.passage_index = passage_embeddings

    def retrieve(
        self,
        queries: list[str],
        top_k: int = 100,
    ) -> list[list[tuple[str, dict[str, Any], float]]]:
        """
        Retrieve top-k passages for each query.

        Args:
            queries: List of query strings
            top_k: Number of top passages to retrieve

        Returns:
            List of lists of (passage_id, passage_data, score) tuples
        """
        if self.passage_index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Encode queries
        query_embeddings = self.encode_query(queries)

        # Retrieve
        if self.use_faiss:
            # Search
            scores, indices = self.passage_index.search(query_embeddings.cpu().numpy(), top_k)

            # Format results
            results = []
            for i in range(len(queries)):
                query_results = []
                for j in range(top_k):
                    idx = indices[i, j]
                    score = float(scores[i, j])
                    passage_id = self.passage_ids[idx]
                    passage_data = self.passage_data[passage_id]
                    query_results.append((passage_id, passage_data, score))
                results.append(query_results)
        else:
            # Compute similarities
            similarities = torch.matmul(query_embeddings, self.passage_index.T)

            # Get top-k
            top_scores, top_indices = torch.topk(similarities, k=top_k, dim=1)

            # Format results
            results = []
            for i in range(len(queries)):
                query_results = []
                for j in range(top_k):
                    idx = top_indices[i, j].item()
                    score = top_scores[i, j].item()
                    passage_id = self.passage_ids[idx]
                    passage_data = self.passage_data[passage_id]
                    query_results.append((passage_id, passage_data, score))
                results.append(query_results)

        return results

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for inference (encode queries or passages).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Dense embeddings
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean pooling
        embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)

        return embeddings

    def forward_train(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_ids: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass with contrastive loss.

        Supports both in-batch negatives and explicit hard negatives.

        Args:
            query_ids: Query token IDs [batch_size, query_len]
            query_mask: Query attention mask [batch_size, query_len]
            positive_ids: Positive passage IDs [batch_size, passage_len]
            positive_mask: Positive attention mask [batch_size, passage_len]
            negative_ids: Negative passage IDs [batch_size, num_negatives, passage_len]
            negative_mask: Negative attention mask [batch_size, num_negatives, passage_len]

        Returns:
            Dictionary containing:
                - loss: Contrastive loss value
                - logits: Similarity scores
                - query_emb: Query embeddings
                - pos_emb: Positive passage embeddings
        """
        batch_size = query_ids.size(0)

        # Encode queries with query encoder
        query_outputs = self.encoder(input_ids=query_ids, attention_mask=query_mask)
        query_emb = self._mean_pooling(query_outputs.last_hidden_state, query_mask)

        # Encode positive passages with passage encoder
        pos_outputs = self.passage_encoder(
            input_ids=positive_ids,
            attention_mask=positive_mask,
        )
        pos_emb = self._mean_pooling(pos_outputs.last_hidden_state, positive_mask)

        # Normalize embeddings if using BCE loss (for cosine similarity)
        if self.loss_type == "bce":
            query_emb = nn.functional.normalize(query_emb, p=2, dim=1)
            pos_emb = nn.functional.normalize(pos_emb, p=2, dim=1)

        # Compute scores based on whether we have explicit negatives
        if negative_ids is not None and negative_ids.numel() > 0:
            # Explicit negatives provided
            neg_batch_size, num_negatives, neg_len = negative_ids.shape

            # Reshape for encoding
            neg_ids_flat = negative_ids.view(-1, neg_len)
            neg_mask_flat = negative_mask.view(-1, neg_len)

            # Encode negatives
            neg_outputs = self.passage_encoder(
                input_ids=neg_ids_flat,
                attention_mask=neg_mask_flat,
            )
            neg_emb = self._mean_pooling(neg_outputs.last_hidden_state, neg_mask_flat)
            neg_emb = neg_emb.view(batch_size, num_negatives, -1)

            if self.loss_type == "bce":
                neg_emb = nn.functional.normalize(neg_emb, p=2, dim=2)

            # Compute positive scores: [batch, 1]
            pos_scores = (query_emb * pos_emb).sum(dim=1, keepdim=True)

            # Compute negative scores: [batch, num_neg]
            neg_scores = torch.bmm(
                query_emb.unsqueeze(1),  # [batch, 1, hidden]
                neg_emb.transpose(1, 2),  # [batch, hidden, num_neg]
            ).squeeze(1)

            # Combine scores: [batch, 1 + num_neg]
            all_scores = torch.cat([pos_scores, neg_scores], dim=1)
            all_scores = all_scores / self.temperature

        else:
            # Use in-batch negatives (all other positives act as negatives)
            # Compute similarity matrix: [batch, batch]
            scores = torch.matmul(query_emb, pos_emb.T)
            scores = scores / self.temperature
            all_scores = scores

        # Compute loss
        if self.loss_type == "nll":
            # Labels are on diagonal (positive is at index 0 for each query)
            if negative_ids is not None:
                # First position is positive, rest are negatives
                labels = torch.zeros(batch_size, dtype=torch.long, device=query_ids.device)
            else:
                # Diagonal elements are positives (in-batch negatives)
                labels = torch.arange(batch_size, device=query_ids.device)

            log_probs = nn.functional.log_softmax(all_scores, dim=1)
            loss = self.loss_fn(log_probs, labels)

        elif self.loss_type == "bce":
            # Binary labels (first passage is positive, rest are negative)
            labels = torch.zeros_like(all_scores)
            if negative_ids is not None:
                labels[:, 0] = 1.0  # First position is positive
            else:
                # Diagonal elements are positives
                labels[torch.arange(batch_size), torch.arange(batch_size)] = 1.0

            loss = self.loss_fn(all_scores, labels)

        return {
            "loss": loss,
            "logits": all_scores,
            "query_emb": query_emb,
            "pos_emb": pos_emb,
        }
