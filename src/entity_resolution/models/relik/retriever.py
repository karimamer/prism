"""
ReLiK Retriever: Dense retrieval for candidate entities and relations.

Implements a bi-encoder architecture for retrieving candidate entities or relations
based on Dense Passage Retrieval (DPR).
"""

from typing import Any

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
    ):
        """
        Initialize the ReLiK Retriever.

        Args:
            model_name: Pre-trained model name for encoding
            max_query_length: Maximum query sequence length
            max_passage_length: Maximum passage sequence length
            use_faiss: Whether to use FAISS for efficient retrieval
        """
        super().__init__()

        self.model_name = model_name
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.use_faiss = use_faiss

        # Load encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size

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
        Forward pass for training.

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
