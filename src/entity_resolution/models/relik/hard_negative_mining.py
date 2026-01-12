"""
Hard Negative Mining for ReLiK Retriever Training.

Implements strategies for mining hard negatives to improve
retriever discrimination and ranking quality.
"""

from typing import Any, Optional

import torch

from .retriever import ReLiKRetriever


class HardNegativeMiner:
    """
    Hard negative mining for retriever training.

    Strategies:
    1. In-batch negatives: Use other positives in batch as negatives
    2. Top-k hard negatives: Retrieve top-k from index, filter out positives
    3. Random negatives: Sample random entities from knowledge base
    """

    def __init__(
        self,
        retriever: ReLiKRetriever,
        knowledge_base: dict[str, dict[str, Any]],
        strategy: str = "top_k",
        num_negatives: int = 7,
    ):
        """
        Initialize hard negative miner.

        Args:
            retriever: ReLiK retriever
            knowledge_base: Entity knowledge base
            strategy: Mining strategy ('top_k', 'random', 'mixed')
            num_negatives: Number of negatives per query
        """
        self.retriever = retriever
        self.knowledge_base = knowledge_base
        self.strategy = strategy
        self.num_negatives = num_negatives

        # Build index if not already built
        if self.retriever.passage_index is None:
            self.retriever.build_index(knowledge_base)

    def mine_hard_negatives(
        self,
        queries: list[str],
        positive_ids: list[str],
        top_k: int = 100,
    ) -> list[list[dict[str, Any]]]:
        """
        Mine hard negatives for a batch of queries.

        Args:
            queries: List of query texts
            positive_ids: List of positive entity IDs (one per query)
            top_k: Number of candidates to retrieve for mining

        Returns:
            List of hard negatives for each query
        """
        if self.strategy == "top_k":
            return self._mine_top_k(queries, positive_ids, top_k)
        elif self.strategy == "random":
            return self._mine_random(positive_ids)
        elif self.strategy == "mixed":
            return self._mine_mixed(queries, positive_ids, top_k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _mine_top_k(
        self,
        queries: list[str],
        positive_ids: list[str],
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        """
        Mine hard negatives from top-k retrieved candidates.

        These are "hard" because they have high similarity to the query
        but are not the correct answer.
        """
        # Retrieve top-k candidates for all queries
        all_retrieved = self.retriever.retrieve(queries, top_k=top_k)

        hard_negatives = []

        for retrieved, positive_id in zip(all_retrieved, positive_ids):
            query_negatives = []

            for cand_id, cand_data, score in retrieved:
                # Skip if this is the positive
                if cand_id == positive_id:
                    continue

                query_negatives.append(
                    {
                        "id": cand_id,
                        "data": cand_data,
                        "score": score,
                        "type": "hard_top_k",
                    }
                )

                # Stop when we have enough negatives
                if len(query_negatives) >= self.num_negatives:
                    break

            # If we don't have enough, pad with random
            if len(query_negatives) < self.num_negatives:
                random_negs = self._sample_random(
                    positive_id,
                    self.num_negatives - len(query_negatives),
                )
                query_negatives.extend(random_negs)

            hard_negatives.append(query_negatives)

        return hard_negatives

    def _mine_random(
        self,
        positive_ids: list[str],
    ) -> list[list[dict[str, Any]]]:
        """
        Mine random negatives from knowledge base.

        Less "hard" but provides diversity.
        """
        hard_negatives = []

        for positive_id in positive_ids:
            query_negatives = self._sample_random(positive_id, self.num_negatives)
            hard_negatives.append(query_negatives)

        return hard_negatives

    def _mine_mixed(
        self,
        queries: list[str],
        positive_ids: list[str],
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        """
        Mine mixed hard negatives: half from top-k, half random.

        Balances hardness with diversity.
        """
        num_hard = self.num_negatives // 2
        num_random = self.num_negatives - num_hard

        # Get top-k hard negatives
        hard_negs = self._mine_top_k(queries, positive_ids, top_k)

        # Add random negatives
        for i, positive_id in enumerate(positive_ids):
            # Keep only num_hard from top-k
            hard_negs[i] = hard_negs[i][:num_hard]

            # Add random
            random_negs = self._sample_random(positive_id, num_random)
            hard_negs[i].extend(random_negs)

        return hard_negs

    def _sample_random(
        self,
        positive_id: str,
        num_samples: int,
    ) -> list[dict[str, Any]]:
        """Sample random negatives from knowledge base."""
        import random

        # Get all entity IDs except positive
        all_ids = [eid for eid in self.knowledge_base.keys() if eid != positive_id]

        # Sample
        if len(all_ids) < num_samples:
            sampled_ids = all_ids
        else:
            sampled_ids = random.sample(all_ids, num_samples)

        # Format as negative examples
        negatives = []
        for eid in sampled_ids:
            negatives.append(
                {
                    "id": eid,
                    "data": self.knowledge_base[eid],
                    "score": 0.0,
                    "type": "random",
                }
            )

        return negatives

    def prepare_training_batch(
        self,
        queries: list[str],
        positive_ids: list[str],
        positive_texts: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prepare a complete training batch with hard negatives.

        Args:
            queries: List of query texts
            positive_ids: List of positive entity IDs
            positive_texts: Optional positive texts (uses KB if None)

        Returns:
            Dictionary with tokenized queries, positives, and negatives
        """
        # Get positive texts from KB if not provided
        if positive_texts is None:
            positive_texts = [self.knowledge_base[pid]["text"] for pid in positive_ids]

        # Mine hard negatives
        hard_negatives = self.mine_hard_negatives(queries, positive_ids)

        # Tokenize queries
        query_encoded = self.retriever.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=self.retriever.max_query_length,
            return_tensors="pt",
        )

        # Tokenize positives
        positive_encoded = self.retriever.tokenizer(
            positive_texts,
            padding="max_length",
            truncation=True,
            max_length=self.retriever.max_passage_length,
            return_tensors="pt",
        )

        # Tokenize negatives
        negative_texts = []
        for query_negs in hard_negatives:
            for neg in query_negs:
                negative_texts.append(neg["data"]["text"])

        negative_encoded = self.retriever.tokenizer(
            negative_texts,
            padding="max_length",
            truncation=True,
            max_length=self.retriever.max_passage_length,
            return_tensors="pt",
        )

        # Reshape negatives: [batch, num_negatives, seq_len]
        batch_size = len(queries)
        negative_ids = negative_encoded["input_ids"].view(batch_size, self.num_negatives, -1)
        negative_mask = negative_encoded["attention_mask"].view(batch_size, self.num_negatives, -1)

        return {
            "query_ids": query_encoded["input_ids"],
            "query_mask": query_encoded["attention_mask"],
            "positive_ids": positive_encoded["input_ids"],
            "positive_mask": positive_encoded["attention_mask"],
            "negative_ids": negative_ids,
            "negative_mask": negative_mask,
        }


__all__ = ["HardNegativeMiner"]
