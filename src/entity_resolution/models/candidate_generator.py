"""
Multi-Source Entity Candidate Generator.

This module implements a unified candidate generation system that combines:
- SpEL-style mention detection
- ReLiK-style dense retrieval
- OneNet-style candidate filtering

Following the sketch.md architecture.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EntityCandidateGenerator(nn.Module):
    """
    Unified entity candidate generator that combines multiple approaches:
    1. Mention detection (SpEL-style BIO tagging)
    2. Dense retrieval (ReLiK-style)
    3. Candidate scoring and filtering (OneNet-style)

    Args:
        embedding_dim: Dimension of input embeddings
        knowledge_base: Knowledge base for entity retrieval
        num_bio_tags: Number of BIO tags (default 3: B, I, O)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        knowledge_base=None,
        num_bio_tags: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.knowledge_base = knowledge_base

        logger.info("Initializing EntityCandidateGenerator")

        # 1. SpEL-style mention detector (BIO tagging)
        self.mention_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_bio_tags),  # B-I-O tagging
        )

        # 2. ReLiK-style dense retrieval
        self.entity_retriever = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. OneNet-style candidate filtering
        self.candidate_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

        # Additional span representation layer
        self.span_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        logger.info(
            f"EntityCandidateGenerator initialized with "
            f"embedding_dim={embedding_dim}, "
            f"bio_tags={num_bio_tags}"
        )

    def detect_mentions(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[int, int]]]:
        """
        Detect entity mentions using SpEL-style BIO tagging.

        Args:
            text_embeddings: Text embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            List of mentions per batch item, where each mention is (start_idx, end_idx)
        """
        # Apply mention detector to get BIO tags
        tag_logits = self.mention_detector(text_embeddings)  # [batch, seq_len, 3]
        tag_probs = F.softmax(tag_logits, dim=-1)
        tag_ids = torch.argmax(tag_probs, dim=-1)  # [batch, seq_len]

        # Extract mentions from BIO tags for each item in batch
        batch_mentions = []

        for i in range(tag_ids.size(0)):
            sent_tags = tag_ids[i].cpu().numpy()
            mask = attention_mask[i].cpu().numpy() if attention_mask is not None else None

            mentions = []
            current_mention = None

            for j, tag in enumerate(sent_tags):
                # Skip padding tokens
                if mask is not None and mask[j] == 0:
                    if current_mention is not None:
                        mentions.append((current_mention, j - 1))
                        current_mention = None
                    continue

                if tag == 1:  # B tag - beginning of entity
                    if current_mention is not None:
                        mentions.append((current_mention, j - 1))
                    current_mention = j
                elif tag == 2:  # I tag - inside entity
                    continue
                elif tag == 0:  # O tag - outside entity
                    if current_mention is not None:
                        mentions.append((current_mention, j - 1))
                        current_mention = None

            # Handle mention continuing to end
            if current_mention is not None:
                mentions.append((current_mention, len(sent_tags) - 1))

            batch_mentions.append(mentions)

        return batch_mentions

    def get_mention_embeddings(
        self,
        text_embeddings: torch.Tensor,
        mention_spans: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Get embeddings for entity mentions by pooling span representations.

        Args:
            text_embeddings: Text embeddings [seq_len, embedding_dim]
            mention_spans: List of (start, end) tuples

        Returns:
            Mention embeddings [num_mentions, embedding_dim]
        """
        mention_embeddings = []

        for start, end in mention_spans:
            # Average pooling over span tokens
            span_emb = text_embeddings[start : end + 1].mean(dim=0)
            # Project to mention space
            span_emb = self.span_projector(span_emb)
            mention_embeddings.append(span_emb)

        if len(mention_embeddings) == 0:
            # Return empty tensor if no mentions
            return torch.zeros((0, self.embedding_dim), device=text_embeddings.device)

        return torch.stack(mention_embeddings)

    def retrieve_candidates(
        self,
        mention_embeddings: torch.Tensor,
        top_k: int = 100,
    ) -> Tuple[List[List[Dict]], List[List[torch.Tensor]]]:
        """
        Retrieve candidate entities using ReLiK-style dense retrieval.

        Args:
            mention_embeddings: Mention embeddings [num_mentions, embedding_dim]
            top_k: Number of candidates to retrieve per mention

        Returns:
            all_candidates: List of candidate lists per mention
            all_candidate_embeddings: List of candidate embedding lists per mention
        """
        if self.knowledge_base is None:
            logger.warning("No knowledge base provided, returning empty candidates")
            return [], []

        # Transform mention embeddings for retrieval
        query_vectors = self.entity_retriever(mention_embeddings)

        # Retrieve candidates from knowledge base
        all_candidates = []
        all_candidate_embeddings = []

        for query in query_vectors:
            # Retrieve top-k * 2 candidates (will filter down later)
            candidates = self.knowledge_base.search(
                query.unsqueeze(0).cpu().detach().numpy(),
                k=min(top_k * 2, self.knowledge_base.num_entities),
            )

            if len(candidates) > 0:
                candidate_dicts = []
                candidate_embs = []

                for cand_id, score in candidates[0]:
                    # Get candidate info from knowledge base
                    cand_info = self.knowledge_base.get_entity(cand_id)
                    if cand_info is not None:
                        candidate_dicts.append(
                            {
                                "id": cand_id,
                                "name": cand_info.get("name", ""),
                                "type": cand_info.get("type", ""),
                                "score": float(score),
                            }
                        )
                        # Get candidate embedding
                        cand_emb = self.knowledge_base.get_embedding(cand_id)
                        if cand_emb is not None:
                            candidate_embs.append(torch.tensor(cand_emb, device=query.device))
                        else:
                            # Use zero embedding if not available
                            candidate_embs.append(torch.zeros_like(query))

                all_candidates.append(candidate_dicts)
                all_candidate_embeddings.append(candidate_embs)
            else:
                all_candidates.append([])
                all_candidate_embeddings.append([])

        return all_candidates, all_candidate_embeddings

    def score_and_filter_candidates(
        self,
        mention_embeddings: torch.Tensor,
        all_candidates: List[List[Dict]],
        all_candidate_embeddings: List[List[torch.Tensor]],
        top_k: int = 100,
    ) -> List[Dict]:
        """
        Score and filter candidates using OneNet-style approach.

        Args:
            mention_embeddings: Mention embeddings [num_mentions, embedding_dim]
            all_candidates: List of candidate lists per mention
            all_candidate_embeddings: List of candidate embedding lists per mention
            top_k: Number of top candidates to keep

        Returns:
            List of filtered candidate sets with scores
        """
        filtered_candidates = []

        for i, (mention_emb, candidates, candidate_embs) in enumerate(
            zip(mention_embeddings, all_candidates, all_candidate_embeddings)
        ):
            if len(candidates) == 0:
                filtered_candidates.append(
                    {
                        "mention_idx": i,
                        "candidates": [],
                    }
                )
                continue

            # Score each candidate
            scores = []
            for cand_emb in candidate_embs:
                # Concatenate mention and candidate embeddings
                combined = torch.cat([mention_emb, cand_emb])
                # Score using candidate scorer
                score = self.candidate_scorer(combined)
                scores.append(score.item())

            # Keep top-k highest scoring candidates
            if len(scores) > 0:
                # Convert to tensor for topk operation
                scores_tensor = torch.tensor(scores, device=mention_emb.device)
                top_scores, indices = scores_tensor.topk(min(top_k, len(scores)), largest=True)

                # Update candidate scores with filtered scores
                filtered_cands = []
                for idx, score in zip(indices.cpu().numpy(), top_scores.cpu().numpy()):
                    cand = candidates[idx].copy()
                    cand["filtered_score"] = float(score)
                    filtered_cands.append(cand)

                filtered_candidates.append(
                    {
                        "mention_idx": i,
                        "candidates": filtered_cands,
                    }
                )
            else:
                filtered_candidates.append(
                    {
                        "mention_idx": i,
                        "candidates": [],
                    }
                )

        return filtered_candidates

    def forward(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 100,
        return_all_mentions: bool = True,
    ) -> List[Dict]:
        """
        Full forward pass: detect mentions, retrieve candidates, and filter.

        Args:
            text_embeddings: Text embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            top_k: Number of top candidates to keep per mention
            return_all_mentions: Whether to return mentions even without candidates

        Returns:
            List of candidate sets per batch item, each containing:
                - mention_span: (start, end) tuple
                - mention_embedding: Mention embedding
                - candidates: List of candidate dicts with scores
        """
        batch_size = text_embeddings.size(0)
        all_results = []

        # 1. Detect mentions for entire batch
        batch_mentions = self.detect_mentions(text_embeddings, attention_mask)

        # 2. Process each item in batch
        for batch_idx in range(batch_size):
            item_mentions = batch_mentions[batch_idx]

            if len(item_mentions) == 0:
                all_results.append([])
                continue

            # Get mention embeddings for this item
            mention_embeddings = self.get_mention_embeddings(
                text_embeddings[batch_idx], item_mentions
            )

            # 3. Retrieve candidates
            all_candidates, all_candidate_embeddings = self.retrieve_candidates(
                mention_embeddings, top_k
            )

            # 4. Score and filter candidates
            filtered_results = self.score_and_filter_candidates(
                mention_embeddings,
                all_candidates,
                all_candidate_embeddings,
                top_k,
            )

            # 5. Combine mentions with candidates
            item_results = []
            for mention_idx, (start, end) in enumerate(item_mentions):
                result = {
                    "mention_span": (start, end),
                    "mention_embedding": mention_embeddings[mention_idx],
                    "candidates": filtered_results[mention_idx]["candidates"]
                    if mention_idx < len(filtered_results)
                    else [],
                }

                # Only include if has candidates or if returning all mentions
                if len(result["candidates"]) > 0 or return_all_mentions:
                    item_results.append(result)

            all_results.append(item_results)

        return all_results
