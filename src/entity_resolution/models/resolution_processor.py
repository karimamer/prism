"""
Cross-Model Entity Resolution Processor.

This module implements a multi-method entity resolution processor that combines:
- ReLiK reader component
- ATG entity representation
- UniRel entity correlation
- Cross-model attention for information exchange

Following the sketch.md architecture.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EntityResolutionProcessor(nn.Module):
    """
    Multi-method entity resolution processor that combines outputs from:
    1. ReLiK reader (entity linking scores)
    2. ATG entity encoder (entity representation enhancement)
    3. UniRel interaction (entity correlation modeling)
    4. Cross-model attention (information exchange between methods)

    Args:
        encoder_dim: Dimension of encoder embeddings
        num_heads: Number of attention heads for cross-model attention
        dropout: Dropout probability
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim

        logger.info("Initializing EntityResolutionProcessor")

        # 1. ReLiK reader component - entity linking scorer
        self.relik_reader = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 1),
        )

        # 2. ATG entity representation component - transformer encoder
        self.atg_entity_encoder = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        # 3. UniRel entity correlation modeling
        self.unirel_interaction = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim),
        )

        # Scoring layer for UniRel
        self.unirel_scorer = nn.Sequential(
            nn.Linear(encoder_dim, 1),
            nn.Sigmoid(),
        )

        # 4. Cross-model attention for information exchange
        self.cross_model_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization for cross-model attention
        self.cross_model_norm = nn.LayerNorm(encoder_dim)

        # Final fusion layer to combine all scores
        self.score_fusion = nn.Sequential(
            nn.Linear(3, 16),  # 3 scores: ReLiK, ATG, UniRel
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"EntityResolutionProcessor initialized with "
            f"encoder_dim={encoder_dim}, num_heads={num_heads}"
        )

    def process_relik(
        self,
        mention_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process with ReLiK reader scoring.

        Args:
            mention_embeddings: Mention embeddings [num_mentions, encoder_dim]
            candidate_embeddings: Candidate embeddings [num_mentions, num_candidates, encoder_dim]

        Returns:
            ReLiK scores [num_mentions, num_candidates]
        """
        num_mentions = mention_embeddings.size(0)
        num_candidates = candidate_embeddings.size(1)

        # Expand mention embeddings to match candidates
        mention_expanded = mention_embeddings.unsqueeze(1).expand(
            -1, num_candidates, -1
        )  # [num_mentions, num_candidates, encoder_dim]

        # Concatenate mention and candidate embeddings
        combined = torch.cat(
            [mention_expanded, candidate_embeddings], dim=-1
        )  # [num_mentions, num_candidates, encoder_dim * 2]

        # Apply ReLiK reader
        relik_scores = self.relik_reader(combined).squeeze(-1)  # [num_mentions, num_candidates]

        return relik_scores

    def process_atg(
        self,
        mention_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process with ATG entity representation enhancement.

        Args:
            mention_embeddings: Mention embeddings [num_mentions, encoder_dim]
            candidate_embeddings: Candidate embeddings [num_mentions, num_candidates, encoder_dim]

        Returns:
            ATG scores [num_mentions, num_candidates]
        """
        num_mentions = mention_embeddings.size(0)
        num_candidates = candidate_embeddings.size(1)

        atg_scores = []

        for i in range(num_mentions):
            mention_emb = mention_embeddings[i : i + 1]  # [1, encoder_dim]
            candidates_emb = candidate_embeddings[i]  # [num_candidates, encoder_dim]

            # Stack mention and candidates for transformer processing
            stacked = torch.cat(
                [mention_emb, candidates_emb], dim=0
            )  # [1 + num_candidates, encoder_dim]

            # Apply transformer encoder for entity representation enhancement
            enhanced = self.atg_entity_encoder(stacked.unsqueeze(0)).squeeze(
                0
            )  # [1 + num_candidates, encoder_dim]

            # Split back into mention and candidates
            enhanced_mention = enhanced[0]  # [encoder_dim]
            enhanced_candidates = enhanced[1:]  # [num_candidates, encoder_dim]

            # Compute cosine similarity between enhanced representations
            similarity = F.cosine_similarity(
                enhanced_mention.unsqueeze(0), enhanced_candidates, dim=-1
            )  # [num_candidates]

            atg_scores.append(similarity)

        # Stack all scores
        atg_scores = torch.stack(atg_scores)  # [num_mentions, num_candidates]

        return atg_scores

    def process_unirel(
        self,
        mention_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process with UniRel entity correlation modeling.

        Args:
            mention_embeddings: Mention embeddings [num_mentions, encoder_dim]
            candidate_embeddings: Candidate embeddings [num_mentions, num_candidates, encoder_dim]
            context_embeddings: Context embeddings around mentions [num_mentions, encoder_dim]

        Returns:
            UniRel scores [num_mentions, num_candidates]
        """
        num_mentions = mention_embeddings.size(0)
        num_candidates = candidate_embeddings.size(1)

        # Expand context embeddings to match candidates
        context_expanded = context_embeddings.unsqueeze(1).expand(
            -1, num_candidates, -1
        )  # [num_mentions, num_candidates, encoder_dim]

        # Concatenate context and candidate embeddings
        combined = torch.cat(
            [context_expanded, candidate_embeddings], dim=-1
        )  # [num_mentions, num_candidates, encoder_dim * 2]

        # Apply UniRel interaction
        interaction = self.unirel_interaction(
            combined
        )  # [num_mentions, num_candidates, encoder_dim]

        # Score the interactions
        unirel_scores = self.unirel_scorer(interaction).squeeze(
            -1
        )  # [num_mentions, num_candidates]

        return unirel_scores

    def apply_cross_model_attention(
        self,
        relik_features: torch.Tensor,
        atg_features: torch.Tensor,
        unirel_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply cross-model attention for information exchange.

        Args:
            relik_features: ReLiK features [num_mentions, num_candidates, encoder_dim]
            atg_features: ATG features [num_mentions, num_candidates, encoder_dim]
            unirel_features: UniRel features [num_mentions, num_candidates, encoder_dim]

        Returns:
            Enhanced features for each method
        """
        num_mentions, num_candidates, _ = relik_features.shape

        # Reshape for batch processing
        # Treat each mention's candidates as a sequence
        relik_flat = relik_features.view(-1, num_candidates, self.encoder_dim)
        atg_flat = atg_features.view(-1, num_candidates, self.encoder_dim)
        unirel_flat = unirel_features.view(-1, num_candidates, self.encoder_dim)

        # Stack all features for cross-attention
        # [batch, num_candidates * 3, encoder_dim]
        all_features = torch.cat([relik_flat, atg_flat, unirel_flat], dim=1)

        # Apply self-attention across all model features
        attended_features, _ = self.cross_model_attention(
            query=all_features,
            key=all_features,
            value=all_features,
        )

        # Apply residual connection and layer norm
        attended_features = self.cross_model_norm(all_features + attended_features)

        # Split back into separate model features
        relik_enhanced = attended_features[:, :num_candidates, :]
        atg_enhanced = attended_features[:, num_candidates : num_candidates * 2, :]
        unirel_enhanced = attended_features[:, num_candidates * 2 :, :]

        # Reshape back to original shape
        relik_enhanced = relik_enhanced.view(num_mentions, num_candidates, self.encoder_dim)
        atg_enhanced = atg_enhanced.view(num_mentions, num_candidates, self.encoder_dim)
        unirel_enhanced = unirel_enhanced.view(num_mentions, num_candidates, self.encoder_dim)

        return relik_enhanced, atg_enhanced, unirel_enhanced

    def forward(
        self,
        text_embeddings: torch.Tensor,
        entity_candidates: List[Dict],
    ) -> List[Dict]:
        """
        Process entity candidates with multiple methods and cross-model attention.

        Args:
            text_embeddings: Text embeddings [seq_len, encoder_dim]
            entity_candidates: List of candidate sets, each containing:
                - mention_span: (start, end) tuple
                - mention_embedding: Mention embedding
                - candidates: List of candidate dicts

        Returns:
            List of resolution results with scores from all methods
        """
        results = []

        for candidate_set in entity_candidates:
            mention_span = candidate_set["mention_span"]
            mention_embedding = candidate_set["mention_embedding"]
            candidates = candidate_set["candidates"]

            if len(candidates) == 0:
                results.append(
                    {
                        "mention_span": mention_span,
                        "candidate_scores": {},
                    }
                )
                continue

            # Prepare candidate embeddings
            candidate_embeddings = []
            for cand in candidates:
                if "embedding" in cand:
                    candidate_embeddings.append(cand["embedding"])
                else:
                    # Use zero embedding if not available
                    candidate_embeddings.append(
                        torch.zeros(self.encoder_dim, device=mention_embedding.device)
                    )

            candidate_embeddings = torch.stack(candidate_embeddings).unsqueeze(
                0
            )  # [1, num_candidates, encoder_dim]
            mention_embedding = mention_embedding.unsqueeze(0)  # [1, encoder_dim]

            # Get context embedding (tokens around mention)
            start, end = mention_span
            context_start = max(0, start - 5)
            context_end = min(text_embeddings.size(0), end + 6)
            context_embedding = text_embeddings[context_start:context_end].mean(
                dim=0, keepdim=True
            )  # [1, encoder_dim]

            # 1. Process with ReLiK reader
            relik_scores = self.process_relik(
                mention_embedding, candidate_embeddings
            )  # [1, num_candidates]

            # 2. Process with ATG entity encoder
            atg_scores = self.process_atg(
                mention_embedding, candidate_embeddings
            )  # [1, num_candidates]

            # 3. Process with UniRel interaction
            unirel_scores = self.process_unirel(
                mention_embedding, candidate_embeddings, context_embedding
            )  # [1, num_candidates]

            # Create feature representations for cross-model attention
            # Use candidate embeddings as base features
            relik_features = candidate_embeddings  # [1, num_candidates, encoder_dim]
            atg_features = candidate_embeddings.clone()
            unirel_features = candidate_embeddings.clone()

            # 4. Apply cross-model attention
            relik_enhanced, atg_enhanced, unirel_enhanced = self.apply_cross_model_attention(
                relik_features, atg_features, unirel_features
            )

            # Combine enhanced features with original scores
            # (Enhanced features can influence final fusion)

            # Stack all scores for fusion
            all_scores = torch.stack(
                [relik_scores[0], atg_scores[0], unirel_scores[0]], dim=-1
            )  # [num_candidates, 3]

            # 5. Fuse scores from all methods
            final_scores = self.score_fusion(all_scores).squeeze(-1)  # [num_candidates]

            # Store scores for each candidate
            candidate_scores = {}
            for idx, cand in enumerate(candidates):
                candidate_id = cand.get("id", idx)
                candidate_scores[candidate_id] = {
                    "relik_score": relik_scores[0, idx].item(),
                    "atg_score": atg_scores[0, idx].item(),
                    "unirel_score": unirel_scores[0, idx].item(),
                    "final_score": final_scores[idx].item(),
                    "candidate": cand,
                }

            results.append(
                {
                    "mention_span": mention_span,
                    "candidate_scores": candidate_scores,
                }
            )

        return results
