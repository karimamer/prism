"""
Entity-Focused Encoder for Entity Resolution.

This module implements a specialized encoder that prioritizes entity-aware
contextualization over general text understanding, following the sketch.md design.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class EntityFocusedEncoder(nn.Module):
    """
    Entity-focused encoder that enhances text representations with entity-specific
    information including entity type embeddings and knowledge integration.

    Components:
    - Base encoder: Transformer-based text encoder
    - Entity projector: Projects entity information to lower dimension
    - Entity type embedder: Embeds entity type information
    - Knowledge attention: Integrates external entity knowledge

    Args:
        pretrained_model_name: Name of pretrained model to use as base encoder
        entity_knowledge_dim: Dimension for entity-specific projections
        num_entity_types: Number of entity types to support
        dropout: Dropout probability
    """

    def __init__(
        self,
        pretrained_model_name: str = "microsoft/deberta-v3-base",
        entity_knowledge_dim: int = 256,
        num_entity_types: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()

        logger.info(f"Initializing EntityFocusedEncoder with {pretrained_model_name}")

        # Load base encoder
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.base_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.config.hidden_size
        self.entity_knowledge_dim = entity_knowledge_dim

        # Entity-specific projections (prioritize entity information)
        self.entity_projector = nn.Sequential(
            nn.Linear(self.hidden_size, entity_knowledge_dim),
            nn.LayerNorm(entity_knowledge_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        # Entity type embedding layer (from SpEL)
        self.entity_type_embedder = nn.Embedding(
            num_embeddings=num_entity_types,
            embedding_dim=entity_knowledge_dim,
            padding_idx=0,
        )

        # Knowledge integration from ReLiK using multi-head attention
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization and dropout for knowledge integration
        self.knowledge_norm = nn.LayerNorm(self.hidden_size)
        self.knowledge_dropout = nn.Dropout(dropout)

        # Additional feed-forward layer for entity-aware processing
        self.entity_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(dropout),
        )
        self.entity_ffn_norm = nn.LayerNorm(self.hidden_size)

        logger.info(
            f"EntityFocusedEncoder initialized: "
            f"hidden_size={self.hidden_size}, "
            f"entity_dim={entity_knowledge_dim}, "
            f"entity_types={num_entity_types}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_knowledge: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None,
        return_entity_projections: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through entity-focused encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            entity_knowledge: Optional entity knowledge embeddings
                [batch_size, num_entities, entity_knowledge_dim]
            entity_types: Optional entity type IDs [batch_size, num_entities]
            return_entity_projections: Whether to return entity projections

        Returns:
            text_embeddings: Contextualized text embeddings [batch_size, seq_len, hidden_size]
            entity_projections: Optional entity projections [batch_size, seq_len, entity_knowledge_dim]
        """
        # Base text encoding
        outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Enhance with entity knowledge if provided (ReLiK approach)
        if entity_knowledge is not None:
            # Project entity knowledge to match text embedding dimension
            if entity_knowledge.size(-1) != self.hidden_size:
                # Expand entity knowledge dimension if needed
                entity_knowledge_expanded = torch.zeros(
                    entity_knowledge.size(0),
                    entity_knowledge.size(1),
                    self.hidden_size,
                    device=entity_knowledge.device,
                    dtype=entity_knowledge.dtype,
                )
                entity_knowledge_expanded[..., : entity_knowledge.size(-1)] = entity_knowledge
                entity_knowledge = entity_knowledge_expanded

            # Add entity type embeddings if provided
            if entity_types is not None:
                type_embeddings = self.entity_type_embedder(
                    entity_types
                )  # [batch, num_entities, entity_dim]
                # Expand to hidden size
                type_embeddings_expanded = torch.zeros(
                    type_embeddings.size(0),
                    type_embeddings.size(1),
                    self.hidden_size,
                    device=type_embeddings.device,
                    dtype=type_embeddings.dtype,
                )
                type_embeddings_expanded[..., : type_embeddings.size(-1)] = type_embeddings
                entity_knowledge = entity_knowledge + type_embeddings_expanded

            # Apply multi-head attention to integrate knowledge
            # text_embeddings is query, entity_knowledge is key and value
            attended_embeddings, _ = self.knowledge_attention(
                query=text_embeddings,
                key=entity_knowledge,
                value=entity_knowledge,
            )

            # Residual connection and layer normalization
            text_embeddings = self.knowledge_norm(
                text_embeddings + self.knowledge_dropout(attended_embeddings)
            )

        # Apply entity-aware feed-forward network
        ffn_output = self.entity_ffn(text_embeddings)
        text_embeddings = self.entity_ffn_norm(text_embeddings + ffn_output)

        # Generate entity projections if requested
        entity_projections = None
        if return_entity_projections:
            entity_projections = self.entity_projector(text_embeddings)

        return text_embeddings, entity_projections

    def encode_entities(
        self,
        entity_names: list[str],
        entity_types: Optional[list[int]] = None,
        tokenizer=None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode entity names into entity-aware embeddings.

        Args:
            entity_names: List of entity names to encode
            entity_types: Optional list of entity type IDs
            tokenizer: Tokenizer to use for encoding
            device: Device to use for computation

        Returns:
            Entity embeddings [num_entities, entity_knowledge_dim]
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize entity names
        encoded = tokenizer(
            entity_names,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Prepare entity types if provided
        entity_type_tensor = None
        if entity_types is not None:
            entity_type_tensor = torch.tensor(entity_types, device=device).unsqueeze(0)

        # Encode with entity focus
        with torch.no_grad():
            text_embeddings, entity_projections = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_types=entity_type_tensor,
                return_entity_projections=True,
            )

            # Use CLS token representation or mean pooling
            if hasattr(self.base_encoder.config, "pooler") and self.base_encoder.config.pooler:
                # Use CLS token (first token)
                entity_embeddings = entity_projections[:, 0, :]
            else:
                # Mean pooling over non-padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(entity_projections.size())
                sum_embeddings = torch.sum(entity_projections * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                entity_embeddings = sum_embeddings / sum_mask

        return entity_embeddings

    def get_entity_type_embeddings(self, entity_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Get entity type embeddings for given type IDs.

        Args:
            entity_type_ids: Entity type IDs [batch_size] or [batch_size, num_entities]

        Returns:
            Entity type embeddings [batch_size, entity_knowledge_dim] or
                                  [batch_size, num_entities, entity_knowledge_dim]
        """
        return self.entity_type_embedder(entity_type_ids)
