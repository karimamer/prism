"""
Interaction Map for UniRel model.

The Interaction Map is the core innovation of UniRel, modeling both:
- Entity-Entity interactions (for relation extraction)
- Entity-Relation interactions (for linking entities to relations)

Built on top of self-attention mechanism.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionMap(nn.Module):
    """
    Interaction Map for modeling entity-entity and entity-relation interactions.

    The Interaction Map uses multi-head self-attention to capture:
    1. Entity-Entity interactions: Which entities are related?
    2. Entity-Relation interactions: Which entities participate in which relations?

    This unified interaction modeling enables joint extraction of relational triples.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize Interaction Map.

        Args:
            hidden_size: Hidden size of encoder representations
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Output projection
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        # Dropout for attention
        self.attention_dropout = nn.Dropout(dropout)

        # Entity-Entity interaction classifier
        self.ee_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Binary: interacting or not
        )

        # Entity-Relation interaction classifier
        self.er_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Binary: entity participates in relation or not
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        entity_positions: Optional[torch.Tensor] = None,
        relation_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of Interaction Map.

        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            entity_positions: Boolean tensor indicating entity positions [batch_size, seq_len]
            relation_positions: Boolean tensor indicating relation positions [batch_size, seq_len]

        Returns:
            Tuple of:
            - Enhanced hidden states [batch_size, seq_len, hidden_size]
            - Entity-Entity interaction scores [batch_size, num_entities, num_entities]
            - Entity-Relation interaction scores [batch_size, num_entities, num_relations]
        """
        batch_size = hidden_states.size(0)

        # Multi-head self-attention
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(
                self.attention_head_size, dtype=torch.float32, device=attention_scores.device
            )
        )

        # Apply attention mask
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + (1.0 - extended_mask) * -10000.0

        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        # Output projection
        attention_output = self.output_dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        attention_output = self.output_layer_norm(attention_output + hidden_states)

        # Extract entity-entity interactions
        ee_scores = None
        if entity_positions is not None:
            # Get entity representations - handle batch dimension
            # entity_positions can be boolean mask or integer indices
            entity_hidden = None
            if len(entity_positions.shape) == 1:
                # Single sequence - index first batch item
                # Handle both boolean masks and integer indices
                if entity_positions.dtype == torch.bool:
                    entity_hidden = hidden_states[0, entity_positions]
                elif entity_positions.dtype in (torch.long, torch.int, torch.int64):
                    entity_hidden = hidden_states[0, entity_positions]

            if entity_hidden is not None:
                num_entities = entity_hidden.size(0)

                if num_entities > 0:
                    # Compute pairwise entity-entity interactions
                    entity_pairs = []
                    for i in range(num_entities):
                        for j in range(num_entities):
                            if i != j:  # Exclude self-interactions
                                pair = torch.cat([entity_hidden[i], entity_hidden[j]], dim=-1)
                                entity_pairs.append(pair)

                    if entity_pairs:
                        entity_pairs = torch.stack(entity_pairs)  # [num_pairs, hidden_size * 2]
                        ee_logits = self.ee_classifier(entity_pairs)  # [num_pairs, 1]
                        ee_scores = ee_logits.squeeze(-1)  # [num_pairs]
                        # Reshape to matrix form
                        ee_scores = ee_scores.view(num_entities, num_entities - 1)

        # Extract entity-relation interactions
        er_scores = None
        if entity_positions is not None and relation_positions is not None:
            # Get entity and relation representations - handle batch dimension
            entity_hidden = None
            relation_hidden = None

            if len(entity_positions.shape) == 1 and len(relation_positions.shape) == 1:
                # Single sequence - index first batch item
                # Handle both boolean masks and integer indices
                if entity_positions.dtype == torch.bool:
                    entity_hidden = hidden_states[0, entity_positions]
                elif entity_positions.dtype in (torch.long, torch.int, torch.int64):
                    entity_hidden = hidden_states[0, entity_positions]

                if relation_positions.dtype == torch.bool:
                    relation_hidden = hidden_states[0, relation_positions]
                elif relation_positions.dtype in (torch.long, torch.int, torch.int64):
                    relation_hidden = hidden_states[0, relation_positions]

            if entity_hidden is not None and relation_hidden is not None:
                num_entities = entity_hidden.size(0)
                num_relations = relation_hidden.size(0)

                if num_entities > 0 and num_relations > 0:
                    # Compute entity-relation interactions
                    er_pairs = []
                    for i in range(num_entities):
                        for j in range(num_relations):
                            pair = torch.cat([entity_hidden[i], relation_hidden[j]], dim=-1)
                            er_pairs.append(pair)

                    if er_pairs:
                        er_pairs = torch.stack(
                            er_pairs
                        )  # [num_entities * num_relations, hidden_size * 2]
                        er_logits = self.er_classifier(
                            er_pairs
                        )  # [num_entities * num_relations, 1]
                        er_scores = er_logits.squeeze(-1)  # [num_entities * num_relations]
                        # Reshape to matrix form
                        er_scores = er_scores.view(num_entities, num_relations)

        return attention_output, ee_scores, er_scores


class InteractionDecoder(nn.Module):
    """
    Decoder for extracting triples from Interaction Map.

    Uses entity-entity and entity-relation interactions to form
    <subject-relation-object> triples.
    """

    def __init__(
        self,
        entity_threshold: float = 0.5,
        relation_threshold: float = 0.5,
        triple_threshold: float = 0.5,
        max_triples: int = 20,
    ):
        """
        Initialize Interaction Decoder.

        Args:
            entity_threshold: Threshold for entity detection
            relation_threshold: Threshold for relation detection
            triple_threshold: Threshold for triple extraction
            max_triples: Maximum number of triples to extract
        """
        super().__init__()
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self.triple_threshold = triple_threshold
        self.max_triples = max_triples

    def forward(
        self,
        ee_scores: torch.Tensor,
        er_scores: torch.Tensor,
        entity_labels: torch.Tensor,
        relation_types: list,
    ) -> list:
        """
        Decode triples from interaction scores.

        Args:
            ee_scores: Entity-entity interaction scores [num_entities, num_entities]
            er_scores: Entity-relation interaction scores [num_entities, num_relations]
            entity_labels: Entity labels/spans
            relation_types: List of relation type names

        Returns:
            List of extracted triples in format [(subject, relation, object), ...]
        """
        if ee_scores is None or er_scores is None:
            return []

        # Convert scores to probabilities
        ee_probs = torch.sigmoid(ee_scores)
        er_probs = torch.sigmoid(er_scores)

        num_entities = ee_probs.size(0)
        num_relations = er_probs.size(1)

        # Extract triples
        triples = []

        for rel_idx in range(num_relations):
            # Find entities participating in this relation
            participating_entities = (er_probs[:, rel_idx] > self.relation_threshold).nonzero(
                as_tuple=True
            )[0]

            # Find entity pairs with strong interaction
            for i in participating_entities:
                for j in participating_entities:
                    if i != j:
                        # Check entity-entity interaction strength
                        # Note: ee_scores has shape [num_entities, num_entities - 1]
                        # Need to map (i, j) to the correct index
                        if i < j:
                            ee_idx = j - 1 if i < j else j
                        else:
                            ee_idx = j

                        if i < ee_probs.size(0) and ee_idx < ee_probs.size(1):
                            interaction_score = ee_probs[i, ee_idx]

                            # Combine scores for triple confidence
                            triple_score = (
                                interaction_score * er_probs[i, rel_idx] * er_probs[j, rel_idx]
                            )

                            if triple_score > self.triple_threshold:
                                # Convert tensor indices to integers
                                i_idx = i.item() if torch.is_tensor(i) else int(i)
                                j_idx = j.item() if torch.is_tensor(j) else int(j)

                                # Check bounds
                                if i_idx < len(entity_labels) and j_idx < len(entity_labels):
                                    triple = (
                                        entity_labels[i_idx],
                                        relation_types[rel_idx],
                                        entity_labels[j_idx],
                                        triple_score.item(),
                                    )
                                    triples.append(triple)

        # Sort by confidence and limit
        triples.sort(key=lambda x: x[3], reverse=True)
        triples = triples[: self.max_triples]

        # Remove confidence scores for output
        triples = [(s, r, o) for s, r, o, _ in triples]

        return triples
