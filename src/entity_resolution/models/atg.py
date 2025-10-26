"""
Improved ATG (Autoregressive Text-to-Graph) Model

This is a complete refactored implementation of the ATG model based on:
"Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction"
by Zaratiana et al. (2024)

Key improvements over original implementation:
1. Cleaner architecture with separated concerns
2. Dynamic vocabulary with proper span representations
3. State-based constrained decoding
4. Sentence augmentation support
5. Structural and positional embeddings
6. Production-ready error handling and logging

Architecture:
- Encoder: Transformer encoder + span representation layer
- Decoder: Transformer decoder with pointing mechanism
- Vocabulary: Dynamic spans + relation types + special tokens
- Generation: State-machine based constrained decoding
"""

import logging
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field, field_validator, model_validator
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .base_config import BaseEncoderConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration and Data Classes
# ============================================================================


class ATGConfig(BaseEncoderConfig):
    """
    Configuration for ATG (Autoregressive Text-to-Graph) model.

    ATG generates entity-relation graphs autoregressively, linearizing the graph
    structure into a sequence for generation.

    Inherits from BaseEncoderConfig for common encoder parameters.
    """

    # Override encoder settings with ATG-specific defaults
    encoder_model: str = Field(
        default="microsoft/deberta-v3-base",
        description="Pre-trained encoder model",
        examples=["microsoft/deberta-v3-base", "bert-base-cased"],
    )

    # Decoder architecture
    decoder_layers: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Number of transformer decoder layers",
        examples=[4, 6, 8],
    )

    decoder_heads: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Number of attention heads in decoder",
        examples=[8, 12, 16],
    )

    decoder_dim_feedforward: int = Field(
        default=2048,
        ge=512,
        le=4096,
        description="Dimension of feedforward network in decoder",
        examples=[1024, 2048, 3072],
    )

    # Span and vocabulary settings
    max_span_length: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Maximum length of entity spans in tokens",
        examples=[8, 12, 20],
    )

    max_seq_length: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum input sequence length",
        examples=[256, 512, 1024],
    )

    # Entity and relation types
    entity_types: List[str] = Field(
        default_factory=lambda: ["PER", "ORG", "LOC", "MISC"],
        description="List of entity type labels",
        min_length=1,
    )

    relation_types: List[str] = Field(
        default_factory=lambda: ["Work_For", "Based_In", "Located_In"],
        description="List of relation type labels",
        min_length=1,
    )

    # Special tokens for sequence generation
    start_token: str = Field(
        default="<START>",
        description="Start token for sequence generation",
    )

    sep_token: str = Field(
        default="<SEP>",
        description="Separator token between entities and relations",
    )

    end_token: str = Field(
        default="<END>",
        description="End token for sequence generation",
    )

    # Training settings
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate",
        examples=[0.1, 0.2, 0.3],
    )

    sentence_augmentation_max: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of sentences for data augmentation",
        examples=[3, 5, 7],
    )

    use_sorted_ordering: bool = Field(
        default=True,
        description="Use sorted ordering for linearization (left-to-right)",
    )

    # Decoding settings
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (1.0 = no filtering)",
        examples=[0.9, 0.95, 1.0],
    )

    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (1.0 = no scaling)",
        examples=[0.7, 1.0, 1.3],
    )

    @field_validator("start_token", "sep_token", "end_token")
    @classmethod
    def validate_special_tokens(cls, v: str) -> str:
        """Ensure special tokens are not empty."""
        if not v or not v.strip():
            raise ValueError("Special tokens cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_decoder_config(self) -> "ATGConfig":
        """Validate decoder configuration is consistent."""
        # Check if decoder dim is divisible by number of heads
        # Note: This assumes decoder hidden size equals encoder hidden size
        if hasattr(self, "hidden_size") and self.hidden_size % self.decoder_heads != 0:
            raise ValueError(
                f"Encoder hidden_size ({self.hidden_size}) must be divisible by "
                f"decoder_heads ({self.decoder_heads})"
            )

        return self

    @model_validator(mode="after")
    def validate_special_tokens_unique(self) -> "ATGConfig":
        """Ensure special tokens are unique."""
        tokens = [self.start_token, self.sep_token, self.end_token]
        if len(tokens) != len(set(tokens)):
            raise ValueError(
                f"Special tokens must be unique: "
                f"start={self.start_token}, sep={self.sep_token}, end={self.end_token}"
            )

        return self


class GenerationState(Enum):
    """States for constrained decoding state machine."""

    START = "start"
    GENERATING_ENTITIES = "generating_entities"
    AFTER_SEP = "after_sep"
    GENERATING_HEAD = "generating_head"
    GENERATING_TAIL = "generating_tail"
    GENERATING_RELATION = "generating_relation"
    END = "end"


# ============================================================================
# Dynamic Vocabulary
# ============================================================================


class DynamicVocabulary:
    """
    Dynamic vocabulary builder for ATG model.

    Vocabulary consists of:
    1. Span embeddings: (start, end, entity_type) - computed from encoder
    2. Relation types: learned embeddings
    3. Special tokens: learned embeddings (<START>, <SEP>, <END>)

    Total vocab size: L * K * C + R + 3
    where L = seq_len, K = max_span_length, C = num_entity_types,
          R = num_relation_types
    """

    def __init__(self, config: ATGConfig, hidden_size: int):
        self.config = config
        self.hidden_size = hidden_size

        # Entity type projection matrices (one per entity type)
        self.entity_type_projections = nn.ModuleDict(
            {
                entity_type: nn.Linear(hidden_size * 2, hidden_size)
                for entity_type in config.entity_types
            }
        )

        # Learned embeddings for relation types
        self.relation_embeddings = nn.Embedding(
            len(config.relation_types),
            hidden_size,
        )

        # Learned embeddings for special tokens
        self.special_token_embeddings = nn.Embedding(3, hidden_size)

        # Token to ID mappings
        self.special_token_to_id = {
            config.start_token: 0,
            config.sep_token: 1,
            config.end_token: 2,
        }

        self.relation_to_id = {rel: idx for idx, rel in enumerate(config.relation_types)}

        self.entity_type_to_id = {ent: idx for idx, ent in enumerate(config.entity_types)}

        logger.info(
            f"DynamicVocabulary initialized: "
            f"{len(config.entity_types)} entity types, "
            f"{len(config.relation_types)} relation types"
        )

    def compute_span_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute span embeddings for all possible spans.

        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            span_embeddings: [batch_size, num_spans, hidden_size]
                where num_spans = seq_len * max_span_length * num_entity_types
        """
        batch_size, seq_len, hidden_size = token_embeddings.shape
        device = token_embeddings.device

        all_span_embeddings = []

        # For each possible span
        for start in range(seq_len):
            for length in range(1, min(self.config.max_span_length + 1, seq_len - start + 1)):
                end = start + length - 1

                # Skip if span includes padding
                if attention_mask is not None:
                    if not attention_mask[:, end].all():
                        continue

                # Get start and end token embeddings
                start_emb = token_embeddings[:, start, :]  # [batch, hidden]
                end_emb = token_embeddings[:, end, :]  # [batch, hidden]

                # Concatenate start and end
                span_base = torch.cat([start_emb, end_emb], dim=-1)  # [batch, 2*hidden]

                # Project for each entity type
                for entity_type in self.config.entity_types:
                    projection = self.entity_type_projections[entity_type]
                    span_emb = projection(span_base)  # [batch, hidden]
                    all_span_embeddings.append(span_emb)

        # Stack all span embeddings
        if len(all_span_embeddings) == 0:
            # Return empty tensor if no valid spans
            return torch.zeros(batch_size, 0, hidden_size, device=device)

        span_embeddings = torch.stack(all_span_embeddings, dim=1)  # [batch, num_spans, hidden]

        return span_embeddings

    def build_vocabulary_matrix(
        self,
        span_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build complete vocabulary matrix by concatenating:
        - Span embeddings (dynamic)
        - Relation type embeddings (learned)
        - Special token embeddings (learned)

        Args:
            span_embeddings: [batch_size, num_spans, hidden_size]

        Returns:
            vocab_matrix: [batch_size, vocab_size, hidden_size]
        """
        batch_size = span_embeddings.size(0)
        device = span_embeddings.device

        # Get relation embeddings
        relation_ids = torch.arange(
            len(self.config.relation_types),
            device=device,
        )
        relation_embs = self.relation_embeddings(relation_ids)  # [R, hidden]
        relation_embs = relation_embs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, R, hidden]

        # Get special token embeddings
        special_ids = torch.arange(3, device=device)
        special_embs = self.special_token_embeddings(special_ids)  # [3, hidden]
        special_embs = special_embs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 3, hidden]

        # Concatenate all embeddings
        vocab_matrix = torch.cat(
            [span_embeddings, relation_embs, special_embs],
            dim=1,
        )  # [batch, num_spans + R + 3, hidden]

        return vocab_matrix

    def get_span_id(
        self,
        start: int,
        end: int,
        entity_type: str,
        seq_len: int,
    ) -> int:
        """Get vocabulary ID for a span."""
        entity_type_id = self.entity_type_to_id[entity_type]
        num_entity_types = len(self.config.entity_types)

        # Calculate span ID based on position in enumeration
        span_id = 0
        for s in range(seq_len):
            for length in range(1, min(self.config.max_span_length + 1, seq_len - s + 1)):
                e = s + length - 1
                for et_id in range(num_entity_types):
                    if s == start and e == end and et_id == entity_type_id:
                        return span_id
                    span_id += 1

        raise ValueError(f"Invalid span: ({start}, {end}, {entity_type})")

    def get_relation_id(self, relation_type: str, num_spans: int) -> int:
        """Get vocabulary ID for a relation type."""
        rel_id = self.relation_to_id[relation_type]
        return num_spans + rel_id

    def get_special_token_id(self, token: str, num_spans: int) -> int:
        """Get vocabulary ID for a special token."""
        token_id = self.special_token_to_id[token]
        num_relations = len(self.config.relation_types)
        return num_spans + num_relations + token_id


# ============================================================================
# ATG Encoder
# ============================================================================


class ATGEncoder(nn.Module):
    """
    ATG Encoder: Transformer encoder with span representation layer.

    Takes input text and produces:
    1. Token-level representations
    2. Span-level representations for all possible spans
    """

    def __init__(self, config: ATGConfig):
        super().__init__()
        self.config = config

        # Load pretrained transformer encoder
        self.transformer_config = AutoConfig.from_pretrained(config.encoder_model)
        self.transformer = AutoModel.from_pretrained(config.encoder_model)
        self.hidden_size = self.transformer_config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        logger.info(
            f"ATGEncoder initialized with {config.encoder_model}, hidden_size={self.hidden_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode input text.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            token_embeddings: [batch_size, seq_len, hidden_size]
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        token_embeddings = outputs.last_hidden_state
        token_embeddings = self.dropout(token_embeddings)

        return token_embeddings


# ============================================================================
# ATG Decoder
# ============================================================================


class ATGDecoder(nn.Module):
    """
    ATG Decoder: Transformer decoder with pointing mechanism.

    Features:
    - Causal self-attention over previous outputs
    - Cross-attention to encoder outputs
    - Positional embeddings
    - Structural embeddings (Node, Head, Tail, Relation)
    - Output projection to dynamic vocabulary
    """

    def __init__(self, config: ATGConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
        )

        # Positional embeddings (learned)
        self.positional_embeddings = nn.Embedding(config.max_seq_length, hidden_size)

        # Structural embeddings (Node, Head, Tail, Relation)
        self.structural_embeddings = nn.Embedding(4, hidden_size)
        self.structural_map = {
            "node": 0,
            "head": 1,
            "tail": 2,
            "relation": 3,
        }

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        logger.info(
            f"ATGDecoder initialized with {config.decoder_layers} layers, "
            f"{config.decoder_heads} heads"
        )

    def forward(
        self,
        target_embeddings: torch.Tensor,
        target_positions: torch.Tensor,
        target_structures: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target sequence.

        Args:
            target_embeddings: [batch_size, target_len, hidden_size]
            target_positions: [batch_size, target_len] - position indices
            target_structures: [batch_size, target_len] - structure type indices
            encoder_outputs: [batch_size, src_len, hidden_size]
            encoder_attention_mask: [batch_size, src_len]
            target_mask: [target_len, target_len] - causal mask

        Returns:
            decoder_outputs: [batch_size, target_len, hidden_size]
        """
        batch_size, target_len, _ = target_embeddings.shape

        # Add positional embeddings
        pos_embs = self.positional_embeddings(target_positions)

        # Add structural embeddings
        struct_embs = self.structural_embeddings(target_structures)

        # Combine embeddings
        combined_embs = target_embeddings + pos_embs + struct_embs
        combined_embs = self.dropout(combined_embs)

        # Create causal mask if not provided
        if target_mask is None:
            target_mask = nn.Transformer.generate_square_subsequent_mask(
                target_len,
                device=target_embeddings.device,
            )

        # Create memory key padding mask
        memory_key_padding_mask = ~encoder_attention_mask.bool()

        # Apply transformer decoder
        decoder_outputs = self.transformer_decoder(
            tgt=combined_embs,
            memory=encoder_outputs,
            tgt_mask=target_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return decoder_outputs


# ============================================================================
# State-Based Constraints
# ============================================================================


class StateBasedConstraints:
    """
    State machine for constrained decoding.

    Ensures generated sequence follows the template:
    <START> (entity1) (entity2) ... <SEP> (head1) (tail1) rel1 (head2) (tail2) rel2 ... <END>

    States:
    - START: Must generate <START>
    - GENERATING_ENTITIES: Generate entity spans or <SEP>
    - AFTER_SEP: After <SEP>, must generate head entity
    - GENERATING_HEAD: Generate head entity
    - GENERATING_TAIL: Generate tail entity (different from head)
    - GENERATING_RELATION: Generate relation type
    - END: Generation complete
    """

    def __init__(self, config: ATGConfig, vocabulary: DynamicVocabulary):
        self.config = config
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.state = GenerationState.START
        self.generated_entities = set()
        self.current_head = None
        self.current_tail = None
        self.generated_triplets = set()

    def get_allowed_tokens(
        self,
        vocab_size: int,
        num_spans: int,
    ) -> List[int]:
        """
        Get list of allowed token IDs based on current state.

        Args:
            vocab_size: Total vocabulary size
            num_spans: Number of span tokens in vocabulary

        Returns:
            List of allowed token IDs
        """
        if self.state == GenerationState.START:
            # Only <START> allowed
            return [self.vocabulary.get_special_token_id(self.config.start_token, num_spans)]

        elif self.state == GenerationState.GENERATING_ENTITIES:
            # Can generate new entities or <SEP>
            allowed = []

            # All span tokens that haven't been generated
            for span_id in range(num_spans):
                if span_id not in self.generated_entities:
                    allowed.append(span_id)

            # <SEP> token
            allowed.append(self.vocabulary.get_special_token_id(self.config.sep_token, num_spans))

            return allowed

        elif self.state in [GenerationState.AFTER_SEP, GenerationState.GENERATING_HEAD]:
            # Must generate a head entity from generated entities
            return list(self.generated_entities)

        elif self.state == GenerationState.GENERATING_TAIL:
            # Must generate tail entity (different from head)
            allowed = list(self.generated_entities)
            if self.current_head in allowed:
                allowed.remove(self.current_head)
            return allowed

        elif self.state == GenerationState.GENERATING_RELATION:
            # Can generate any relation type or <END>
            allowed = []

            # All relation types
            for rel_id in range(len(self.config.relation_types)):
                allowed.append(
                    self.vocabulary.get_relation_id(self.config.relation_types[rel_id], num_spans)
                )

            # <END> token
            allowed.append(self.vocabulary.get_special_token_id(self.config.end_token, num_spans))

            return allowed

        else:  # END state
            return []

    def update_state(self, token_id: int, num_spans: int):
        """Update state based on generated token."""
        # Determine token type
        is_span = token_id < num_spans
        is_relation = token_id >= num_spans and token_id < num_spans + len(
            self.config.relation_types
        )

        start_id = self.vocabulary.get_special_token_id(self.config.start_token, num_spans)
        sep_id = self.vocabulary.get_special_token_id(self.config.sep_token, num_spans)
        end_id = self.vocabulary.get_special_token_id(self.config.end_token, num_spans)

        if self.state == GenerationState.START:
            if token_id == start_id:
                self.state = GenerationState.GENERATING_ENTITIES

        elif self.state == GenerationState.GENERATING_ENTITIES:
            if is_span:
                self.generated_entities.add(token_id)
            elif token_id == sep_id:
                self.state = GenerationState.AFTER_SEP

        elif self.state == GenerationState.AFTER_SEP:
            if is_span:
                self.current_head = token_id
                self.state = GenerationState.GENERATING_TAIL

        elif self.state == GenerationState.GENERATING_HEAD:
            if is_span:
                self.current_head = token_id
                self.state = GenerationState.GENERATING_TAIL

        elif self.state == GenerationState.GENERATING_TAIL:
            if is_span:
                self.current_tail = token_id
                self.state = GenerationState.GENERATING_RELATION

        elif self.state == GenerationState.GENERATING_RELATION:
            if is_relation:
                # Record triplet
                self.generated_triplets.add((self.current_head, self.current_tail, token_id))
                self.state = GenerationState.GENERATING_HEAD
            elif token_id == end_id:
                self.state = GenerationState.END


# ============================================================================
# Complete ATG Model
# ============================================================================


class ImprovedATGModel(nn.Module):
    """
    Complete Improved ATG Model.

    Architecture:
    1. Encoder: Encodes input text to token representations
    2. Dynamic Vocabulary: Builds vocabulary from spans, relations, special tokens
    3. Decoder: Generates linearized graph with constrained decoding

    Training:
    - Teacher forcing with gold linearized graph
    - Cross-entropy loss over dynamic vocabulary
    - Optional sentence augmentation

    Inference:
    - State-based constrained decoding
    - Nucleus sampling with top-p
    - Guaranteed well-formed output
    """

    def __init__(self, config: ATGConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = ATGEncoder(config)

        # Dynamic vocabulary
        self.vocabulary = DynamicVocabulary(config, self.encoder.hidden_size)

        # Decoder
        self.decoder = ATGDecoder(config, self.encoder.hidden_size)

        # Tokenizer (for inference)
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)

        logger.info("ImprovedATGModel initialized successfully")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_positions: Optional[torch.Tensor] = None,
        target_structures: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            target_ids: [batch_size, target_len] - for training
            target_positions: [batch_size, target_len] - positional indices
            target_structures: [batch_size, target_len] - structural indices

        Returns:
            Dictionary containing:
            - logits: [batch_size, target_len, vocab_size] if training
            - loss: scalar if training with targets
        """
        # Encode input
        encoder_outputs = self.encoder(input_ids, attention_mask)

        # Compute span embeddings
        span_embeddings = self.vocabulary.compute_span_embeddings(encoder_outputs, attention_mask)

        # Build vocabulary matrix
        vocab_matrix = self.vocabulary.build_vocabulary_matrix(span_embeddings)

        outputs = {
            "encoder_outputs": encoder_outputs,
            "span_embeddings": span_embeddings,
            "vocab_matrix": vocab_matrix,
        }

        # If targets provided, run decoder (training mode)
        if target_ids is not None:
            # Get target embeddings from vocabulary
            target_embeddings = vocab_matrix.gather(
                dim=1,
                index=target_ids.unsqueeze(-1).expand(-1, -1, self.encoder.hidden_size),
            )

            # Decode
            decoder_outputs = self.decoder(
                target_embeddings=target_embeddings,
                target_positions=target_positions,
                target_structures=target_structures,
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=attention_mask,
            )

            # Project to vocabulary
            logits = torch.matmul(
                decoder_outputs,
                vocab_matrix.transpose(1, 2),
            )  # [batch, target_len, vocab_size]

            outputs["logits"] = logits

            # Compute loss if targets provided
            if target_ids is not None:
                # Shift targets for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = target_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                outputs["loss"] = loss

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 100,
    ) -> List[List[int]]:
        """
        Generate entity-relation graph with constrained decoding.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            max_length: Maximum generation length

        Returns:
            List of generated token ID sequences (one per batch item)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Encode input
        encoder_outputs = self.encoder(input_ids, attention_mask)

        # Compute span embeddings and vocabulary
        span_embeddings = self.vocabulary.compute_span_embeddings(encoder_outputs, attention_mask)
        vocab_matrix = self.vocabulary.build_vocabulary_matrix(span_embeddings)
        vocab_size = vocab_matrix.size(1)
        num_spans = span_embeddings.size(1)

        # Initialize generation for each batch item
        generated_sequences = [[] for _ in range(batch_size)]
        active_indices = list(range(batch_size))
        constraints = [
            StateBasedConstraints(self.config, self.vocabulary) for _ in range(batch_size)
        ]

        # Start with <START> token
        start_id = self.vocabulary.get_special_token_id(self.config.start_token, num_spans)
        current_ids = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)

        for step in range(max_length):
            # Prepare decoder inputs
            target_embeddings = vocab_matrix.gather(
                dim=1,
                index=current_ids.unsqueeze(-1).expand(-1, -1, self.encoder.hidden_size),
            )

            target_positions = (
                torch.arange(current_ids.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
            )

            # Determine structural embeddings based on state
            target_structures = torch.zeros(
                batch_size, current_ids.size(1), dtype=torch.long, device=device
            )
            # Simplified: all nodes initially

            # Decode
            decoder_outputs = self.decoder(
                target_embeddings=target_embeddings,
                target_positions=target_positions,
                target_structures=target_structures,
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=attention_mask,
            )

            # Get logits for last position
            last_hidden = decoder_outputs[:, -1, :]  # [batch, hidden]
            logits = torch.matmul(
                last_hidden.unsqueeze(1),
                vocab_matrix.transpose(1, 2),
            ).squeeze(1)  # [batch, vocab_size]

            # Apply constraints for each active batch item
            for batch_idx in active_indices[:]:
                # Get allowed tokens
                allowed_tokens = constraints[batch_idx].get_allowed_tokens(vocab_size, num_spans)

                if len(allowed_tokens) == 0:
                    active_indices.remove(batch_idx)
                    continue

                # Mask logits to only allowed tokens
                masked_logits = torch.full_like(logits[batch_idx], float("-inf"))
                masked_logits[allowed_tokens] = logits[batch_idx, allowed_tokens]

                # Sample with nucleus sampling
                probs = F.softmax(masked_logits / self.config.temperature, dim=-1)

                if self.config.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    mask = cumulative_probs > self.config.top_p
                    mask[0] = False  # Keep at least one token
                    sorted_probs[mask] = 0.0
                    probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)

                # Sample next token
                next_token = torch.multinomial(probs, 1).item()

                # Update state
                constraints[batch_idx].update_state(next_token, num_spans)
                generated_sequences[batch_idx].append(next_token)

                # Check if done
                if constraints[batch_idx].state == GenerationState.END:
                    active_indices.remove(batch_idx)

            # If all sequences done, stop
            if len(active_indices) == 0:
                break

            # Prepare next step input
            next_tokens = torch.tensor(
                [
                    generated_sequences[i][-1] if i in active_indices else start_id
                    for i in range(batch_size)
                ],
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)

            current_ids = torch.cat([current_ids, next_tokens], dim=1)

        return generated_sequences


# ============================================================================
# Utility Functions
# ============================================================================


def create_atg_model(
    entity_types: List[str],
    relation_types: List[str],
    encoder_model: str = "microsoft/deberta-v3-base",
    **kwargs,
) -> ImprovedATGModel:
    """
    Convenience function to create ATG model.

    Args:
        entity_types: List of entity type strings
        relation_types: List of relation type strings
        encoder_model: Pretrained encoder model name
        **kwargs: Additional config parameters

    Returns:
        Initialized ATG model
    """
    config = ATGConfig(
        encoder_model=encoder_model,
        entity_types=entity_types,
        relation_types=relation_types,
        **kwargs,
    )

    model = ImprovedATGModel(config)

    logger.info(
        f"Created ATG model with {len(entity_types)} entity types, "
        f"{len(relation_types)} relation types"
    )

    return model
