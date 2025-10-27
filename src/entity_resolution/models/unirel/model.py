"""
UniRel: Unified Representation and Interaction for Joint Relational Triple Extraction.

Based on: "UniRel: Unified Representation and Interaction for Joint Relational Triple Extraction"
by Tang et al.

Key innovations:
1. Unified Representation: Encodes entities and relations in natural language sequences
2. Interaction Map: Models entity-entity and entity-relation interactions simultaneously
3. Joint extraction of relational triples <subject-relation-object>
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .config import UniRelConfig
from .interaction_map import InteractionDecoder, InteractionMap


class UniRelModel(nn.Module):
    """
    UniRel model for joint entity and relation extraction.

    Architecture:
    1. Encoder: BERT-based encoder for sequence representation
    2. Entity Tagger: Identifies entity spans in text
    3. Relation Verbalizer: Converts relation types to natural language
    4. Interaction Map: Models interactions between entities and relations
    5. Triple Decoder: Extracts <subject-relation-object> triples
    """

    def __init__(self, config: UniRelConfig):
        """
        Initialize UniRel model.

        Args:
            config: UniRelConfig object with model settings
        """
        super().__init__()
        self.config = config

        # Load encoder (BERT recommended)
        self.encoder = AutoModel.from_pretrained(config.encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)

        if config.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Entity tagging head (BIO tagging)
        self.entity_tagger = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, len(config.entity_types) * 3),  # B-I-O for each type
        )

        # Relation verbalization embeddings
        # Store verbalized relation representations
        self.relation_embeddings = nn.Parameter(
            torch.randn(len(config.relation_types), config.hidden_size)
        )

        # Interaction Map (core component)
        self.interaction_map = InteractionMap(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.interaction_dropout,
        )

        # Triple decoder
        self.triple_decoder = InteractionDecoder(
            entity_threshold=config.entity_threshold,
            relation_threshold=config.interaction_threshold,
            triple_threshold=config.triple_threshold,
            max_triples=config.max_triples_per_sentence,
        )

        # Loss function
        self.entity_loss_fn = nn.CrossEntropyLoss()
        self.interaction_loss_fn = nn.BCEWithLogitsLoss()

    def verbalize_relations(self) -> list[str]:
        """
        Convert relation types to natural language forms.

        Returns:
            List of verbalized relation strings
        """
        return [
            self.config.relation_verbalizations.get(rel, rel) for rel in self.config.relation_types
        ]

    def encode_text_with_relations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text with verbalized relations appended.

        The unified representation includes both the input text and all relation types
        in natural language, allowing the model to jointly reason about entities and relations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of:
            - Hidden states [batch_size, seq_len, hidden_size]
            - Relation positions [batch_size, num_relations] (boolean mask)
        """
        # Encode input text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        # In a complete implementation, we would append verbalized relations to the input
        # For now, we use learned relation embeddings
        batch_size = hidden_states.size(0)
        num_relations = len(self.config.relation_types)

        # Expand relation embeddings to batch
        relation_hidden = self.relation_embeddings.unsqueeze(0).expand(
            batch_size, num_relations, -1
        )  # [batch_size, num_relations, hidden_size]

        # Concatenate text and relation representations
        # Note: In practice, relations are tokenized and encoded, not just embeddings
        combined_hidden = torch.cat([hidden_states, relation_hidden], dim=1)

        # Update attention mask
        relation_mask = torch.ones(
            batch_size, num_relations, dtype=attention_mask.dtype, device=attention_mask.device
        )
        torch.cat([attention_mask, relation_mask], dim=1)

        # Create relation position mask
        seq_len = hidden_states.size(1)
        relation_positions = torch.zeros(
            batch_size, seq_len + num_relations, dtype=torch.bool, device=hidden_states.device
        )
        relation_positions[:, seq_len:] = True

        return combined_hidden, relation_positions

    def extract_entities(
        self,
        hidden_states: torch.Tensor,
        entity_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list]:
        """
        Extract entity spans from hidden states.

        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_size]
            entity_labels: Ground truth entity labels [batch_size, seq_len] (optional)

        Returns:
            Tuple of:
            - Entity logits [batch_size, seq_len, num_entity_tags]
            - Entity loss (if labels provided)
            - Entity positions (boolean mask)
        """
        # Predict entity tags (BIO scheme)
        entity_logits = self.entity_tagger(hidden_states)

        # Compute loss if labels provided
        entity_loss = None
        if entity_labels is not None:
            entity_loss = self.entity_loss_fn(
                entity_logits.view(-1, entity_logits.size(-1)), entity_labels.view(-1)
            )

        # Extract entity positions (for inference)
        # Find tokens tagged as B- or I- (not O)
        entity_preds = torch.argmax(entity_logits, dim=-1)

        # In BIO tagging: 0=O, odd indices=B-, even indices=I-
        # Entity positions are where tag != 0
        entity_positions = entity_preds != 0

        return entity_logits, entity_loss, entity_positions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_labels: Optional[torch.Tensor] = None,
        ee_labels: Optional[torch.Tensor] = None,
        er_labels: Optional[torch.Tensor] = None,
        return_triples: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of UniRel model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            entity_labels: Entity labels for training [batch_size, seq_len]
            ee_labels: Entity-entity interaction labels [batch_size, num_pairs]
            er_labels: Entity-relation interaction labels [batch_size, num_entities, num_relations]
            return_triples: Whether to return extracted triples (inference mode)

        Returns:
            Dictionary containing:
            - loss: Total loss (if labels provided)
            - entity_logits: Entity prediction logits
            - ee_scores: Entity-entity interaction scores
            - er_scores: Entity-relation interaction scores
            - triples: Extracted triples (if return_triples=True)
        """
        # Encode text with verbalized relations
        combined_hidden, relation_positions = self.encode_text_with_relations(
            input_ids, attention_mask
        )

        # Extract text-only hidden states for entity tagging
        seq_len = input_ids.size(1)
        text_hidden = combined_hidden[:, :seq_len, :]

        # Extract entities
        entity_logits, entity_loss, entity_positions = self.extract_entities(
            text_hidden, entity_labels
        )

        # Build Interaction Map
        # For interaction map, we need to handle entity and relation positions properly
        # Since entity_positions are indices into text_hidden, we pass text_hidden
        # Relations will be handled separately using the relation_embeddings
        batch_size = text_hidden.size(0)

        # Create a combined representation with text + relation embeddings
        # Expand relation embeddings to match batch size
        num_relations = len(self.config.relation_types)
        relation_hidden = self.relation_embeddings.unsqueeze(0).expand(
            batch_size, num_relations, -1
        )

        # Extract entity representations using the boolean mask from entity_positions
        # Then concatenate with relations for interaction computation
        if entity_positions.size(0) > 0 and entity_positions[0].any():
            # Get entity mask for first batch item
            entity_mask = entity_positions[0]  # [seq_len] boolean

            # Extract entity representations from text_hidden
            entity_reps = text_hidden[0, entity_mask, :]  # [num_entities, hidden_size]
            num_entities = entity_reps.size(0)

            # Get relation representations for first batch only
            relation_reps = relation_hidden[0]  # [num_relations, hidden_size]

            # Concatenate entities and relations
            interaction_input = torch.cat([entity_reps, relation_reps], dim=0).unsqueeze(
                0
            )  # [1, num_entities + num_relations, hidden_size]

            # Create position indices
            entity_pos_adjusted = torch.arange(
                num_entities, dtype=torch.long, device=text_hidden.device
            )
            relation_pos_adjusted = torch.arange(
                num_entities,
                num_entities + num_relations,
                dtype=torch.long,
                device=text_hidden.device,
            )
        else:
            # No entities, just use text_hidden
            interaction_input = text_hidden
            entity_pos_adjusted = None
            relation_pos_adjusted = None

        enhanced_hidden, ee_scores, er_scores = self.interaction_map(
            interaction_input,
            attention_mask=None,
            entity_positions=entity_pos_adjusted,
            relation_positions=relation_pos_adjusted,
        )

        # Compute interaction losses
        ee_loss = None
        er_loss = None

        if ee_labels is not None and ee_scores is not None:
            ee_loss = self.interaction_loss_fn(ee_scores, ee_labels.float())

        if er_labels is not None and er_scores is not None:
            er_loss = self.interaction_loss_fn(er_scores, er_labels.float())

        # Compute total loss
        total_loss = None
        if entity_loss is not None:
            total_loss = self.config.entity_loss_weight * entity_loss
            if ee_loss is not None:
                total_loss += self.config.interaction_loss_weight * ee_loss
            if er_loss is not None:
                total_loss += self.config.relation_loss_weight * er_loss

        # Decode triples (inference mode)
        triples = None
        if return_triples and ee_scores is not None and er_scores is not None:
            # Get entity spans from predictions
            entity_preds = torch.argmax(entity_logits[0], dim=-1)
            entity_spans = self._extract_entity_spans(entity_preds, input_ids[0])

            triples = self.triple_decoder(
                ee_scores,
                er_scores,
                entity_spans,
                self.config.relation_types,
            )

        return {
            "loss": total_loss,
            "entity_logits": entity_logits,
            "ee_scores": ee_scores,
            "er_scores": er_scores,
            "triples": triples,
        }

    def _extract_entity_spans(
        self,
        entity_preds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> list[str]:
        """
        Extract entity spans from BIO predictions.

        Args:
            entity_preds: Entity predictions [seq_len]
            input_ids: Input token IDs [seq_len]

        Returns:
            List of entity text spans
        """
        spans = []
        current_span = []

        for i, pred in enumerate(entity_preds):
            if pred == 0:  # O tag
                if current_span:
                    # End current span
                    span_text = self.tokenizer.decode(current_span)
                    spans.append(span_text.strip())
                    current_span = []
            elif pred % 2 == 1:  # B- tag (odd indices)
                if current_span:
                    # End previous span
                    span_text = self.tokenizer.decode(current_span)
                    spans.append(span_text.strip())
                # Start new span
                current_span = [input_ids[i].item()]
            else:  # I- tag (even indices)
                if current_span:
                    current_span.append(input_ids[i].item())

        # Handle final span
        if current_span:
            span_text = self.tokenizer.decode(current_span)
            spans.append(span_text.strip())

        return spans

    def predict(
        self,
        text: str,
        device: Optional[torch.device] = None,
    ) -> list[tuple[str, str, str]]:
        """
        Extract triples from input text.

        Args:
            text: Input text
            device: Device to run inference on

        Returns:
            List of extracted triples [(subject, relation, object), ...]
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Run model
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_triples=True,
            )

        return outputs["triples"] or []


def create_unirel_model(config: UniRelConfig) -> UniRelModel:
    """
    Factory function to create UniRel model.

    Args:
        config: UniRelConfig object

    Returns:
        Initialized UniRelModel
    """
    return UniRelModel(config)
