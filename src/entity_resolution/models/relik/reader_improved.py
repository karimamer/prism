"""
Improved ReLiK Reader: Enhanced span extraction and entity linking.

Implements proper candidate encoding with special tokens and efficient
span detection following the official ReLiK implementation.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .tokenizer import ReLiKTokenizer


class ImprovedReLiKReader(nn.Module):
    """
    Improved ReLiK Reader with proper span extraction and candidate encoding.

    Key improvements over basic reader:
    1. Uses ReLiKTokenizer for special token handling
    2. Efficient span detection (only valid starts)
    3. Proper entity disambiguation head
    4. Mention-candidate interaction through attention
    5. Top-k candidate ranking with confidence scores
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_seq_length: int = 1024,
        num_entity_types: int = 4,
        num_relation_types: int = 3,
        dropout: float = 0.1,
        use_entity_linking: bool = True,
        use_relation_extraction: bool = False,
        gradient_checkpointing: bool = False,
        max_span_length: int = 10,
        span_start_threshold: float = 0.1,
    ):
        """
        Initialize improved ReLiK Reader.

        Args:
            model_name: Pre-trained model name
            max_seq_length: Maximum sequence length
            num_entity_types: Number of entity types
            num_relation_types: Number of relation types
            dropout: Dropout rate
            use_entity_linking: Whether to enable entity linking
            use_relation_extraction: Whether to enable relation extraction
            gradient_checkpointing: Whether to use gradient checkpointing
            max_span_length: Maximum span length to consider
            span_start_threshold: Minimum probability for span start during inference
        """
        super().__init__()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types
        self.use_entity_linking = use_entity_linking
        self.use_relation_extraction = use_relation_extraction
        self.max_span_length = max_span_length
        self.span_start_threshold = span_start_threshold

        # Initialize tokenizer with special tokens
        self.tokenizer = ReLiKTokenizer(model_name)

        # Load encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        # Resize token embeddings for special tokens
        self.encoder.resize_token_embeddings(self.tokenizer.get_vocab_size())

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size

        # Span detection heads (separate classifiers for start and end)
        self.span_start_classifier = nn.Linear(self.hidden_size, 1)
        self.span_end_classifier = nn.Linear(self.hidden_size, 1)

        # Entity disambiguation head (for linking to candidates)
        if use_entity_linking:
            self.entity_disambiguation = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, 1),
            )

        # Relation extraction heads
        if use_relation_extraction:
            self.relation_scorer = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, 1),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_end: Optional[int] = None,
        marker_positions: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with proper candidate encoding.

        Args:
            input_ids: Input token IDs [batch, seq_len]
                      Format: text [SEP] <ST0> candidate0 <ST1> candidate1 ...
            attention_mask: Attention mask [batch, seq_len]
            text_end: Position where text ends (SEP token)
            marker_positions: Positions of <STi> tokens [batch, num_candidates]

        Returns:
            Dictionary containing:
                - span_start_logits: Start position scores [batch, text_length]
                - span_end_logits: End position scores (per start)
                - entity_logits: Entity disambiguation scores
                - hidden_states: Full hidden states
                - text_length: Length of text portion
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        hidden_states = self.dropout(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape

        # Determine text boundary
        if text_end is None:
            # Find SEP token
            sep_token_id = self.tokenizer.sep_token_id
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)
            if len(sep_positions[1]) > 0:
                text_end = sep_positions[1][0].item()
            else:
                text_end = seq_len

        # Extract text portion (before [SEP])
        text_hidden = hidden_states[:, :text_end, :]  # [batch, text_len, hidden]

        # Span start predictions (over text only)
        span_start_logits = self.span_start_classifier(text_hidden).squeeze(-1)
        # [batch, text_length]

        # For training: return logits for all positions
        # For inference: compute end logits only for high-probability starts
        result = {
            "span_start_logits": span_start_logits,
            "hidden_states": hidden_states,
            "text_length": text_end,
        }

        # Entity disambiguation (if candidates provided)
        if self.use_entity_linking and marker_positions is not None:
            # Extract candidate embeddings
            # Each candidate is represented by the <STi> marker token
            candidate_embeddings = []

            for batch_idx in range(batch_size):
                batch_cand_embs = []
                for cand_idx in range(marker_positions.size(1)):
                    marker_pos = marker_positions[batch_idx, cand_idx].item()
                    if marker_pos >= 0 and marker_pos < seq_len:
                        # Get embedding at marker position
                        cand_emb = hidden_states[batch_idx, marker_pos, :]
                        batch_cand_embs.append(cand_emb)

                if batch_cand_embs:
                    candidate_embeddings.append(torch.stack(batch_cand_embs))
                else:
                    # No valid candidates
                    candidate_embeddings.append(
                        torch.zeros(1, self.hidden_size, device=hidden_states.device)
                    )

            # Store candidate embeddings for later use
            result["candidate_embeddings"] = candidate_embeddings
            result["marker_positions"] = marker_positions

        return result

    def predict_spans_with_linking(
        self,
        text: str,
        candidates: list[dict[str, Any]],
        span_threshold: float = 0.5,
        entity_threshold: float = 0.3,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Predict spans and link to candidates.

        This is the main inference method that combines span detection
        and entity linking in a single pass.

        Args:
            text: Input text
            candidates: List of candidate entities with 'id' and 'text' fields
            span_threshold: Threshold for span detection
            entity_threshold: Threshold for entity linking
            top_k: Number of top candidates to return per span

        Returns:
            List of detected spans with linked entities
        """
        # Encode text with candidates
        candidate_texts = [c.get("text", c.get("name", "")) for c in candidates]
        encoded = self.tokenizer.encode_with_candidates(
            text,
            candidate_texts,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Move to device
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        marker_positions = encoded["marker_positions"].unsqueeze(0).to(device)
        text_end = encoded["text_end"]
        offsets = encoded["offset_mapping"][0]

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask,
                text_end=text_end,
                marker_positions=marker_positions,
            )

        # Get span start probabilities
        span_start_probs = torch.sigmoid(outputs["span_start_logits"][0])

        # Find valid starts
        valid_starts = (span_start_probs > self.span_start_threshold).nonzero(as_tuple=True)[0]

        detected_spans = []

        # Extract candidate embeddings
        if "candidate_embeddings" in outputs:
            candidate_embs = outputs["candidate_embeddings"][0]  # [num_candidates, hidden]
        else:
            candidate_embs = None

        # For each valid start, find best end and link to entities
        for start_pos in valid_starts:
            start_pos = start_pos.item()

            if span_start_probs[start_pos] < span_threshold:
                continue

            # Get hidden state at start position
            start_hidden = outputs["hidden_states"][0, start_pos, :]

            # Find best end position (within max_span_length)
            best_end_pos = None
            best_end_score = 0.0

            max_end = min(start_pos + self.max_span_length, text_end)

            for end_pos in range(start_pos, max_end):
                end_hidden = outputs["hidden_states"][0, end_pos, :]

                # Compute end score using interaction
                combined = start_hidden * end_hidden  # Element-wise product
                end_score = torch.sigmoid(self.span_end_classifier(combined)).item()

                if end_score > best_end_score:
                    best_end_score = end_score
                    best_end_pos = end_pos

            if best_end_pos is None or best_end_score < span_threshold:
                continue

            # Get character offsets
            start_char = offsets[start_pos][0].item()
            end_char = offsets[best_end_pos][1].item()

            if start_char >= end_char or end_char > len(text):
                continue

            span_text = text[start_char:end_char]

            # Compute mention embedding (mean of span)
            mention_emb = outputs["hidden_states"][0, start_pos : best_end_pos + 1, :].mean(dim=0)

            # Link to candidates
            linked_entities = []

            if candidate_embs is not None and self.use_entity_linking:
                # Compute entity disambiguation scores
                num_candidates = candidate_embs.size(0)

                for cand_idx in range(num_candidates):
                    cand_emb = candidate_embs[cand_idx]

                    # Combine mention and candidate
                    combined = torch.cat([mention_emb, cand_emb])

                    # Score
                    score = torch.sigmoid(self.entity_disambiguation(combined)).item()

                    if score >= entity_threshold:
                        linked_entities.append(
                            {
                                "entity_id": candidates[cand_idx].get("id"),
                                "entity_name": candidates[cand_idx].get("name", ""),
                                "entity_text": candidates[cand_idx].get("text", ""),
                                "score": score,
                            }
                        )

                # Sort by score and take top-k
                linked_entities.sort(key=lambda x: x["score"], reverse=True)
                linked_entities = linked_entities[:top_k]

            detected_spans.append(
                {
                    "start": start_char,
                    "end": end_char,
                    "text": span_text,
                    "span_score": (span_start_probs[start_pos].item() + best_end_score) / 2,
                    "candidates": linked_entities,
                    "best_entity": linked_entities[0] if linked_entities else None,
                }
            )

        return detected_spans

    def predict_spans(
        self,
        text: str,
        span_threshold: float = 0.5,
    ) -> list[tuple[int, int, str]]:
        """
        Predict entity spans in text (without linking).

        Args:
            text: Input text
            span_threshold: Threshold for span detection

        Returns:
            List of (start_idx, end_idx, span_text) tuples
        """
        # Encode without candidates
        encoded = self.tokenizer.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Move to device
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        offsets = encoded["offset_mapping"][0]

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        # Get span start probabilities
        span_start_probs = torch.sigmoid(outputs["span_start_logits"][0])

        # Find valid starts
        valid_starts = (span_start_probs > self.span_start_threshold).nonzero(as_tuple=True)[0]

        spans = []

        # For each valid start, find best end
        for start_pos in valid_starts:
            start_pos = start_pos.item()

            if span_start_probs[start_pos] < span_threshold:
                continue

            # Get hidden state at start
            start_hidden = outputs["hidden_states"][0, start_pos, :]

            best_end_pos = None
            best_end_score = 0.0

            text_length = outputs["text_length"]
            max_end = min(start_pos + self.max_span_length, text_length)

            for end_pos in range(start_pos, max_end):
                end_hidden = outputs["hidden_states"][0, end_pos, :]
                combined = start_hidden * end_hidden
                end_score = torch.sigmoid(self.span_end_classifier(combined)).item()

                if end_score > best_end_score:
                    best_end_score = end_score
                    best_end_pos = end_pos

            if best_end_pos is None or best_end_score < span_threshold:
                continue

            # Get character offsets
            start_char = offsets[start_pos][0].item()
            end_char = offsets[best_end_pos][1].item()

            if start_char < end_char and end_char <= len(text):
                span_text = text[start_char:end_char]
                spans.append((start_char, end_char, span_text))

        return spans


__all__ = ["ImprovedReLiKReader"]
