"""
ReLiK Reader: Single-pass entity linking and relation extraction.

Implements a Reader that takes text and retrieved candidates and performs
entity linking and/or relation extraction in a single forward pass.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ReLiKReader(nn.Module):
    """
    ReLiK Reader for entity linking and relation extraction.

    Performs mention detection, entity linking, and relation extraction
    in a single forward pass using contextualized representations.
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
    ):
        """
        Initialize the ReLiK Reader.

        Args:
            model_name: Pre-trained model name
            max_seq_length: Maximum sequence length
            num_entity_types: Number of entity types
            num_relation_types: Number of relation types
            dropout: Dropout rate
            use_entity_linking: Whether to enable entity linking
            use_relation_extraction: Whether to enable relation extraction
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types
        self.use_entity_linking = use_entity_linking
        self.use_relation_extraction = use_relation_extraction

        # Load encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size

        # Span detection layers
        self.span_start = nn.Linear(self.hidden_size, 2)
        self.span_end = nn.Linear(self.hidden_size * 2, 2)

        # Entity linking layers
        if use_entity_linking:
            self.entity_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Relation extraction layers
        if use_relation_extraction:
            self.subject_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.object_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.relation_projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.relation_classifier = nn.Linear(self.hidden_size, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_passages: Optional[int] = None,
        passage_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Reader.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            num_passages: Number of retrieved passages
            passage_mask: Mask indicating passage positions [batch, num_passages]

        Returns:
            Dictionary containing:
                - span_start_logits: Logits for span starts
                - span_end_logits: Logits for span ends
                - entity_logits: Logits for entity linking (if enabled)
                - relation_logits: Logits for relation extraction (if enabled)
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        hidden_states = self.dropout(hidden_states)

        # Get sequence length (query portion)
        batch_size, seq_len, _ = hidden_states.shape

        # Span start detection
        span_start_logits = self.span_start(hidden_states)  # [batch, seq_len, 2]

        # Span end detection (conditioned on start)
        # For simplicity, we compute all possible (start, end) pairs
        # In practice, we only compute for valid starts during inference
        span_end_logits = []
        for s in range(seq_len):
            start_hidden = hidden_states[:, s : s + 1, :].expand(-1, seq_len, -1)
            combined = torch.cat([start_hidden, hidden_states], dim=-1)
            end_logits = self.span_end(combined)  # [batch, seq_len, 2]
            span_end_logits.append(end_logits)

        span_end_logits = torch.stack(span_end_logits, dim=1)  # [batch, seq_len, seq_len, 2]

        result = {
            "hidden_states": hidden_states,
            "span_start_logits": span_start_logits,
            "span_end_logits": span_end_logits,
        }

        # Entity linking
        if self.use_entity_linking and num_passages is not None:
            # Extract special token positions for passages
            # Assuming format: text [SEP] <ST0> p0 <ST1> p1 ...
            # We need to identify <STi> positions

            # For now, simplified: project mentions and passages
            # This would be more sophisticated in a full implementation
            result["entity_logits"] = None  # Placeholder

        # Relation extraction
        if self.use_relation_extraction:
            result["relation_logits"] = None  # Placeholder

        return result

    def predict_spans(
        self,
        text: str,
        span_threshold: float = 0.5,
    ) -> List[Tuple[int, int, str]]:
        """
        Predict entity spans in text.

        Args:
            text: Input text
            span_threshold: Threshold for span detection

        Returns:
            List of (start_idx, end_idx, span_text) tuples
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Move to device
        input_ids = encoded["input_ids"].to(self.encoder.device)
        attention_mask = encoded["attention_mask"].to(self.encoder.device)
        offsets = encoded["offset_mapping"][0]

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        # Get predictions
        span_start_probs = torch.softmax(outputs["span_start_logits"][0], dim=-1)
        span_end_logits = outputs["span_end_logits"][0]

        # Find valid starts
        valid_starts = []
        for i in range(len(span_start_probs)):
            if span_start_probs[i, 1] > span_threshold:
                valid_starts.append(i)

        # Find valid ends for each start
        spans = []
        for start_idx in valid_starts:
            span_end_probs = torch.softmax(span_end_logits[start_idx], dim=-1)

            for end_idx in range(start_idx, len(span_end_probs)):
                if span_end_probs[end_idx, 1] > span_threshold:
                    # Get character offsets
                    start_char = offsets[start_idx][0].item()
                    end_char = offsets[end_idx][1].item()

                    if start_char < end_char:
                        span_text = text[start_char:end_char]
                        spans.append((start_char, end_char, span_text))
                        break  # Take first valid end

        return spans

    def link_entities(
        self,
        text: str,
        candidates: List[Dict[str, Any]],
        entity_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Link entities in text to candidates.

        Args:
            text: Input text
            candidates: List of candidate entities
            entity_threshold: Threshold for entity linking

        Returns:
            List of linked entities with spans and IDs
        """
        # First detect spans
        spans = self.predict_spans(text)

        # Then link to candidates
        # This is a simplified placeholder
        results = []
        for start, end, span_text in spans:
            # In a full implementation, we would:
            # 1. Encode text with all candidates
            # 2. Compute linking scores
            # 3. Select best candidate

            # For now, just return spans with placeholder entity IDs
            results.append(
                {
                    "start": start,
                    "end": end,
                    "text": span_text,
                    "entity_id": None,
                    "entity_name": None,
                    "score": 1.0,
                }
            )

        return results

    def extract_relations(
        self,
        text: str,
        relation_types: List[str],
        relation_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Extract relations from text.

        Args:
            text: Input text
            relation_types: List of possible relation types
            relation_threshold: Threshold for relation extraction

        Returns:
            List of extracted relations (subject, relation, object)
        """
        # First detect entity spans
        spans = self.predict_spans(text)

        # Then extract relations between span pairs
        # This is a simplified placeholder
        results = []

        # In a full implementation, we would:
        # 1. Encode text with relation type candidates
        # 2. For each pair of spans, compute relation scores
        # 3. Extract relations above threshold

        return results
