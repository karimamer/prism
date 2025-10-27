"""
SPEL Model: Structured Prediction for Entity Linking.

Main model implementation combining RoBERTa encoder with classification head
for token-level entity prediction.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (AutoModel, AutoTokenizer, RobertaModel,
                          RobertaTokenizer)

from .aggregation import PredictionAggregator
from .candidate_sets import CandidateSetManager
from .config import SPELConfig

logger = logging.getLogger(__name__)


class SPELModel(nn.Module):
    """
    SPEL: Structured Prediction for Entity Linking.

    Uses token-level classification for entity prediction with:
    - RoBERTa encoder for contextual representations
    - Classification head over fixed candidate set
    - Context-sensitive prediction aggregation
    - Hard negative mining during training
    """

    def __init__(self, config: SPELConfig):
        """
        Initialize SPEL model.

        Args:
            config: SPEL configuration
        """
        super().__init__()

        self.config = config

        # Load encoder (RoBERTa recommended)
        self.encoder = AutoModel.from_pretrained(config.encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)

        # Get hidden size from encoder
        self.hidden_size = self.encoder.config.hidden_size

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Initialize candidate set manager
        self.candidate_manager = None

        # Classification head (will be initialized after loading candidates)
        self.classification_head = None

        # Prediction aggregator
        self.aggregator = PredictionAggregator(
            tokenizer=self.tokenizer,
            filter_punctuation=config.filter_single_punctuation,
            filter_function_words=config.filter_function_words,
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def initialize_classification_head(self, vocab_size: int):
        """
        Initialize classification head.

        Args:
            vocab_size: Number of entities in vocabulary (including O token)
        """
        self.classification_head = nn.Linear(self.hidden_size, vocab_size, bias=False)
        logger.info(f"Initialized classification head with vocab size: {vocab_size}")

    def load_candidate_sets(
        self,
        fixed_candidates: Optional[List[str]] = None,
        mention_candidates: Optional[Dict[str, List[str]]] = None,
        entity_frequencies: Optional[Dict[str, int]] = None,
    ):
        """
        Load candidate sets and initialize classification head.

        Args:
            fixed_candidates: List of entity IDs for fixed candidate set
            mention_candidates: Dict mapping mention to candidate entity IDs
            entity_frequencies: Dict mapping entity ID to frequency (for building fixed set)
        """
        # Initialize candidate manager
        self.candidate_manager = CandidateSetManager(
            fixed_candidates=fixed_candidates,
            mention_candidates=mention_candidates,
        )

        # Build from frequencies if provided
        if entity_frequencies and not fixed_candidates:
            self.candidate_manager.build_from_frequency(
                entity_frequencies,
                top_k=self.config.fixed_candidate_set_size,
            )

        # Initialize classification head
        vocab_size = self.candidate_manager.get_vocab_size()
        self.initialize_classification_head(vocab_size)

        logger.info(f"Loaded candidate sets with {vocab_size} entities")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Gold entity indices for each token [batch, seq_len]
            hard_negatives: Hard negative entity indices [batch, seq_len, num_hard_neg]

        Returns:
            Dictionary with:
                - logits: Classification logits [batch, seq_len, vocab_size]
                - loss: BCE loss (if labels provided)
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        hidden_states = self.dropout(hidden_states)

        # Classify each token
        logits = self.classification_head(hidden_states)  # [batch, seq_len, vocab_size]

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = self.compute_loss(logits, labels, attention_mask, hard_negatives)
            result["loss"] = loss

        return result

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss with hard negative mining.

        Args:
            logits: Predicted logits [batch, seq_len, vocab_size]
            labels: Gold entity indices [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            hard_negatives: Hard negative indices [batch, seq_len, num_hard_neg]

        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Create binary labels for each entity in vocabulary
        # 1 for correct entity, 0 for others
        targets = torch.zeros_like(logits)

        # Set positive labels
        for b in range(batch_size):
            for t in range(seq_len):
                if attention_mask[b, t] > 0 and labels[b, t] >= 0:
                    targets[b, t, labels[b, t]] = 1.0

        # Binary cross-entropy with logits
        loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fct(logits, targets)

        # Mask padding tokens
        loss = loss * attention_mask.unsqueeze(-1)

        # Average loss
        return loss.sum() / attention_mask.sum()

    def predict(
        self,
        text: str,
        use_mention_specific_candidates: bool = False,
        return_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Predict entities in text.

        Args:
            text: Input text
            use_mention_specific_candidates: Whether to use mention-specific candidates
            return_scores: Whether to return prediction scores

        Returns:
            List of entity predictions with format:
            [{"start": char_idx, "end": char_idx, "entity": entity_id, "score": prob}, ...]
        """
        if self.candidate_manager is None:
            raise RuntimeError("Candidate sets not loaded. Call load_candidate_sets() first.")

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.encoder.device)
        attention_mask = encoding["attention_mask"].to(self.encoder.device)
        offsets = encoding["offset_mapping"][0]

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"][0]  # [seq_len, vocab_size]

        # Get top-k predictions for each token
        probs = torch.sigmoid(logits)
        top_k = 3
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)

        # Convert to entity IDs
        subword_predictions = []
        for t in range(len(top_indices)):
            token_preds = []
            for k in range(len(top_indices[t])):
                entity_idx = top_indices[t][k].item()
                entity_id = self.candidate_manager.get_entity_from_idx(entity_idx)
                prob = top_probs[t][k].item()

                if entity_id:
                    token_preds.append((entity_id, prob))

            subword_predictions.append(token_preds)

        # Map subwords to words (simplified - would need proper word boundary detection)
        subword_to_word = list(range(len(subword_predictions)))

        # Aggregate predictions
        spans = self.aggregator.aggregate_subword_predictions(
            text=text,
            subword_predictions=subword_predictions,
            subword_to_word_map=subword_to_word,
            top_k=top_k,
        )

        return spans

    def save(self, path: str):
        """Save model."""
        import os

        os.makedirs(path, exist_ok=True)

        # Save encoder and tokenizer
        self.encoder.save_pretrained(os.path.join(path, "encoder"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

        # Save classification head
        if self.classification_head is not None:
            torch.save(
                self.classification_head.state_dict(), os.path.join(path, "classification_head.pt")
            )

        # Save candidate sets
        if self.candidate_manager is not None:
            self.candidate_manager.save(os.path.join(path, "candidates"))

        # Save config
        import json

        config_dict = {
            "encoder_model": self.config.encoder_model,
            "max_seq_length": self.config.max_seq_length,
            "fixed_candidate_set_size": self.config.fixed_candidate_set_size,
            "dropout": self.config.dropout,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved SPEL model to {path}")

    @classmethod
    def load(cls, path: str) -> "SPELModel":
        """Load model from path."""
        import json
        import os

        # Load config
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)

        config = SPELConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load encoder (already loaded in __init__)

        # Load candidate sets
        candidate_path = os.path.join(path, "candidates")
        if os.path.exists(candidate_path):
            model.candidate_manager = CandidateSetManager.load(candidate_path)

            # Initialize and load classification head
            vocab_size = model.candidate_manager.get_vocab_size()
            model.initialize_classification_head(vocab_size)

            head_path = os.path.join(path, "classification_head.pt")
            if os.path.exists(head_path):
                model.classification_head.load_state_dict(torch.load(head_path))

        logger.info(f"Loaded SPEL model from {path}")

        return model


def create_spel_model(
    model_name: str = "roberta-base",
    fixed_candidate_set_size: int = 500000,
    entity_types: Optional[List[str]] = None,
    **kwargs,
) -> SPELModel:
    """
    Convenience function to create a SPEL model.

    Args:
        model_name: Pre-trained model name
        fixed_candidate_set_size: Size of fixed candidate set
        entity_types: List of entity types (for compatibility)
        **kwargs: Additional config parameters

    Returns:
        SPEL model
    """
    config = SPELConfig(
        model_name=model_name,
        fixed_candidate_set_size=fixed_candidate_set_size,
        entity_types=entity_types or ["PER", "ORG", "LOC", "MISC"],
        **kwargs,
    )

    return SPELModel(config)
