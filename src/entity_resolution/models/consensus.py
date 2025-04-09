import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class ConsensusModule(nn.Module):
    """
    OneNet-inspired consensus module for resolving entity conflicts.

    This module combines multiple entity resolution strategies and resolves
    conflicts between overlapping entity mentions.
    """
    def __init__(self, hidden_size=768, threshold=0.6):
        super().__init__()

        self.hidden_size = hidden_size
        self.threshold = threshold

        # Confidence calibration module
        self.confidence_calibration = nn.Sequential(
            nn.Linear(3, 16),  # Combine scores from multiple methods
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Conflict resolution module
        self.conflict_resolver = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def calibrate_confidence(self, entity_scores):
        """
        Calibrate confidence scores for entity predictions.

        Args:
            entity_scores: Dictionary or tensor with raw scores from different methods

        Returns:
            Calibrated confidence score
        """
        if isinstance(entity_scores, dict):
            # Extract scores from dictionary
            scores = [
                entity_scores.get("mention_score", 0.5),
                entity_scores.get("entity_score", 0.5),
                entity_scores.get("context_score", 0.5)
            ]
            scores_tensor = torch.tensor(scores, device=next(self.parameters()).device).float()
        else:
            # Use tensor directly
            scores_tensor = entity_scores

        # Calibrate confidence
        calibrated_score = self.confidence_calibration(scores_tensor)

        return calibrated_score

    def resolve_conflicts(self, entities, entity_embeddings=None):
        """
        Resolve conflicts between overlapping entity mentions.

        Args:
            entities: List of entity dictionaries
            entity_embeddings: Optional embeddings for each entity

        Returns:
            List of resolved entity dictionaries
        """
        if not entities:
            return []

        # Group entities by overlapping spans
        span_groups = self._group_overlapping_spans(entities)

        # Resolve each group
        resolved_entities = []

        for group in span_groups:
            if len(group) == 1:
                # No conflict
                resolved_entities.append(group[0])
            else:
                # Resolve conflict
                if entity_embeddings is not None:
                    # Use entity embeddings for resolution
                    best_entity = self._resolve_with_embeddings(group, entity_embeddings)
                else:
                    # Use confidence scores for resolution
                    best_entity = max(group, key=lambda e: e.get("confidence", 0))

                resolved_entities.append(best_entity)

        return resolved_entities

    def _group_overlapping_spans(self, entities):
        """
        Group entities with overlapping spans.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of entity groups
        """
        # Sort entities by span start position
        sorted_entities = sorted(entities, key=lambda e: e["mention_span"][0])

        # Group overlapping spans
        groups = []
        current_group = []

        for entity in sorted_entities:
            if not current_group:
                # First entity in group
                current_group = [entity]
            else:
                # Check if entity overlaps with current group
                curr_end = max(e["mention_span"][1] for e in current_group)
                if entity["mention_span"][0] <= curr_end:
                    # Overlap
                    current_group.append(entity)
                else:
                    # No overlap, start new group
                    groups.append(current_group)
                    current_group = [entity]

        # Add last group
        if current_group:
            groups.append(current_group)

        return groups

    def _resolve_with_embeddings(self, entities, entity_embeddings):
        """
        Resolve conflicts using entity embeddings.

        Args:
            entities: List of conflicting entity dictionaries
            entity_embeddings: Embeddings for each entity

        Returns:
            Best entity from the conflicting group
        """
        # Get embeddings for each entity
        embs = []
        for entity in entities:
            entity_id = entity["entity_id"]
            if entity_id in entity_embeddings:
                embs.append(entity_embeddings[entity_id])
            else:
                # Use zero embedding if not found
                embs.append(torch.zeros(self.hidden_size, device=next(self.parameters()).device))

        # Stack embeddings
        embs = torch.stack(embs)

        # Get confidence scores
        conf_scores = torch.tensor([entity.get("confidence", 0.5) for entity in entities],
                                  device=next(self.parameters()).device)

        # Combine embeddings and confidence scores
        combined = torch.cat([embs, conf_scores.unsqueeze(1).expand(-1, self.hidden_size)], dim=1)

        # Compute resolution scores
        scores = self.conflict_resolver(combined)

        # Get best entity
        best_idx = torch.argmax(scores).item()

        return entities[best_idx]

    def resolve_entities(self, entities, context=None):
        """
        Resolve entities and filter by confidence threshold.

        Args:
            entities: List of entity dictionaries
            context: Optional context text

        Returns:
            List of resolved and filtered entity dictionaries
        """
        # Calibrate confidence for each entity
        for entity in entities:
            if "confidence" not in entity:
                # Create scores dictionary
                scores = {
                    "mention_score": entity.get("mention_score", 0.5),
                    "entity_score": entity.get("entity_score", 0.5),
                    "context_score": entity.get("context_score", 0.5)
                }

                # Calibrate confidence
                entity["confidence"] = self.calibrate_confidence(scores).item()

        # Resolve conflicts
        resolved = self.resolve_conflicts(entities)

        # Filter by confidence threshold
        filtered = [e for e in resolved if e.get("confidence", 0) >= self.threshold]

        return filtered

    def loss(self, predicted_entities, gold_entities):
        """
        Compute loss for consensus module training.

        Args:
            predicted_entities: Predicted entity dictionaries
            gold_entities: Gold entity dictionaries

        Returns:
            Loss value
        """
        # Extract confidence scores
        pred_scores = []
        gold_labels = []

        for pred in predicted_entities:
            pred_id = (pred["mention"], pred["entity_id"])
            score = pred.get("confidence", 0.5)
            pred_scores.append(score)

            # Check if prediction matches any gold entity
            label = 0
            for gold in gold_entities:
                gold_id = (gold["mention"], gold["entity_id"])
                if pred_id == gold_id:
                    label = 1
                    break

            gold_labels.append(label)

        # Convert to tensors
        if pred_scores:
            pred_scores = torch.tensor(pred_scores, device=next(self.parameters()).device)
            gold_labels = torch.tensor(gold_labels, device=next(self.parameters()).device)

            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(pred_scores, gold_labels.float())
        else:
            # No predictions
            loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)

        return loss

    def forward(self, entities, context=None):
        """
        Forward pass for consensus module.

        Args:
            entities: List of entity dictionaries
            context: Optional context text

        Returns:
            List of resolved and filtered entity dictionaries
        """
        return self.resolve_entities(entities, context)
