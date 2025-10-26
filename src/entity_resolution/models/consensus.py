from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class ExtractionMethod(Enum):
    """Different extraction methods for consensus resolution."""

    RELIK = "relik"
    ATG = "atg"
    CONSENSUS = "consensus"


class EntityPrediction(BaseModel):
    """Data class for entity predictions from different linkers."""

    entity_id: str
    entity_name: str
    confidence: float
    linker_type: str  # "contextual", "prior", "relik", "atg"
    reasoning: Optional[str] = None
    mention_span: Optional[Tuple[int, int]] = None
    extraction_method: Optional[ExtractionMethod] = None


class MethodPrediction(BaseModel):
    """Predictions from a specific extraction method."""

    method: ExtractionMethod
    entities: List[EntityPrediction]
    relations: List[Tuple[EntityPrediction, EntityPrediction, str]]
    confidence: float
    reasoning: str


class DualPerspectiveEntityLinker(nn.Module):
    """
    Dual-perspective Entity Linker (DEL) with contextual and prior components.

    This implements the core concept from OneNet where two different linkers
    provide predictions from different perspectives:
    - Contextual: Uses context and CoT reasoning
    - Prior: Uses inherent prior knowledge without context
    """

    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size

        # Contextual entity linker (uses context + CoT)
        self.contextual_linker = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # mention + context + entity
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Prior entity linker (uses only mention + entity, no context)
        self.prior_linker = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # mention + entity only
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward_contextual(
        self,
        mention_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contextual linker forward pass using context and CoT reasoning.

        Args:
            mention_embedding: Embedding of the mention
            context_embedding: Embedding of the surrounding context
            entity_embeddings: Embeddings of candidate entities

        Returns:
            Contextual linking scores for each entity
        """
        batch_size, num_entities, hidden_size = entity_embeddings.shape

        # Expand mention and context embeddings to match entities
        mention_exp = mention_embedding.unsqueeze(1).expand(-1, num_entities, -1)
        context_exp = context_embedding.unsqueeze(1).expand(-1, num_entities, -1)

        # Concatenate mention, context, and entity embeddings
        combined = torch.cat([mention_exp, context_exp, entity_embeddings], dim=-1)

        # Reshape for linear layers
        combined = combined.view(-1, hidden_size * 3)

        # Get contextual scores
        contextual_scores = self.contextual_linker(combined)

        # Reshape back
        contextual_scores = contextual_scores.view(batch_size, num_entities)

        return contextual_scores

    def forward_prior(
        self, mention_embedding: torch.Tensor, entity_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Prior linker forward pass using only inherent knowledge.

        Args:
            mention_embedding: Embedding of the mention
            entity_embeddings: Embeddings of candidate entities

        Returns:
            Prior linking scores for each entity
        """
        batch_size, num_entities, hidden_size = entity_embeddings.shape

        # Expand mention embedding to match entities
        mention_exp = mention_embedding.unsqueeze(1).expand(-1, num_entities, -1)

        # Concatenate mention and entity embeddings (no context)
        combined = torch.cat([mention_exp, entity_embeddings], dim=-1)

        # Reshape for linear layers
        combined = combined.view(-1, hidden_size * 2)

        # Get prior scores
        prior_scores = self.prior_linker(combined)

        # Reshape back
        prior_scores = prior_scores.view(batch_size, num_entities)

        return prior_scores

    def predict_dual(
        self,
        mention_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        entity_embeddings: torch.Tensor,
        entity_names: List[str],
    ) -> Tuple[EntityPrediction, EntityPrediction]:
        """
        Generate predictions from both contextual and prior linkers.

        Args:
            mention_embedding: Embedding of the mention
            context_embedding: Embedding of the surrounding context
            entity_embeddings: Embeddings of candidate entities
            entity_names: Names of candidate entities

        Returns:
            Tuple of (contextual_prediction, prior_prediction)
        """
        # Get contextual scores
        contextual_scores = self.forward_contextual(
            mention_embedding, context_embedding, entity_embeddings
        )

        # Get prior scores
        prior_scores = self.forward_prior(mention_embedding, entity_embeddings)

        # Get best predictions from each linker
        contextual_idx = torch.argmax(contextual_scores, dim=1)[0].item()
        prior_idx = torch.argmax(prior_scores, dim=1)[0].item()

        contextual_pred = EntityPrediction(
            entity_id=f"entity_{contextual_idx}",
            entity_name=entity_names[contextual_idx]
            if contextual_idx < len(entity_names)
            else f"entity_{contextual_idx}",
            confidence=contextual_scores[0, contextual_idx].item(),
            linker_type="contextual",
            reasoning="Context-aware inference with CoT reasoning",
        )

        prior_pred = EntityPrediction(
            entity_id=f"entity_{prior_idx}",
            entity_name=entity_names[prior_idx]
            if prior_idx < len(entity_names)
            else f"entity_{prior_idx}",
            confidence=prior_scores[0, prior_idx].item(),
            linker_type="prior",
            reasoning="Prior knowledge without contextual influence",
        )

        return contextual_pred, prior_pred


class EntityConsensusJudger(nn.Module):
    """
    Entity Consensus Judger (ECJ) implementing OneNet's consistency algorithm.

    This module resolves conflicts between contextual and prior linker predictions
    using a sophisticated consensus mechanism with auxiliary LLM support.
    """

    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size

        # Auxiliary decision network (simulates auxiliary LLM)
        self.auxiliary_judge = nn.Sequential(
            nn.Linear(
                hidden_size * 4 + 2, hidden_size
            ),  # 2 entities + 2 confidences + mention + context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),  # Binary choice: 0=contextual, 1=prior
            nn.Softmax(dim=-1),
        )

        # Confidence reconciliation network
        self.confidence_reconciler = nn.Sequential(
            nn.Linear(4, 16),  # 2 confidences + 2 additional features
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def detect_concurrence(
        self, contextual_pred: EntityPrediction, prior_pred: EntityPrediction
    ) -> bool:
        """
        Detect if contextual and prior predictions are in concurrence.

        Args:
            contextual_pred: Prediction from contextual linker
            prior_pred: Prediction from prior linker

        Returns:
            True if predictions agree, False if there's discordance
        """
        return contextual_pred.entity_id == prior_pred.entity_id

    def resolve_discordance(
        self,
        contextual_pred: EntityPrediction,
        prior_pred: EntityPrediction,
        mention_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> EntityPrediction:
        """
        Resolve discordance between predictions using auxiliary judge.

        Args:
            contextual_pred: Prediction from contextual linker
            prior_pred: Prediction from prior linker
            mention_embedding: Embedding of the mention
            context_embedding: Embedding of the context
            entity_embeddings: Embeddings of candidate entities

        Returns:
            Final resolved prediction
        """
        # Extract entity embeddings for the two predictions
        contextual_idx = int(contextual_pred.entity_id.split("_")[1])
        prior_idx = int(prior_pred.entity_id.split("_")[1])

        contextual_entity_emb = entity_embeddings[0, contextual_idx, :]
        prior_entity_emb = entity_embeddings[0, prior_idx, :]

        # Prepare input for auxiliary judge
        aux_input = torch.cat(
            [
                mention_embedding.squeeze(0),
                context_embedding.squeeze(0),
                contextual_entity_emb,
                prior_entity_emb,
                torch.tensor([contextual_pred.confidence], device=mention_embedding.device),
                torch.tensor([prior_pred.confidence], device=mention_embedding.device),
            ]
        )

        # Get auxiliary judgment
        aux_decision = self.auxiliary_judge(aux_input.unsqueeze(0))

        # Choose prediction based on auxiliary decision
        if aux_decision[0, 0].item() > aux_decision[0, 1].item():
            # Choose contextual prediction
            chosen_pred = contextual_pred
            reasoning = f"Auxiliary judge favored contextual prediction (score: {aux_decision[0, 0].item():.3f})"
        else:
            # Choose prior prediction
            chosen_pred = prior_pred
            reasoning = (
                f"Auxiliary judge favored prior prediction (score: {aux_decision[0, 1].item():.3f})"
            )

        # Reconcile confidence using both predictions
        confidence_features = torch.tensor(
            [
                contextual_pred.confidence,
                prior_pred.confidence,
                aux_decision[0, 0].item(),  # Auxiliary confidence for contextual
                aux_decision[0, 1].item(),  # Auxiliary confidence for prior
            ],
            device=mention_embedding.device,
        )

        reconciled_confidence = self.confidence_reconciler(confidence_features.unsqueeze(0))[
            0, 0
        ].item()

        # Create final prediction with reconciled confidence
        final_pred = EntityPrediction(
            entity_id=chosen_pred.entity_id,
            entity_name=chosen_pred.entity_name,
            confidence=reconciled_confidence,
            linker_type="consensus",
            reasoning=reasoning,
        )

        return final_pred

    def judge_consensus(
        self,
        contextual_pred: EntityPrediction,
        prior_pred: EntityPrediction,
        mention_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> EntityPrediction:
        """
        Main consensus judging function implementing OneNet's consistency algorithm.

        Args:
            contextual_pred: Prediction from contextual linker
            prior_pred: Prediction from prior linker
            mention_embedding: Embedding of the mention
            context_embedding: Embedding of the context
            entity_embeddings: Embeddings of candidate entities

        Returns:
            Final consensus prediction
        """
        # Check for concurrence
        if self.detect_concurrence(contextual_pred, prior_pred):
            # Concurrence: both linkers agree
            # Combine confidences for stronger prediction
            combined_confidence = (contextual_pred.confidence + prior_pred.confidence) / 2

            final_pred = EntityPrediction(
                entity_id=contextual_pred.entity_id,
                entity_name=contextual_pred.entity_name,
                confidence=combined_confidence,
                linker_type="consensus",
                reasoning="Contextual and prior linkers in full agreement",
            )
        else:
            # Discordance: resolve using auxiliary judge
            final_pred = self.resolve_discordance(
                contextual_pred, prior_pred, mention_embedding, context_embedding, entity_embeddings
            )

        return final_pred


class ConsensusModule(nn.Module):
    """
    OneNet-inspired consensus module for resolving entity conflicts.

    This module implements the full OneNet framework with:
    - Dual-perspective Entity Linker (DEL)
    - Entity Consensus Judger (ECJ)
    - Sophisticated conflict resolution
    """

    def __init__(self, hidden_size=768, threshold=0.6):
        super().__init__()

        self.hidden_size = hidden_size
        self.threshold = threshold

        # Core OneNet components
        self.dual_linker = DualPerspectiveEntityLinker(hidden_size)
        self.consensus_judger = EntityConsensusJudger(hidden_size)

        # Legacy components for backward compatibility
        self.confidence_calibration = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1), nn.Sigmoid()
        )

        # Span-level conflict resolution
        self.span_conflict_resolver = nn.Sequential(
            nn.Linear(hidden_size * 2 + 4, hidden_size),  # 2 entities + 4 features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
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
                entity_scores.get("context_score", 0.5),
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
        # Filter entities that have mention_span and sort by start position
        entities_with_spans = []
        for entity in entities:
            if "mention_span" in entity and entity["mention_span"] is not None:
                entities_with_spans.append(entity)
            elif "start" in entity and "end" in entity:
                # Convert start/end fields to mention_span
                entity["mention_span"] = [entity["start"], entity["end"]]
                entities_with_spans.append(entity)
            else:
                # Skip entities without span information
                logger.warning(f"Entity missing span information: {entity}")
                continue

        if not entities_with_spans:
            return []

        sorted_entities = sorted(entities_with_spans, key=lambda e: e["mention_span"][0])

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
        conf_scores = torch.tensor(
            [entity.get("confidence", 0.5) for entity in entities],
            device=next(self.parameters()).device,
        )

        # Combine embeddings and confidence scores
        combined = torch.cat([embs, conf_scores.unsqueeze(1).expand(-1, self.hidden_size)], dim=1)

        # Compute resolution scores
        scores = self.conflict_resolver(combined)

        # Get best entity
        best_idx = torch.argmax(scores).item()

        return entities[best_idx]

    def resolve_entities_onenet(
        self,
        mentions: List[Dict],
        mention_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        entity_names: List[str],
    ) -> List[EntityPrediction]:
        """
        Resolve entities using OneNet's dual-perspective approach.

        Args:
            mentions: List of mention dictionaries
            mention_embeddings: Embeddings for mentions
            context_embeddings: Context embeddings for each mention
            entity_embeddings: Candidate entity embeddings
            entity_names: Names of candidate entities

        Returns:
            List of resolved entity predictions
        """
        resolved_entities = []

        for i, mention in enumerate(mentions):
            # Get embeddings for this mention
            mention_emb = mention_embeddings[i : i + 1]  # Keep batch dimension
            context_emb = context_embeddings[i : i + 1]  # Keep batch dimension

            # Get dual predictions
            contextual_pred, prior_pred = self.dual_linker.predict_dual(
                mention_emb, context_emb, entity_embeddings, entity_names
            )

            # Use consensus judger to resolve
            final_pred = self.consensus_judger.judge_consensus(
                contextual_pred, prior_pred, mention_emb, context_emb, entity_embeddings
            )

            # Filter by confidence threshold
            if final_pred.confidence >= self.threshold:
                resolved_entities.append(final_pred)

        return resolved_entities

    def resolve_entities(self, entities, context=None):
        """
        Legacy resolve entities method for backward compatibility.

        Args:
            entities: List of entity dictionaries
            context: Optional context text

        Returns:
            List of resolved and filtered entity dictionaries
        """
        # Calibrate confidence for each entity
        for entity in entities:
            if "confidence" not in entity:
                scores = {
                    "mention_score": entity.get("mention_score", 0.5),
                    "entity_score": entity.get("entity_score", 0.5),
                    "context_score": entity.get("context_score", 0.5),
                }
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

    def forward(
        self,
        entities=None,
        context=None,
        mentions=None,
        mention_embeddings=None,
        context_embeddings=None,
        entity_embeddings=None,
        entity_names=None,
        use_onenet=True,
    ):
        """
        Forward pass for consensus module.

        Args:
            entities: Legacy entity dictionaries (for backward compatibility)
            context: Optional context text
            mentions: List of mention dictionaries (for OneNet)
            mention_embeddings: Mention embeddings (for OneNet)
            context_embeddings: Context embeddings (for OneNet)
            entity_embeddings: Entity embeddings (for OneNet)
            entity_names: Entity names (for OneNet)
            use_onenet: Whether to use OneNet approach

        Returns:
            List of resolved entity predictions or dictionaries
        """
        if use_onenet and mentions is not None and mention_embeddings is not None:
            # Use OneNet approach
            return self.resolve_entities_onenet(
                mentions, mention_embeddings, context_embeddings, entity_embeddings, entity_names
            )
        else:
            # Use legacy approach
            return self.resolve_entities(entities, context)

    def resolve_multi_method(
        self,
        relik_prediction: Optional[MethodPrediction] = None,
        atg_prediction: Optional[MethodPrediction] = None,
        context_embedding: Optional[torch.Tensor] = None,
    ) -> MethodPrediction:
        """
        Resolve predictions from multiple extraction methods.

        Args:
            relik_prediction: Prediction from ReLiK method
            atg_prediction: Prediction from ATG method
            context_embedding: Context representation

        Returns:
            Resolved consensus prediction
        """
        if relik_prediction is None and atg_prediction is None:
            # No predictions to resolve
            return MethodPrediction(
                method=ExtractionMethod.CONSENSUS,
                entities=[],
                relations=[],
                confidence=0.0,
                reasoning="No predictions provided",
            )

        elif relik_prediction is None:
            # Only ATG prediction
            return atg_prediction

        elif atg_prediction is None:
            # Only ReLiK prediction
            return relik_prediction

        else:
            # Both methods have predictions - use simple resolution for now
            if context_embedding is None:
                # Create dummy context embedding
                context_embedding = torch.zeros(1, self.hidden_size)

            # Simple heuristic: choose method with higher confidence
            if relik_prediction.confidence > atg_prediction.confidence:
                chosen_pred = relik_prediction
                reasoning = f"ReLiK chosen (confidence: {relik_prediction.confidence:.3f} vs {atg_prediction.confidence:.3f})"
            else:
                chosen_pred = atg_prediction
                reasoning = f"ATG chosen (confidence: {atg_prediction.confidence:.3f} vs {relik_prediction.confidence:.3f})"

            # Create consensus prediction
            consensus_pred = MethodPrediction(
                method=ExtractionMethod.CONSENSUS,
                entities=chosen_pred.entities,
                relations=chosen_pred.relations,
                confidence=(relik_prediction.confidence + atg_prediction.confidence) / 2,
                reasoning=f"Simple consensus: {reasoning}",
            )

            return consensus_pred

    def compute_onenet_losses(
        self,
        predictions: List[EntityPrediction],
        gold_entities: List[Dict],
        contextual_scores: torch.Tensor,
        prior_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for OneNet training.

        Args:
            predictions: Final consensus predictions
            gold_entities: Gold standard entities
            contextual_scores: Raw contextual linker scores
            prior_scores: Raw prior linker scores

        Returns:
            Dictionary of computed losses
        """
        losses = {}

        # Consensus loss (how well final predictions match gold)
        consensus_loss = 0.0
        contextual_loss = 0.0
        prior_loss = 0.0

        if predictions and gold_entities:
            # Convert predictions to format for loss computation
            pred_entities = []
            for pred in predictions:
                pred_entities.append(
                    {
                        "mention": "",  # Would need mention text
                        "entity_id": pred.entity_id,
                        "confidence": pred.confidence,
                    }
                )

            consensus_loss = self.loss(pred_entities, gold_entities)

            # Individual linker losses (would need gold labels for each linker)
            if len(contextual_scores.shape) > 0:
                # Placeholder - would need proper gold labels
                contextual_targets = torch.zeros_like(contextual_scores)
                contextual_loss = F.binary_cross_entropy(contextual_scores, contextual_targets)

            if len(prior_scores.shape) > 0:
                # Placeholder - would need proper gold labels
                prior_targets = torch.zeros_like(prior_scores)
                prior_loss = F.binary_cross_entropy(prior_scores, prior_targets)

        losses.update(
            {
                "consensus_loss": consensus_loss
                if isinstance(consensus_loss, torch.Tensor)
                else torch.tensor(consensus_loss),
                "contextual_loss": contextual_loss
                if isinstance(contextual_loss, torch.Tensor)
                else torch.tensor(contextual_loss),
                "prior_loss": prior_loss
                if isinstance(prior_loss, torch.Tensor)
                else torch.tensor(prior_loss),
                "total_loss": (consensus_loss + contextual_loss + prior_loss)
                if all(
                    isinstance(x, torch.Tensor)
                    for x in [consensus_loss, contextual_loss, prior_loss]
                )
                else torch.tensor(0.0),
            }
        )

        return losses

    def create_method_prediction_from_relik(self, relik_results: Dict) -> MethodPrediction:
        """
        Convert ReLiK results to MethodPrediction format.

        Args:
            relik_results: Results from ReLiK model

        Returns:
            MethodPrediction object
        """
        entities = []

        # Convert ReLiK entities
        if "mention_spans" in relik_results and "best_entities" in relik_results:
            for i, (span, entity_idx) in enumerate(
                zip(relik_results["mention_spans"], relik_results["best_entities"])
            ):
                confidence = 0.5  # Default confidence
                if "entity_linking_probs" in relik_results and i < len(
                    relik_results["entity_linking_probs"]
                ):
                    confidence = relik_results["entity_linking_probs"][i].max().item()

                entity = EntityPrediction(
                    entity_id=f"relik_entity_{i}",
                    entity_name=f"entity_{entity_idx}",
                    confidence=confidence,
                    linker_type="relik",
                    reasoning="ReLiK span-based prediction",
                    mention_span=span,
                    extraction_method=ExtractionMethod.RELIK,
                )
                entities.append(entity)

        # Convert ReLiK relations
        relations = []
        if "relation_probs" in relik_results:
            # Simplified relation conversion
            pass  # Would need more detailed relation parsing

        # Compute overall confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0.0

        return MethodPrediction(
            method=ExtractionMethod.RELIK,
            entities=entities,
            relations=relations,
            confidence=avg_confidence,
            reasoning="ReLiK span-based entity linking with start/end token prediction",
        )

    def create_method_prediction_from_atg(self, atg_results) -> MethodPrediction:
        """
        Convert ATG results to MethodPrediction format.

        Args:
            atg_results: Results from ATG model (ATGOutput)

        Returns:
            MethodPrediction object
        """
        entities = []

        # Convert ATG entities
        for i, span_repr in enumerate(atg_results.entities):
            entity = EntityPrediction(
                entity_id=f"atg_entity_{i}",
                entity_name=span_repr.entity_type,
                confidence=0.8,  # ATG doesn't provide explicit confidence
                linker_type="atg",
                reasoning="ATG autoregressive text-to-graph generation",
                mention_span=(span_repr.start, span_repr.end),
                extraction_method=ExtractionMethod.ATG,
            )
            entities.append(entity)

        # Convert ATG relations
        relations = []
        for head, tail, rel_type in atg_results.relations:
            # Convert to EntityPrediction format
            head_entity = EntityPrediction(
                entity_id=f"head_{head.start}_{head.end}",
                entity_name=head.entity_type,
                confidence=0.8,
                linker_type="atg",
                mention_span=(head.start, head.end),
                extraction_method=ExtractionMethod.ATG,
            )
            tail_entity = EntityPrediction(
                entity_id=f"tail_{tail.start}_{tail.end}",
                entity_name=tail.entity_type,
                confidence=0.8,
                linker_type="atg",
                mention_span=(tail.start, tail.end),
                extraction_method=ExtractionMethod.ATG,
            )
            relations.append((head_entity, tail_entity, rel_type))

        # Compute overall confidence
        avg_confidence = (
            sum(s for s in atg_results.generation_scores) / len(atg_results.generation_scores)
            if atg_results.generation_scores
            else 0.8
        )

        return MethodPrediction(
            method=ExtractionMethod.ATG,
            entities=entities,
            relations=relations,
            confidence=avg_confidence,
            reasoning="ATG autoregressive generation with dynamic vocabulary and constrained decoding",
        )
