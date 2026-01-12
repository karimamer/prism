"""
ATG Model Adapter.

This adapter wraps the ImprovedATGModel to conform to the BaseModelAdapter interface.
"""

from typing import Any, Optional

import torch

from entity_resolution.models.atg import ATGConfig, ImprovedATGModel
from entity_resolution.models.base_adapter import (
    BaseModelAdapter,
    ModelMetadata,
    ModelPrediction,
    register_adapter,
)
from entity_resolution.models.output import EntityPrediction


@register_adapter("atg")
class ATGAdapter(BaseModelAdapter):
    """Adapter for Improved ATG (Autoregressive Text-to-Graph) model."""

    def __init__(self, config: ATGConfig, device: Optional[torch.device] = None):
        """
        Initialize ATG adapter.

        Args:
            config: ATG configuration
            device: Torch device
        """
        super().__init__(config, device)

        # Create the underlying ATG model
        self.model = ImprovedATGModel(config)
        self.model.to(self.device)

    def get_metadata(self) -> ModelMetadata:
        """Get ATG model metadata."""
        return ModelMetadata(
            name="ATG",
            version="2.0",
            model_type="joint",
            description="Autoregressive Text-to-Graph generation for entities and relations",
            capabilities=["entity_linking", "relation_extraction", "graph_generation"],
            required_inputs=["text"],
            optional_inputs=["candidates", "entity_types", "relation_types"],
            metadata={
                "encoder_model": self.config.encoder_model,
                "decoder_layers": self.config.decoder_layers,
                "max_span_length": self.config.max_span_length,
            },
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass through ATG model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        return self.model(input_ids, attention_mask, **kwargs)

    def predict(
        self,
        text: str,
        candidates: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelPrediction:
        """
        Make predictions on input text using ATG.

        Args:
            text: Input text
            candidates: Optional candidate entities
            **kwargs: Additional parameters

        Returns:
            Normalized predictions
        """
        self.validate_input(text)

        # Tokenize input
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_model)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        # Generate predictions
        generated_sequences = self.model.generate(
            input_ids,
            attention_mask,
            max_length=kwargs.get("max_length", 100),
        )

        # Parse generated sequences into entities and relations
        # This is a simplified version - real implementation would decode properly
        entities = self._parse_entities_from_generation(generated_sequences, text)
        relations = self._parse_relations_from_generation(generated_sequences, text)

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(entities, relations)

        return ModelPrediction(
            entities=entities,
            relations=relations,
            confidence=confidence,
            metadata={
                "model_name": "ATG",
                "generated_sequence_length": len(generated_sequences[0])
                if generated_sequences
                else 0,
            },
        )

    def predict_batch(
        self,
        texts: list[str],
        candidates: Optional[list[list[dict[str, Any]]]] = None,
        **kwargs,
    ) -> list[ModelPrediction]:
        """
        Make batch predictions.

        Args:
            texts: List of input texts
            candidates: Optional list of candidate lists
            **kwargs: Additional parameters

        Returns:
            List of predictions
        """
        predictions = []

        for i, text in enumerate(texts):
            text_candidates = candidates[i] if candidates and i < len(candidates) else None
            prediction = self.predict(text, text_candidates, **kwargs)
            predictions.append(prediction)

        return predictions

    def _parse_entities_from_generation(
        self, generated_sequences: list[list[int]], text: str
    ) -> list[EntityPrediction]:
        """Parse entities from generated sequences."""
        # Simplified implementation - real version would decode properly
        entities = []

        # Placeholder: in real implementation, decode sequences to extract entities
        # For now, return empty list
        return entities

    def _parse_relations_from_generation(
        self, generated_sequences: list[list[int]], text: str
    ) -> list:
        """Parse relations from generated sequences."""
        # Simplified implementation
        return []

    def _calculate_overall_confidence(self, entities: list, relations: list) -> float:
        """Calculate overall prediction confidence."""
        if not entities and not relations:
            return 0.0

        total_confidence = 0.0
        count = 0

        for entity in entities:
            total_confidence += entity.confidence
            count += 1

        for relation in relations:
            total_confidence += relation.confidence
            count += 1

        return total_confidence / count if count > 0 else 0.0


__all__ = ["ATGAdapter"]
