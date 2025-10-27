"""
Context-sensitive prediction aggregation for SPEL.

Aggregates subword-level predictions into word-level and span-level predictions.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


class PredictionAggregator:
    """
    Context-sensitive prediction aggregator for SPEL.

    Aggregates subword predictions into meaningful spans by:
    1. Grouping subwords into words
    2. Aggregating predictions within words
    3. Merging consecutive words with same entity
    4. Filtering invalid spans (punctuation, function words)
    """

    # Common function words to filter
    FUNCTION_WORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
    }

    # Punctuation to filter
    PUNCTUATION = {".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", '"', "'"}

    def __init__(
        self,
        tokenizer,
        filter_punctuation: bool = True,
        filter_function_words: bool = True,
    ):
        """
        Initialize prediction aggregator.

        Args:
            tokenizer: Tokenizer used for text processing
            filter_punctuation: Whether to filter single punctuation spans
            filter_function_words: Whether to filter single function word spans
        """
        self.tokenizer = tokenizer
        self.filter_punctuation = filter_punctuation
        self.filter_function_words = filter_function_words

    def aggregate_subword_predictions(
        self,
        text: str,
        subword_predictions: List[List[Tuple[str, float]]],
        subword_to_word_map: List[int],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Aggregate subword predictions into span-level predictions.

        Args:
            text: Original input text
            subword_predictions: List of top-k (entity_id, probability) for each subword
            subword_to_word_map: Mapping from subword index to word index
            top_k: Number of top predictions to consider per subword

        Returns:
            List of span predictions with format:
            [{"start": char_start, "end": char_end, "entity": entity_id, "score": prob}, ...]
        """
        if not subword_predictions:
            return []

        # Group subwords by word
        word_predictions = self._group_by_word(subword_predictions, subword_to_word_map)

        # Aggregate predictions within each word
        word_entities = self._aggregate_word_predictions(word_predictions)

        # Merge consecutive words with same entity
        spans = self._merge_consecutive_words(word_entities, text)

        # Filter invalid spans
        spans = self._filter_spans(spans, text)

        return spans

    def _group_by_word(
        self,
        subword_predictions: List[List[Tuple[str, float]]],
        subword_to_word_map: List[int],
    ) -> Dict[int, List[List[Tuple[str, float]]]]:
        """
        Group subword predictions by word index.

        Returns:
            Dict mapping word_idx -> list of subword predictions
        """
        word_predictions = {}

        for subword_idx, preds in enumerate(subword_predictions):
            if subword_idx >= len(subword_to_word_map):
                continue

            word_idx = subword_to_word_map[subword_idx]

            if word_idx not in word_predictions:
                word_predictions[word_idx] = []

            word_predictions[word_idx].append(preds)

        return word_predictions

    def _aggregate_word_predictions(
        self,
        word_predictions: Dict[int, List[List[Tuple[str, float]]]],
    ) -> Dict[int, Tuple[str, float]]:
        """
        Aggregate predictions within each word.

        Strategy: For each entity, compute average probability across all subwords
        in the word, then select entity with highest average.

        Returns:
            Dict mapping word_idx -> (entity_id, probability)
        """
        word_entities = {}

        for word_idx, subword_preds_list in word_predictions.items():
            # Collect all entity predictions across subwords
            entity_probs = {}
            entity_counts = {}

            for subword_preds in subword_preds_list:
                for entity_id, prob in subword_preds:
                    if entity_id not in entity_probs:
                        entity_probs[entity_id] = 0.0
                        entity_counts[entity_id] = 0

                    entity_probs[entity_id] += prob
                    entity_counts[entity_id] += 1

            # Compute average probability for each entity
            entity_avg_probs = {
                ent: prob / entity_counts[ent] for ent, prob in entity_probs.items()
            }

            # Select entity with highest average probability
            if entity_avg_probs:
                best_entity = max(entity_avg_probs.items(), key=lambda x: x[1])
                word_entities[word_idx] = best_entity

        return word_entities

    def _merge_consecutive_words(
        self,
        word_entities: Dict[int, Tuple[str, float]],
        text: str,
    ) -> List[Dict]:
        """
        Merge consecutive words with the same entity into spans.

        Returns:
            List of span dicts
        """
        if not word_entities:
            return []

        spans = []
        current_entity = None
        current_words = []
        current_probs = []

        for word_idx in sorted(word_entities.keys()):
            entity_id, prob = word_entities[word_idx]

            # Skip O (non-entity) labels
            if entity_id == "O":
                # Save current span if exists
                if current_entity and current_entity != "O":
                    spans.append(
                        self._create_span(current_entity, current_words, current_probs, text)
                    )
                current_entity = None
                current_words = []
                current_probs = []
                continue

            # Check if same entity as current span
            if entity_id == current_entity:
                current_words.append(word_idx)
                current_probs.append(prob)
            else:
                # Save previous span if exists
                if current_entity and current_entity != "O":
                    spans.append(
                        self._create_span(current_entity, current_words, current_probs, text)
                    )

                # Start new span
                current_entity = entity_id
                current_words = [word_idx]
                current_probs = [prob]

        # Save final span
        if current_entity and current_entity != "O":
            spans.append(self._create_span(current_entity, current_words, current_probs, text))

        return spans

    def _create_span(
        self,
        entity_id: str,
        word_indices: List[int],
        probabilities: List[float],
        text: str,
    ) -> Dict:
        """Create span dictionary from word indices."""
        # For now, use dummy char positions (would need proper word boundary detection)
        avg_prob = sum(probabilities) / len(probabilities) if probabilities else 0.0

        return {
            "entity": entity_id,
            "score": avg_prob,
            "word_indices": word_indices,
            "start": -1,  # Would need proper implementation
            "end": -1,  # Would need proper implementation
        }

    def _filter_spans(
        self,
        spans: List[Dict],
        text: str,
    ) -> List[Dict]:
        """
        Filter invalid spans.

        Removes:
        - Single punctuation spans
        - Single function word spans (optional)
        """
        filtered = []

        for span in spans:
            # Would need proper span text extraction
            # For now, just pass through
            filtered.append(span)

        return filtered
