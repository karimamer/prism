import torch
import torch.nn as nn
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntityOutputFormatter(nn.Module):
    """
    Formats entity resolution outputs into structured data.

    This module handles:
    1. Converting token positions to character positions
    2. Formatting entity data for output
    3. Converting between different output formats
    """
    def __init__(self, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer

    def convert_token_to_char_spans(
        self,
        token_spans: List[Tuple[int, int]],
        offset_mapping: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Convert token-level spans to character-level spans.

        Args:
            token_spans: List of token-level spans (start, end)
            offset_mapping: Mapping from tokens to character positions

        Returns:
            List of character-level spans (start, end)
        """
        char_spans = []

        for token_start, token_end in token_spans:
            # Get character positions from offset mapping
            if offset_mapping is not None and offset_mapping.size(0) > 0:
                char_start = offset_mapping[0, token_start, 0].item()
                char_end = offset_mapping[0, token_end, 1].item()
                char_spans.append((char_start, char_end))
            else:
                # If no offset mapping, keep token positions
                char_spans.append((token_start, token_end))

        return char_spans

    def get_mention_text(
        self,
        input_ids: torch.Tensor,
        token_span: Tuple[int, int]
    ) -> str:
        """
        Get mention text from token span.

        Args:
            input_ids: Token IDs
            token_span: Token span (start, end)

        Returns:
            Mention text
        """
        if self.tokenizer is None:
            return "[Unknown mention]"

        # Extract token IDs for mention
        token_start, token_end = token_span
        mention_ids = input_ids[0, token_start:token_end+1]

        # Decode tokens to text
        mention_text = self.tokenizer.decode(mention_ids, skip_special_tokens=True)

        return mention_text.strip()

    def format_entity(
        self,
        mention_span: Tuple[int, int],
        entity_id: str,
        entity_name: str,
        entity_type: str,
        confidence: float,
        input_ids: Optional[torch.Tensor] = None,
        offset_mapping: Optional[torch.Tensor] = None,
        source: str = "entity_resolution"
    ) -> Dict[str, Any]:
        """
        Format entity information for output.

        Args:
            mention_span: Token span (start, end)
            entity_id: Entity ID
            entity_name: Entity name
            entity_type: Entity type
            confidence: Confidence score
            input_ids: Token IDs (optional)
            offset_mapping: Mapping from tokens to character positions (optional)
            source: Source of entity resolution

        Returns:
            Formatted entity dictionary
        """
        # Get mention text if input_ids is provided
        mention_text = ""
        if input_ids is not None and self.tokenizer is not None:
            mention_text = self.get_mention_text(input_ids, mention_span)

        # Convert to character spans if offset_mapping is provided
        char_span = None
        if offset_mapping is not None:
            char_spans = self.convert_token_to_char_spans([mention_span], offset_mapping)
            if char_spans:
                char_span = char_spans[0]

        # Build entity dictionary
        entity = {
            "mention": mention_text,
            "mention_span": mention_span,
            "entity_id": entity_id,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "confidence": confidence,
            "source": source
        }

        # Add character span if available
        if char_span:
            entity["char_span"] = char_span

        return entity

    def format_entities(
        self,
        linked_entities: List[Dict[str, Any]],
        input_ids: Optional[torch.Tensor] = None,
        offset_mapping: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Format multiple entities for output.

        Args:
            linked_entities: List of linked entities
            input_ids: Token IDs (optional)
            offset_mapping: Mapping from tokens to character positions (optional)

        Returns:
            List of formatted entity dictionaries
        """
        formatted_entities = []

        for entity in linked_entities:
            # Extract entity information
            mention_span = entity.get("mention_span", (0, 0))
            entity_id = entity.get("entity_id", "")
            entity_name = entity.get("entity_name", "")
            entity_type = entity.get("entity_type", "UNKNOWN")
            confidence = entity.get("confidence", 0.0)
            source = entity.get("source", "entity_resolution")

            # Format entity
            formatted_entity = self.format_entity(
                mention_span=mention_span,
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=entity_type,
                confidence=confidence,
                input_ids=input_ids,
                offset_mapping=offset_mapping,
                source=source
            )

            # Add extra fields if present
            for key, value in entity.items():
                if key not in formatted_entity:
                    formatted_entity[key] = value

            formatted_entities.append(formatted_entity)

        return formatted_entities

    def forward(
        self,
        linked_entities: List[Dict[str, Any]],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        offset_mapping: Optional[torch.Tensor] = None,
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for entity output formatter.

        Args:
            linked_entities: List of linked entities
            input_ids: Token IDs (optional)
            attention_mask: Attention mask (optional)
            offset_mapping: Mapping from tokens to character positions (optional)
            text: Original input text (optional)

        Returns:
            Dictionary with formatted entities and metadata
        """
        # Format entities
        entities = self.format_entities(
            linked_entities,
            input_ids,
            offset_mapping
        )

        # Get original text if possible
        original_text = text
        if original_text is None and input_ids is not None and self.tokenizer is not None:
            original_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Create output dictionary
        output = {
            "entities": entities,
            "text": original_text if original_text else "",
            "num_entities": len(entities)
        }

        return output

    def to_json(
        self,
        entities_data: Dict[str, Any],
        indent: int = 2
    ) -> str:
        """
        Convert entity data to JSON string.

        Args:
            entities_data: Entity data dictionary
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(entities_data, indent=indent)

    def to_csv(
        self,
        entities_data: Dict[str, Any]
    ) -> str:
        """
        Convert entity data to CSV format.

        Args:
            entities_data: Entity data dictionary

        Returns:
            CSV string
        """
        # Create CSV header
        csv_lines = ["text,mention,entity_id,entity_name,entity_type,confidence"]

        # Add entities
        text = entities_data.get("text", "")
        text = text.replace('"', '""')  # Escape double quotes

        for entity in entities_data.get("entities", []):
            mention = entity.get("mention", "").replace('"', '""')
            entity_id = entity.get("entity_id", "").replace('"', '""')
            entity_name = entity.get("entity_name", "").replace('"', '""')
            entity_type = entity.get("entity_type", "").replace('"', '""')
            confidence = entity.get("confidence", 0.0)

            line = f'"{text}","{mention}","{entity_id}","{entity_name}","{entity_type}",{confidence}'
            csv_lines.append(line)

        return "\n".join(csv_lines)

    def to_text(
        self,
        entities_data: Dict[str, Any]
    ) -> str:
        """
        Convert entity data to human-readable text format.

        Args:
            entities_data: Entity data dictionary

        Returns:
            Formatted text string
        """
        lines = [f"TEXT: {entities_data.get('text', '')}"]
        lines.append("ENTITIES:")

        # Add entities
        for entity in entities_data.get("entities", []):
            line = f"  - {entity.get('mention', '')} ({entity.get('entity_name', '')}, " \
                   f"{entity.get('entity_type', '')}) [{entity.get('confidence', 0.0):.2f}]"
            lines.append(line)

        return "\n".join(lines)

    def convert_format(
        self,
        entities_data: Dict[str, Any],
        output_format: str = "json"
    ) -> str:
        """
        Convert entity data to specified format.

        Args:
            entities_data: Entity data dictionary
            output_format: Output format (json, csv, txt)

        Returns:
            Formatted string
        """
        if output_format.lower() == "json":
            return self.to_json(entities_data)
        elif output_format.lower() == "csv":
            return self.to_csv(entities_data)
        elif output_format.lower() in ["txt", "text"]:
            return self.to_text(entities_data)
        else:
            logger.warning(f"Unknown output format: {output_format}, using JSON")
            return self.to_json(entities_data)
