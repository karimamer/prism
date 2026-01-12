import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class ModelStatus(str, Enum):
    """Status of model processing."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class EntityType(str, Enum):
    """Standard entity types."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    MISC = "MISC"
    UNKNOWN = "UNKNOWN"


class EntitySpan(BaseModel):
    """Entity mention span information."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(..., description="Start position of entity mention")
    end: int = Field(..., description="End position of entity mention")


class ModelAgreement(BaseModel):
    """Agreement information across different models."""

    model_config = ConfigDict(frozen=True)

    total_models: int = Field(..., description="Total number of models attempted")
    agreeing_models: list[str] = Field(default=[], description="Models that agree on this entity")
    confidence_range: dict[str, float] = Field(
        default={"min": 0.0, "max": 1.0}, description="Confidence range across agreeing models"
    )
    agreement_score: float = Field(..., ge=0.0, le=1.0, description="Agreement score (0-1)")


class EntityPrediction(BaseModel):
    """Individual entity prediction with metadata."""

    model_config = ConfigDict(frozen=True)

    mention: str = Field(..., description="Text mention of the entity")
    mention_span: EntitySpan = Field(..., description="Span of the mention in text")
    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_name: str = Field(..., description="Canonical name of the entity")
    entity_type: EntityType = Field(..., description="Type/category of the entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_model: str = Field(..., description="Model that generated this prediction")
    model_agreement: Optional[ModelAgreement] = Field(
        None, description="Agreement information across models"
    )


class RelationPrediction(BaseModel):
    """Individual relation prediction."""

    model_config = ConfigDict(frozen=True)

    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relation type")
    object: str = Field(..., description="Object entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_model: str = Field(..., description="Model that generated this prediction")
    subject_span: Optional[EntitySpan] = Field(None, description="Subject mention span")
    object_span: Optional[EntitySpan] = Field(None, description="Object mention span")


class ModelPredictionStats(BaseModel):
    """Statistics for predictions from a single model."""

    model_config = ConfigDict(frozen=True)

    num_entities: int = Field(..., ge=0, description="Number of entities predicted")
    num_relations: int = Field(..., ge=0, description="Number of relations predicted")
    confidence_avg: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    status: ModelStatus = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class PipelineStage(BaseModel):
    """Pipeline stage completion status."""

    model_config = ConfigDict(frozen=True)

    entity_encoding: bool = Field(..., description="Entity-focused encoding completed")
    candidate_generation: bool = Field(
        ..., description="Multi-source candidate generation completed"
    )
    cross_model_resolution: bool = Field(..., description="Cross-model entity resolution completed")
    consensus_linking: bool = Field(..., description="Consensus entity linking completed")
    structured_output: bool = Field(..., description="Structured entity output completed")


class UnifiedSystemOutput(BaseModel):
    """Comprehensive output from the unified entity resolution system."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "text": "Apple Inc. was founded by Steve Jobs.",
                "entities": [],
                "relations": [],
                "num_entities": 2,
                "num_relations": 0,
                "consensus_method": "multi_method_weighted",
            }
        },
    )

    # Core data
    text: str = Field(..., description="Original input text")
    entities: list[EntityPrediction] = Field(default=[], description="Resolved entity predictions")
    relations: list[RelationPrediction] = Field(
        default=[], description="Extracted relation predictions"
    )

    # Model information
    model_predictions: dict[str, ModelPredictionStats] = Field(
        ..., description="Statistics for each model attempted"
    )
    models_used: list[str] = Field(..., description="List of models that were attempted")

    # Pipeline information
    pipeline_stages: PipelineStage = Field(..., description="Pipeline stage completion status")

    # Statistics
    num_entities: int = Field(..., ge=0, description="Total number of entities resolved")
    num_relations: int = Field(..., ge=0, description="Total number of relations extracted")
    num_candidates: int = Field(..., ge=0, description="Number of candidate entities retrieved")

    # Metadata
    consensus_method: str = Field(..., description="Consensus method used for resolution")
    processing_timestamp: datetime = Field(
        default_factory=datetime.now, description="When processing was completed"
    )
    telemetry: Optional[Any] = Field(None, description="Pipeline telemetry data")

    # Computed properties
    @property
    def success_rate(self) -> float:
        """Calculate success rate across all models."""
        if not self.model_predictions:
            return 0.0
        successful = sum(
            1 for stats in self.model_predictions.values() if stats.status == ModelStatus.SUCCESS
        )
        return successful / len(self.model_predictions)

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence across all entities."""
        if not self.entities:
            return 0.0
        return sum(entity.confidence for entity in self.entities) / len(self.entities)

    @property
    def entity_types_found(self) -> list[str]:
        """Get list of unique entity types found."""
        return list({entity.entity_type.value for entity in self.entities})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed properties."""
        data = self.model_dump()
        data["success_rate"] = self.success_rate
        data["avg_confidence"] = self.avg_confidence
        data["entity_types_found"] = self.entity_types_found
        return data

    def to_json(self, **kwargs) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(**kwargs)

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """
        Convert to list of CSV row dictionaries (one row per entity).

        Returns:
            List of dictionaries suitable for csv.DictWriter
        """
        rows = []
        for entity in self.entities:
            row = {
                "text": self.text,
                "mention": entity.mention,
                "mention_start": entity.mention_span.start,
                "mention_end": entity.mention_span.end,
                "entity_id": entity.entity_id,
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type.value,
                "confidence": entity.confidence,
                "source_model": entity.source_model,
            }

            # Add agreement info if available
            if entity.model_agreement:
                row["agreement_score"] = entity.model_agreement.agreement_score
                row["agreeing_models"] = ";".join(entity.model_agreement.agreeing_models)
                row["total_models"] = entity.model_agreement.total_models
            else:
                row["agreement_score"] = ""
                row["agreeing_models"] = ""
                row["total_models"] = ""

            rows.append(row)

        return rows

    def to_csv_relations(self) -> list[dict[str, Any]]:
        """
        Convert relations to list of CSV row dictionaries.

        Returns:
            List of dictionaries for relation CSV export
        """
        rows = []
        for relation in self.relations:
            row = {
                "text": self.text,
                "subject": relation.subject,
                "predicate": relation.predicate,
                "object": relation.object,
                "confidence": relation.confidence,
                "source_model": relation.source_model,
            }

            # Add span info if available
            if relation.subject_span:
                row["subject_start"] = relation.subject_span.start
                row["subject_end"] = relation.subject_span.end
            else:
                row["subject_start"] = ""
                row["subject_end"] = ""

            if relation.object_span:
                row["object_start"] = relation.object_span.start
                row["object_end"] = relation.object_span.end
            else:
                row["object_start"] = ""
                row["object_end"] = ""

            rows.append(row)

        return rows


# ============================================================================
# Helper Functions for Creating Pydantic Models
# ============================================================================


def create_entity_prediction(
    mention: str,
    mention_span: tuple[int, int],
    entity_id: str,
    entity_name: str,
    entity_type: str,
    confidence: float,
    source_model: str = "unknown",
    model_agreement: Optional[dict] = None,
) -> EntityPrediction:
    """Create an EntityPrediction from raw data."""
    try:
        entity_type_enum = EntityType(entity_type.upper())
    except ValueError:
        entity_type_enum = EntityType.UNKNOWN

    span = EntitySpan(start=mention_span[0], end=mention_span[1])

    agreement = None
    if model_agreement:
        agreement = ModelAgreement(**model_agreement)

    return EntityPrediction(
        mention=mention,
        mention_span=span,
        entity_id=entity_id if entity_id is not None else "",
        entity_name=entity_name if entity_name is not None else "",
        entity_type=entity_type_enum,
        confidence=confidence,
        source_model=source_model,
        model_agreement=agreement,
    )


def create_relation_prediction(
    subject: str,
    predicate: str,
    object_: str,
    confidence: float,
    source_model: str = "unknown",
    subject_span: Optional[tuple[int, int]] = None,
    object_span: Optional[tuple[int, int]] = None,
) -> RelationPrediction:
    """Create a RelationPrediction from raw data."""
    subj_span = EntitySpan(start=subject_span[0], end=subject_span[1]) if subject_span else None
    obj_span = EntitySpan(start=object_span[0], end=object_span[1]) if object_span else None

    return RelationPrediction(
        subject=subject,
        predicate=predicate,
        object=object_,
        confidence=confidence,
        source_model=source_model,
        subject_span=subj_span,
        object_span=obj_span,
    )


def create_unified_output(
    text: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    model_predictions: dict[str, dict[str, Any]],
    models_used: list[str],
    pipeline_stages: dict[str, bool],
    num_candidates: int,
    consensus_method: str = "multi_method_weighted",
    telemetry: Optional[Any] = None,
) -> UnifiedSystemOutput:
    """Create UnifiedSystemOutput from raw system data."""
    # Convert entities
    entity_predictions = []
    for entity_data in entities:
        mention_span = entity_data.get("mention_span", [0, 0])
        if isinstance(mention_span, list) and len(mention_span) == 2:
            mention_span_tuple = (mention_span[0], mention_span[1])
        else:
            mention_span_tuple = (0, 0)

        entity_pred = create_entity_prediction(
            mention=entity_data.get("mention", ""),
            mention_span=mention_span_tuple,
            entity_id=entity_data.get("entity_id", ""),
            entity_name=entity_data.get("entity_name", ""),
            entity_type=entity_data.get("entity_type", "UNKNOWN"),
            confidence=entity_data.get("confidence", 0.0),
            source_model=entity_data.get("source_model", "unknown"),
            model_agreement=entity_data.get("model_agreement"),
        )
        entity_predictions.append(entity_pred)

    # Convert relations
    relation_predictions = []
    for relation_data in relations:
        relation_pred = create_relation_prediction(
            subject=relation_data.get("subject", ""),
            predicate=relation_data.get("predicate", ""),
            object_=relation_data.get("object", ""),
            confidence=relation_data.get("confidence", 0.0),
            source_model=relation_data.get("source_model", "unknown"),
            subject_span=relation_data.get("subject_span"),
            object_span=relation_data.get("object_span"),
        )
        relation_predictions.append(relation_pred)

    # Convert model predictions
    model_stats = {}
    for model_name, stats_data in model_predictions.items():
        try:
            status = ModelStatus(stats_data.get("status", "failed"))
        except ValueError:
            status = ModelStatus.FAILED

        model_stats[model_name] = ModelPredictionStats(
            num_entities=stats_data.get("num_entities", 0),
            num_relations=stats_data.get("num_relations", 0),
            confidence_avg=stats_data.get("confidence_avg", 0.0),
            status=status,
            error_message=stats_data.get("error_message"),
            processing_time=stats_data.get("processing_time"),
        )

    # Convert pipeline stages
    pipeline = PipelineStage(**pipeline_stages)

    return UnifiedSystemOutput(
        text=text,
        entities=entity_predictions,
        relations=relation_predictions,
        model_predictions=model_stats,
        models_used=models_used,
        pipeline_stages=pipeline,
        num_entities=len(entity_predictions),
        num_relations=len(relation_predictions),
        num_candidates=num_candidates,
        consensus_method=consensus_method,
        telemetry=telemetry,
    )


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
        self, token_spans: list[tuple[int, int]], offset_mapping: torch.Tensor
    ) -> list[tuple[int, int]]:
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

    def get_mention_text(self, input_ids: torch.Tensor, token_span: tuple[int, int]) -> str:
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
        mention_ids = input_ids[0, token_start : token_end + 1]

        # Decode tokens to text
        mention_text = self.tokenizer.decode(mention_ids, skip_special_tokens=True)

        return mention_text.strip()

    def format_entity(
        self,
        mention_span: tuple[int, int],
        entity_id: str,
        entity_name: str,
        entity_type: str,
        confidence: float,
        input_ids: Optional[torch.Tensor] = None,
        offset_mapping: Optional[torch.Tensor] = None,
        source: str = "entity_resolution",
    ) -> dict[str, Any]:
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
            "source": source,
        }

        # Add character span if available
        if char_span:
            entity["char_span"] = char_span

        return entity

    def format_entities(
        self,
        linked_entities: list[dict[str, Any]],
        input_ids: Optional[torch.Tensor] = None,
        offset_mapping: Optional[torch.Tensor] = None,
    ) -> list[dict[str, Any]]:
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
                source=source,
            )

            # Add extra fields if present
            for key, value in entity.items():
                if key not in formatted_entity:
                    formatted_entity[key] = value

            formatted_entities.append(formatted_entity)

        return formatted_entities

    def forward(
        self,
        linked_entities: list[dict[str, Any]],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        offset_mapping: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
    ) -> dict[str, Any]:
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
        entities = self.format_entities(linked_entities, input_ids, offset_mapping)

        # Get original text if possible
        original_text = text
        if original_text is None and input_ids is not None and self.tokenizer is not None:
            original_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Create output dictionary
        output = {
            "entities": entities,
            "text": original_text if original_text else "",
            "num_entities": len(entities),
        }

        return output

    def to_json(self, entities_data: dict[str, Any], indent: int = 2) -> str:
        """
        Convert entity data to JSON string.

        Args:
            entities_data: Entity data dictionary
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(entities_data, indent=indent)

    def to_csv(self, entities_data: dict[str, Any]) -> str:
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

            line = (
                f'"{text}","{mention}","{entity_id}","{entity_name}","{entity_type}",{confidence}'
            )
            csv_lines.append(line)

        return "\n".join(csv_lines)

    def to_text(self, entities_data: dict[str, Any]) -> str:
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
            line = (
                f"  - {entity.get('mention', '')} ({entity.get('entity_name', '')}, "
                f"{entity.get('entity_type', '')}) [{entity.get('confidence', 0.0):.2f}]"
            )
            lines.append(line)

        return "\n".join(lines)

    def convert_format(self, entities_data: dict[str, Any], output_format: str = "json") -> str:
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
