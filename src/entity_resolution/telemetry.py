"""
Structured telemetry and observability module.

This module provides comprehensive telemetry for the entity resolution system,
including:
- Per-stage timing and performance metrics
- Per-model success/failure tracking
- Retrieval diagnostics
- Consensus statistics
- Structured logging and metrics export
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Telemetry Data Models
# ============================================================================


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageTiming(BaseModel):
    """Timing information for a pipeline stage."""

    stage_name: str = Field(..., description="Name of the pipeline stage")
    status: StageStatus = Field(..., description="Stage status")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds."""
        return self.duration_ms / 1000.0 if self.duration_ms else None


class ModelTelemetry(BaseModel):
    """Telemetry for a single model."""

    model_name: str = Field(..., description="Model name")
    status: StageStatus = Field(..., description="Model execution status")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")
    num_entities: int = Field(default=0, description="Number of entities predicted")
    num_relations: int = Field(default=0, description="Number of relations predicted")
    avg_confidence: float = Field(default=0.0, description="Average confidence score")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Model-specific metadata")


class RetrievalDiagnostics(BaseModel):
    """Diagnostics for retrieval performance."""

    num_queries: int = Field(..., description="Number of retrieval queries")
    total_candidates: int = Field(..., description="Total candidates retrieved")
    avg_candidates_per_query: float = Field(..., description="Average candidates per query")
    avg_retrieval_time_ms: float = Field(..., description="Average retrieval time in ms")
    top_1_count: int = Field(default=0, description="Queries where top-1 was correct")
    top_5_count: int = Field(default=0, description="Queries where answer in top-5")
    top_10_count: int = Field(default=0, description="Queries where answer in top-10")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional diagnostics")

    @property
    def recall_at_1(self) -> float:
        """Recall@1 metric."""
        return self.top_1_count / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def recall_at_5(self) -> float:
        """Recall@5 metric."""
        return self.top_5_count / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def recall_at_10(self) -> float:
        """Recall@10 metric."""
        return self.top_10_count / self.num_queries if self.num_queries > 0 else 0.0


class ConsensusStatistics(BaseModel):
    """Statistics from consensus resolution."""

    total_entities: int = Field(..., description="Total entities after consensus")
    avg_agreement_score: float = Field(..., description="Average model agreement score")
    agreement_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of agreement counts"
    )
    confidence_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of confidence scores"
    )
    num_conflicts: int = Field(default=0, description="Number of conflicts resolved")
    num_overlaps: int = Field(default=0, description="Number of overlapping mentions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional statistics")


class PipelineTelemetry(BaseModel):
    """Complete telemetry for a pipeline execution."""

    pipeline_id: str = Field(..., description="Unique pipeline execution ID")
    start_time: datetime = Field(..., description="Pipeline start time")
    end_time: Optional[datetime] = Field(None, description="Pipeline end time")
    total_duration_ms: Optional[float] = Field(None, description="Total duration in ms")

    # Stage timings
    stages: dict[str, StageTiming] = Field(
        default_factory=dict, description="Timing for each stage"
    )

    # Model telemetry
    models: dict[str, ModelTelemetry] = Field(
        default_factory=dict, description="Telemetry for each model"
    )

    # Diagnostics
    retrieval_diagnostics: Optional[RetrievalDiagnostics] = Field(
        None, description="Retrieval diagnostics"
    )
    consensus_statistics: Optional[ConsensusStatistics] = Field(
        None, description="Consensus statistics"
    )

    # Overall metrics
    num_entities_final: int = Field(default=0, description="Final number of entities")
    num_relations_final: int = Field(default=0, description="Final number of relations")
    success: bool = Field(default=False, description="Whether pipeline succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def total_duration_seconds(self) -> Optional[float]:
        """Get total duration in seconds."""
        return self.total_duration_ms / 1000.0 if self.total_duration_ms else None


# ============================================================================
# Telemetry Collector
# ============================================================================


class TelemetryCollector:
    """
    Collects and aggregates telemetry during pipeline execution.

    Usage:
        collector = TelemetryCollector()

        with collector.stage("entity_encoding"):
            # Do encoding work
            pass

        with collector.model("atg"):
            # Run ATG model
            pass

        telemetry = collector.finalize()
    """

    def __init__(self, pipeline_id: Optional[str] = None):
        """
        Initialize telemetry collector.

        Args:
            pipeline_id: Optional pipeline ID (generated if not provided)
        """
        import uuid

        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

        self._stages: dict[str, StageTiming] = {}
        self._models: dict[str, ModelTelemetry] = {}
        self._stage_stack: list[tuple[str, float]] = []
        self._model_stack: list[tuple[str, float]] = []

        self.retrieval_diagnostics: Optional[RetrievalDiagnostics] = None
        self.consensus_statistics: Optional[ConsensusStatistics] = None

        self.num_entities_final = 0
        self.num_relations_final = 0
        self.success = False
        self.error_message: Optional[str] = None
        self.metadata: dict[str, Any] = {}

    @contextmanager
    def stage(self, stage_name: str, **metadata):
        """
        Context manager to time a pipeline stage.

        Args:
            stage_name: Name of the stage
            **metadata: Additional metadata

        Yields:
            StageTiming object
        """
        start_time = time.time()
        start_dt = datetime.now()

        stage_timing = StageTiming(
            stage_name=stage_name,
            status=StageStatus.IN_PROGRESS,
            start_time=start_dt,
            metadata=metadata,
        )

        self._stages[stage_name] = stage_timing
        self._stage_stack.append((stage_name, start_time))

        try:
            yield stage_timing
            # Success
            stage_timing.status = StageStatus.SUCCESS
        except Exception as e:
            # Failure
            stage_timing.status = StageStatus.FAILED
            stage_timing.metadata["error"] = str(e)
            raise
        finally:
            # Record timing
            end_time = time.time()
            stage_timing.end_time = datetime.now()
            stage_timing.duration_ms = (end_time - start_time) * 1000.0

            self._stage_stack.pop()

            logger.debug(
                f"Stage '{stage_name}' completed in {stage_timing.duration_ms:.2f}ms "
                f"(status: {stage_timing.status})"
            )

    @contextmanager
    def model(self, model_name: str, **metadata):
        """
        Context manager to time a model execution.

        Args:
            model_name: Name of the model
            **metadata: Additional metadata

        Yields:
            ModelTelemetry object
        """
        start_time = time.time()
        start_dt = datetime.now()

        model_telemetry = ModelTelemetry(
            model_name=model_name,
            status=StageStatus.IN_PROGRESS,
            start_time=start_dt,
            metadata=metadata,
        )

        self._models[model_name] = model_telemetry
        self._model_stack.append((model_name, start_time))

        try:
            yield model_telemetry
            # Success
            model_telemetry.status = StageStatus.SUCCESS
        except Exception as e:
            # Failure
            model_telemetry.status = StageStatus.FAILED
            model_telemetry.error_message = str(e)
            raise
        finally:
            # Record timing
            end_time = time.time()
            model_telemetry.end_time = datetime.now()
            model_telemetry.duration_ms = (end_time - start_time) * 1000.0

            self._model_stack.pop()

            logger.debug(
                f"Model '{model_name}' completed in {model_telemetry.duration_ms:.2f}ms "
                f"(status: {model_telemetry.status}, entities: {model_telemetry.num_entities})"
            )

    def record_retrieval_diagnostics(self, diagnostics: RetrievalDiagnostics):
        """Record retrieval diagnostics."""
        self.retrieval_diagnostics = diagnostics

    def record_consensus_statistics(self, statistics: ConsensusStatistics):
        """Record consensus statistics."""
        self.consensus_statistics = statistics

    def finalize(self, success: bool = True, error: Optional[str] = None) -> PipelineTelemetry:
        """
        Finalize telemetry collection.

        Args:
            success: Whether pipeline succeeded
            error: Error message if failed

        Returns:
            Complete pipeline telemetry
        """
        self.end_time = datetime.now()
        self.success = success
        self.error_message = error

        total_duration_ms = (
            (self.end_time - self.start_time).total_seconds() * 1000.0 if self.end_time else None
        )

        telemetry = PipelineTelemetry(
            pipeline_id=self.pipeline_id,
            start_time=self.start_time,
            end_time=self.end_time,
            total_duration_ms=total_duration_ms,
            stages=self._stages,
            models=self._models,
            retrieval_diagnostics=self.retrieval_diagnostics,
            consensus_statistics=self.consensus_statistics,
            num_entities_final=self.num_entities_final,
            num_relations_final=self.num_relations_final,
            success=success,
            error_message=error,
            metadata=self.metadata,
        )

        # Log summary
        logger.info(
            f"Pipeline {self.pipeline_id[:8]} completed in {total_duration_ms:.2f}ms "
            f"({len(self._stages)} stages, {len(self._models)} models, "
            f"{self.num_entities_final} entities, success={success})"
        )

        return telemetry

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        telemetry = self.finalize()
        return telemetry.model_dump()


# ============================================================================
# Telemetry Exporters
# ============================================================================


class TelemetryExporter:
    """Base class for telemetry exporters."""

    def export(self, telemetry: PipelineTelemetry):
        """Export telemetry data."""
        raise NotImplementedError


class JsonFileExporter(TelemetryExporter):
    """Export telemetry to JSON file."""

    def __init__(self, output_path: str):
        """Initialize with output path."""
        self.output_path = output_path

    def export(self, telemetry: PipelineTelemetry):
        """Export to JSON file."""
        import json

        with open(self.output_path, "w") as f:
            json.dump(telemetry.model_dump(), f, indent=2, default=str)

        logger.info(f"Exported telemetry to {self.output_path}")


class LogExporter(TelemetryExporter):
    """Export telemetry to structured logs."""

    def export(self, telemetry: PipelineTelemetry):
        """Log telemetry data."""
        logger.info(
            "Pipeline Telemetry",
            extra={
                "pipeline_id": telemetry.pipeline_id,
                "duration_ms": telemetry.total_duration_ms,
                "num_stages": len(telemetry.stages),
                "num_models": len(telemetry.models),
                "num_entities": telemetry.num_entities_final,
                "success": telemetry.success,
            },
        )

        # Log each stage
        for stage_name, stage in telemetry.stages.items():
            logger.info(
                f"Stage: {stage_name}",
                extra={
                    "stage": stage_name,
                    "status": stage.status,
                    "duration_ms": stage.duration_ms,
                },
            )

        # Log each model
        for model_name, model in telemetry.models.items():
            logger.info(
                f"Model: {model_name}",
                extra={
                    "model": model_name,
                    "status": model.status,
                    "duration_ms": model.duration_ms,
                    "num_entities": model.num_entities,
                    "avg_confidence": model.avg_confidence,
                },
            )


__all__ = [
    "TelemetryCollector",
    "PipelineTelemetry",
    "StageTiming",
    "ModelTelemetry",
    "RetrievalDiagnostics",
    "ConsensusStatistics",
    "StageStatus",
    "TelemetryExporter",
    "JsonFileExporter",
    "LogExporter",
]
