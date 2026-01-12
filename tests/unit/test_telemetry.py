"""
Unit tests for telemetry module.

Tests the TelemetryCollector and related telemetry data models.
"""

import time

import pytest

from entity_resolution.telemetry import (
    ConsensusStatistics,
    ModelTelemetry,
    PipelineTelemetry,
    RetrievalDiagnostics,
    StageStatus,
    StageTiming,
    TelemetryCollector,
)


class TestTelemetryDataModels:
    """Test telemetry data models."""

    def test_stage_timing_creation(self):
        """Test StageTiming creation and properties."""
        stage = StageTiming(
            stage_name="test_stage",
            status=StageStatus.SUCCESS,
            duration_ms=1500.0,
        )

        assert stage.stage_name == "test_stage"
        assert stage.status == StageStatus.SUCCESS
        assert stage.duration_ms == 1500.0
        assert stage.duration_seconds == 1.5

    def test_model_telemetry_creation(self):
        """Test ModelTelemetry creation."""
        model_tel = ModelTelemetry(
            model_name="atg",
            status=StageStatus.SUCCESS,
            duration_ms=2000.0,
            num_entities=5,
            num_relations=3,
            avg_confidence=0.85,
        )

        assert model_tel.model_name == "atg"
        assert model_tel.num_entities == 5
        assert model_tel.num_relations == 3
        assert model_tel.avg_confidence == 0.85

    def test_retrieval_diagnostics_metrics(self):
        """Test RetrievalDiagnostics computed metrics."""
        diag = RetrievalDiagnostics(
            num_queries=100,
            total_candidates=500,
            avg_candidates_per_query=5.0,
            avg_retrieval_time_ms=50.0,
            top_1_count=70,
            top_5_count=90,
            top_10_count=95,
        )

        assert diag.recall_at_1 == 0.7
        assert diag.recall_at_5 == 0.9
        assert diag.recall_at_10 == 0.95

    def test_consensus_statistics_creation(self):
        """Test ConsensusStatistics creation."""
        stats = ConsensusStatistics(
            total_entities=10,
            avg_agreement_score=0.75,
            num_conflicts=2,
            num_overlaps=3,
        )

        assert stats.total_entities == 10
        assert stats.avg_agreement_score == 0.75
        assert stats.num_conflicts == 2


class TestTelemetryCollector:
    """Test TelemetryCollector functionality."""

    def test_collector_initialization(self):
        """Test collector initializes correctly."""
        collector = TelemetryCollector(pipeline_id="test-123")

        assert collector.pipeline_id == "test-123"
        assert collector.start_time is not None
        assert collector.end_time is None
        assert collector.success is False

    def test_stage_context_manager(self):
        """Test stage context manager tracks timing."""
        collector = TelemetryCollector()

        with collector.stage("test_stage") as stage:
            time.sleep(0.01)  # Small delay
            assert stage.status == StageStatus.IN_PROGRESS

        # After context exits, should be marked success
        assert collector._stages["test_stage"].status == StageStatus.SUCCESS
        assert collector._stages["test_stage"].duration_ms > 0

    def test_stage_context_manager_with_error(self):
        """Test stage context manager handles errors."""
        collector = TelemetryCollector()

        with pytest.raises(ValueError):
            with collector.stage("test_stage"):
                raise ValueError("Test error")

        # Should be marked as failed
        assert collector._stages["test_stage"].status == StageStatus.FAILED
        assert "error" in collector._stages["test_stage"].metadata

    def test_model_context_manager(self):
        """Test model context manager tracks performance."""
        collector = TelemetryCollector()

        with collector.model("atg") as model_tel:
            time.sleep(0.01)  # Small delay
            model_tel.num_entities = 5
            model_tel.num_relations = 3
            model_tel.avg_confidence = 0.85

        # Check model telemetry was recorded
        assert "atg" in collector._models
        assert collector._models["atg"].num_entities == 5
        assert collector._models["atg"].num_relations == 3
        assert collector._models["atg"].avg_confidence == 0.85
        assert collector._models["atg"].duration_ms > 0

    def test_record_retrieval_diagnostics(self):
        """Test recording retrieval diagnostics."""
        collector = TelemetryCollector()

        diag = RetrievalDiagnostics(
            num_queries=10,
            total_candidates=50,
            avg_candidates_per_query=5.0,
            avg_retrieval_time_ms=25.0,
        )

        collector.record_retrieval_diagnostics(diag)

        assert collector.retrieval_diagnostics == diag

    def test_record_consensus_statistics(self):
        """Test recording consensus statistics."""
        collector = TelemetryCollector()

        stats = ConsensusStatistics(
            total_entities=15,
            avg_agreement_score=0.8,
        )

        collector.record_consensus_statistics(stats)

        assert collector.consensus_statistics == stats

    def test_finalize_success(self):
        """Test finalizing telemetry with success."""
        collector = TelemetryCollector()

        # Simulate some work
        with collector.stage("stage1"):
            time.sleep(0.01)

        with collector.model("model1") as model_tel:
            model_tel.num_entities = 3

        collector.num_entities_final = 3
        collector.num_relations_final = 2

        telemetry = collector.finalize(success=True)

        assert isinstance(telemetry, PipelineTelemetry)
        assert telemetry.success is True
        assert telemetry.error_message is None
        assert telemetry.num_entities_final == 3
        assert telemetry.num_relations_final == 2
        assert telemetry.total_duration_ms > 0
        assert len(telemetry.stages) == 1
        assert len(telemetry.models) == 1

    def test_finalize_with_error(self):
        """Test finalizing telemetry with error."""
        collector = TelemetryCollector()

        telemetry = collector.finalize(success=False, error="Test error")

        assert telemetry.success is False
        assert telemetry.error_message == "Test error"

    def test_full_pipeline_telemetry(self):
        """Test complete pipeline telemetry collection."""
        collector = TelemetryCollector()

        # Simulate pipeline stages
        with collector.stage("entity_encoding"):
            time.sleep(0.01)

        with collector.stage("candidate_retrieval"):
            time.sleep(0.01)

        # Simulate models
        with collector.model("atg") as model_tel:
            time.sleep(0.01)
            model_tel.num_entities = 5
            model_tel.avg_confidence = 0.9

        with collector.model("relik") as model_tel:
            time.sleep(0.01)
            model_tel.num_entities = 4
            model_tel.avg_confidence = 0.85

        # Add diagnostics
        collector.record_retrieval_diagnostics(
            RetrievalDiagnostics(
                num_queries=1,
                total_candidates=10,
                avg_candidates_per_query=10.0,
                avg_retrieval_time_ms=50.0,
            )
        )

        collector.record_consensus_statistics(
            ConsensusStatistics(
                total_entities=6,
                avg_agreement_score=0.75,
            )
        )

        collector.num_entities_final = 6
        collector.num_relations_final = 3

        # Finalize
        telemetry = collector.finalize(success=True)

        # Verify complete telemetry
        assert telemetry.success is True
        assert len(telemetry.stages) == 2
        assert len(telemetry.models) == 2
        assert telemetry.retrieval_diagnostics is not None
        assert telemetry.consensus_statistics is not None
        assert telemetry.num_entities_final == 6
        assert telemetry.num_relations_final == 3

        # Verify stages
        assert "entity_encoding" in telemetry.stages
        assert "candidate_retrieval" in telemetry.stages
        assert telemetry.stages["entity_encoding"].status == StageStatus.SUCCESS

        # Verify models
        assert "atg" in telemetry.models
        assert "relik" in telemetry.models
        assert telemetry.models["atg"].num_entities == 5
        assert telemetry.models["relik"].num_entities == 4


class TestPipelineTelemetry:
    """Test PipelineTelemetry model."""

    def test_total_duration_seconds(self):
        """Test duration conversion."""
        from datetime import datetime

        start = datetime.now()
        telemetry = PipelineTelemetry(
            pipeline_id="test",
            start_time=start,
            total_duration_ms=5000.0,
        )

        assert telemetry.total_duration_seconds == 5.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        collector = TelemetryCollector()
        collector.num_entities_final = 5
        telemetry = collector.finalize(success=True)

        data = telemetry.model_dump()

        assert isinstance(data, dict)
        assert "pipeline_id" in data
        assert "success" in data
        assert data["num_entities_final"] == 5
