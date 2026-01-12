# PRISM: Entity Resolution System

**⚠️ Prism is currently in its early experimental phase and should be considered unstable.**

A unified entity resolution system that combines state-of-the-art techniques from multiple research papers (ReLiK, SpEL, UniRel, ATG, and OneNet) to efficiently identify, classify, and link entities within text to a knowledge base.

## Features

- **Multi-Model Ensemble**: Integrates 4 state-of-the-art entity resolution models (ATG, ReLiK, SPEL, UniREL)
- **Plugin Architecture**: Extensible model adapter system - add new models without modifying core code
- **Unified Pipeline**: Complete retriever-reader-consensus architecture for comprehensive entity resolution
- **Dense Retrieval**: Bi-encoder architecture with FAISS indexing for efficient candidate retrieval
- **Consensus Resolution**: Multi-method consensus mechanism to combine predictions and handle overlapping mentions
- **Structured Telemetry**: Comprehensive performance tracking with per-stage timing, model metrics, and diagnostics
- **Pydantic Validation**: Strong typing and validation using Pydantic v2 for configuration and outputs
- **Flexible I/O**: Supports JSON, CSV, and text formats with structured Pydantic models
- **Knowledge Base Integration**: FAISS-based vector database with versioning for fast entity similarity search
- **Batch Processing**: Efficient processing of multiple documents with configurable batch sizes

## Architecture

The system follows a 5-stage pipeline:

### 1. Entity-Focused Encoding

- Optional entity-focused encoder for enhanced representation
- Specialized entity knowledge embeddings
- Multi-type entity classification support

### 2. Multi-Source Candidate Generation

- Dense bi-encoder retrieval using FAISS
- Top-k candidate selection from knowledge base
- Efficient vector similarity search
- Support for millions of entities

### 3. Cross-Model Entity Resolution

Four parallel entity resolution models:

- **ATG (Autoregressive Text-to-Graph)**: Joint entity and relation extraction with decoder-based generation
- **ReLiK (Retrieval-based Entity Linking)**: Dense retrieval with reader re-ranking for entity linking
- **SPEL (Structured Prediction for Entity Linking)**: Fixed candidate set with structured prediction
- **UniREL (Unified Representation Learning)**: Unified entity-relation representation with interaction maps

### 4. Consensus Entity Linking

- Multi-method weighted consensus across model predictions
- Confidence calibration and aggregation
- Model agreement scoring and metadata
- Threshold-based filtering for quality control

### 5. Structured Entity Output

- Pydantic v2 models for strong typing
- Rich metadata including model agreement, confidence ranges
- JSON serialization with datetime support
- Pipeline stage tracking and model statistics

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd prism

# Install dependencies with uv
uv sync
```

## Quick Start

### Basic Usage

```bash
# Process a text file with entity resolution
uv run python -m src.entity_resolution.run_entity_resolution \
    --input test_input.txt \
    --output results.json \
    --entities sample_entities.json \
    --format json
```

### Python API

```python
from entity_resolution import UnifiedEntityResolutionSystem
from entity_resolution.validation import SystemConfig

# Initialize with configuration
config = SystemConfig(
    # Model selection
    retriever_model="sentence-transformers/all-MiniLM-L6-v2",
    reader_model="distilbert-base-uncased",

    # Enable models
    use_improved_atg=True,
    use_relik=True,
    use_spel=True,
    use_unirel=True,

    # Configuration
    max_seq_length=512,
    max_entity_length=64,
    top_k_candidates=50,
    batch_size=8,

    # Thresholds
    consensus_threshold=0.5,
    relik_entity_threshold=0.3,
    spel_entity_threshold=0.3,
    unirel_entity_threshold=0.3,

    # Hardware
    use_gpu=True  # Set to False for CPU-only
)

# Create system
system = UnifiedEntityResolutionSystem(config)

# Load entity knowledge base
num_entities = system.load_entities("sample_entities.json")
print(f"Loaded {num_entities} entities")

# Process single text
result = system.process_text("Apple Inc. was founded by Steve Jobs in 1976.")

# Access results
print(f"Found {result.num_entities} entities")
for entity in result.entities:
    print(f"  - {entity.mention} → {entity.entity_name} ({entity.confidence:.2f})")

# Process batch
texts = [
    "Microsoft was founded by Bill Gates.",
    "Google is headquartered in California."
]
results = system.process_batch(texts)
```

## Input Data Formats

### Entity Data (JSON)

```json
{
  "Q312": {
    "id": "Q312",
    "name": "Apple Inc.",
    "type": "ORGANIZATION",
    "description": "American multinational technology company"
  }
}
```

### Text Input

- Plain text files (one document per line)
- JSON arrays of strings
- JSON objects with text fields

## Output Format

The system returns a `UnifiedSystemOutput` Pydantic model with rich metadata:

```json
{
  "text": "Apple Inc. was founded by Steve Jobs in 1976.",
  "entities": [
    {
      "mention": "Apple Inc.",
      "mention_span": {
        "start": 0,
        "end": 10
      },
      "entity_id": "Q312",
      "entity_name": "Apple Inc.",
      "entity_type": "ORGANIZATION",
      "confidence": 0.92,
      "source_model": "consensus",
      "model_agreement": {
        "total_models": 4,
        "agreeing_models": ["atg", "relik", "spel"],
        "confidence_range": {
          "min": 0.85,
          "max": 0.95
        },
        "agreement_score": 0.75
      }
    },
    {
      "mention": "Steve Jobs",
      "mention_span": {
        "start": 27,
        "end": 37
      },
      "entity_id": "Q19837",
      "entity_name": "Steve Jobs",
      "entity_type": "PERSON",
      "confidence": 0.89,
      "source_model": "consensus",
      "model_agreement": {
        "total_models": 4,
        "agreeing_models": ["atg", "relik"],
        "confidence_range": {
          "min": 0.82,
          "max": 0.91
        },
        "agreement_score": 0.5
      }
    }
  ],
  "relations": [],
  "num_entities": 2,
  "num_relations": 0,
  "num_candidates": 50,
  "consensus_method": "multi_method_weighted",
  "models_used": ["atg", "relik", "spel", "unirel", "reader"],
  "model_predictions": {
    "atg": {
      "num_entities": 2,
      "num_relations": 0,
      "confidence_avg": 0.88,
      "status": "success"
    },
    "relik": {
      "num_entities": 2,
      "num_relations": 0,
      "confidence_avg": 0.9,
      "status": "success"
    }
  },
  "pipeline_stages": {
    "entity_encoding": true,
    "candidate_generation": true,
    "cross_model_resolution": true,
    "consensus_linking": true,
    "structured_output": true
  },
  "processing_timestamp": "2025-10-26T10:30:45.123456",
  "telemetry": {
    "pipeline_id": "7679417a-af35-4263-9bfe-82c768c1884",
    "total_duration_ms": 164.798,
    "stages": {
      "candidate_retrieval": {
        "stage_name": "candidate_retrieval",
        "status": "success",
        "duration_ms": 12.057
      },
      "consensus_resolution": {
        "stage_name": "consensus_resolution",
        "status": "success",
        "duration_ms": 0.413
      }
    },
    "models": {
      "atg": {
        "model_name": "atg",
        "status": "success",
        "duration_ms": 49.607,
        "num_entities": 1,
        "num_relations": 0,
        "avg_confidence": 0.7
      },
      "relik": {
        "model_name": "relik",
        "status": "success",
        "duration_ms": 43.302,
        "num_entities": 0,
        "avg_confidence": 0.0
      }
    },
    "retrieval_diagnostics": {
      "num_queries": 1,
      "total_candidates": 10,
      "avg_candidates_per_query": 10.0,
      "avg_retrieval_time_ms": 12.21
    },
    "consensus_statistics": {
      "total_entities": 1,
      "avg_agreement_score": 0.4
    },
    "success": true
  }
}
```

## Structured Telemetry

The system includes comprehensive telemetry tracking to monitor performance and diagnose issues:

### Enable Telemetry

```python
# Telemetry is enabled by default
result = system.process_text(text, enable_telemetry=True)

# Access telemetry data
telemetry = result.telemetry
print(f"Total duration: {telemetry.total_duration_ms}ms")
print(f"Number of stages: {len(telemetry.stages)}")

# Per-stage timing
for stage_name, stage_timing in telemetry.stages.items():
    print(f"{stage_name}: {stage_timing.duration_ms}ms ({stage_timing.status})")

# Per-model performance
for model_name, model_tel in telemetry.models.items():
    print(f"{model_name}: {model_tel.num_entities} entities in {model_tel.duration_ms}ms")

# Retrieval diagnostics
if telemetry.retrieval_diagnostics:
    print(f"Retrieval: {telemetry.retrieval_diagnostics.avg_retrieval_time_ms}ms")
    print(f"Candidates: {telemetry.retrieval_diagnostics.total_candidates}")

# Consensus statistics
if telemetry.consensus_statistics:
    print(f"Agreement score: {telemetry.consensus_statistics.avg_agreement_score}")
```

### Export Telemetry

```python
from entity_resolution.telemetry import JsonFileExporter, LogExporter

# Export to JSON file
exporter = JsonFileExporter("telemetry.json")
exporter.export(result.telemetry)

# Export to structured logs
log_exporter = LogExporter()
log_exporter.export(result.telemetry)
```

### Telemetry Data Includes

- **Pipeline-level**: Total duration, success/failure, error messages
- **Stage-level**: Timing for entity encoding, retrieval, consensus resolution
- **Model-level**: Per-model timing, entity/relation counts, confidence scores
- **Retrieval diagnostics**: Number of queries, candidates, retrieval time
- **Consensus statistics**: Agreement scores, conflict counts

## Plugin Architecture

The system uses an extensible adapter pattern to add new entity resolution models without modifying core code.

### Using Existing Adapters

```python
from entity_resolution.models.base_adapter import ModelRegistry

# List available adapters
print(ModelRegistry.list_adapters())  # ['atg', 'relik', 'spel', 'unirel']

# Create adapter instance
from entity_resolution.models.atg import ATGConfig

config = ATGConfig(
    encoder_model="distilbert-base-uncased",
    decoder_layers=6,
    max_span_length=12
)

adapter = ModelRegistry.create("atg", config=config)

# Use adapter for predictions
prediction = adapter.predict("Apple Inc. is a technology company.")
print(f"Found {len(prediction.entities)} entities")
```

### Creating Custom Adapters

```python
from entity_resolution.models.base_adapter import (
    BaseModelAdapter,
    ModelMetadata,
    ModelPrediction,
    register_adapter,
)

@register_adapter("my_model")
class MyModelAdapter(BaseModelAdapter):
    """Adapter for custom entity resolution model."""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="MyModel",
            version="1.0.0",
            model_type="entity_linking",
            capabilities=["entity_linking"],
            required_inputs=["text"],
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        # Your model's forward pass
        return self.model(input_ids, attention_mask)

    def predict(self, text, candidates=None, **kwargs):
        # Your prediction logic
        entities = self._extract_entities(text)

        return ModelPrediction(
            entities=entities,
            relations=[],
            confidence=0.9,
        )

    def predict_batch(self, texts, candidates=None, **kwargs):
        return [self.predict(text, candidates) for text in texts]

# Model is automatically registered and available
adapter = ModelRegistry.create("my_model", config=my_config)
```

See [`src/entity_resolution/models/adapters/README.md`](src/entity_resolution/models/adapters/README.md) for complete documentation.

## Configuration

Key configuration parameters using Pydantic `SystemConfig`:

```python
from entity_resolution.validation import SystemConfig

config = SystemConfig(
    # Model Selection
    retriever_model="sentence-transformers/all-MiniLM-L6-v2",
    reader_model="distilbert-base-uncased",

    # Enable/Disable Models
    use_improved_atg=True,         # ATG model
    use_relik=True,                # ReLiK model
    use_spel=True,                 # SPEL model
    use_unirel=True,               # UniREL model
    use_entity_encoder=False,      # Optional entity-focused encoder
    use_candidate_generator=False, # Optional advanced candidate generation

    # Sequence Configuration
    max_seq_length=512,            # Maximum input sequence length
    max_entity_length=64,          # Maximum entity description length

    # Retrieval Configuration
    top_k_candidates=50,           # Number of candidates to retrieve

    # Model-Specific Configuration
    atg_decoder_layers=6,          # ATG decoder layers
    atg_max_span_length=12,        # ATG maximum entity span
    relik_top_k=50,                # ReLiK retrieval top-k
    relik_num_el_passages=50,      # ReLiK entity linking passages
    spel_fixed_candidate_set_size=1000,  # SPEL candidate set size

    # Thresholds
    consensus_threshold=0.5,       # Consensus confidence threshold
    relik_entity_threshold=0.3,    # ReLiK entity threshold
    spel_entity_threshold=0.3,     # SPEL entity threshold
    unirel_entity_threshold=0.3,   # UniREL entity threshold

    # Processing Configuration
    batch_size=8,                  # Batch size for processing
    use_gpu=True,                  # GPU acceleration
    gradient_checkpointing=False,  # Memory optimization
    quantization=None,             # "int8", "fp16", or None

    # Paths
    cache_dir="./cache",           # Cache directory
    index_path=None                # Optional pre-built index path
)
```

## Command Line Interface

```bash
uv run python -m entity_resolution.run_entity_resolution [OPTIONS]

Options:
  --input, -i          Input text file
  --output, -o         Output file
  --entities, -e       Entity data file (JSON/CSV)
  --format, -f         Output format (json/csv/txt)
  --batch_size, -b     Batch size for processing
  --top_k, -k          Number of candidates to retrieve
  --threshold, -t      Confidence threshold
  --cache_dir, -c      Cache directory
  --verbose, -v        Verbose output
```

## Testing

The project includes comprehensive test coverage:

### Run All Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/entity_resolution
```

### Run E2E Tests

```bash
# Run end-to-end tests with real models (slower)
uv run pytest -m e2e

# Skip E2E tests (faster, for development)
uv run pytest -m "not e2e"
```

### Run Specific Test Categories

```bash
# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# Specific test file
uv run pytest tests/integration/test_e2e_pipeline.py -v
```

### Test Coverage

The project includes **248 comprehensive tests**:

- **Unit Tests** (15 telemetry + others): Individual model components
  - ATG, ReLiK, SPEL, UniREL model implementations
  - Telemetry collector and exporters (88% coverage)
  - Knowledge base operations
  - Validation and configuration

- **Integration Tests** (~60 tests): Full pipeline integration
  - E2E pipeline with real models
  - Unicode and emoji handling
  - Overlapping entity resolution
  - NIL/NME entity handling
  - Consensus tie-breaking
  - Batch processing edge cases
  - Error recovery and graceful degradation

- **Golden Fixtures** (10 tests): Regression testing
  - Deterministic output validation
  - Backwards compatibility checks
  - Schema consistency verification
  - CSV export stability

**Test Categories:**

- System initialization and model loading
- Single text and batch processing
- Knowledge base operations (add, update, remove entities)
- Error handling and edge cases
- Model ensemble behavior and consensus
- Performance characteristics and telemetry
- Output serialization (JSON, CSV)
- Unicode normalization and special characters

## Development

### Add Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --group dev package-name

# Add from a specific index
uv add --index-url https://download.pytorch.org/whl/cpu torch
```

### Code Quality

```bash
# Run linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Project Structure

```
prism/
├── src/
│   └── entity_resolution/
│       ├── models/
│       │   ├── adapters/               # Plugin adapter system
│       │   │   ├── __init__.py         # Adapter exports
│       │   │   ├── README.md           # Adapter documentation
│       │   │   └── atg_adapter.py      # ATG adapter implementation
│       │   ├── atg.py                  # ATG model implementation
│       │   ├── relik/
│       │   │   ├── model.py            # ReLiK model
│       │   │   ├── retriever.py        # ReLiK retriever
│       │   │   ├── reader.py           # ReLiK reader
│       │   │   ├── config.py           # ReLiK configuration
│       │   │   ├── unified_integration.py  # Enhanced ReLiK integration
│       │   │   ├── confidence_calibration.py  # Confidence calibration
│       │   │   └── dynamic_index.py    # Dynamic index updates
│       │   ├── spel/
│       │   │   ├── model.py            # SPEL model
│       │   │   ├── candidate_sets.py   # SPEL candidate management
│       │   │   ├── aggregation.py      # SPEL aggregation
│       │   │   └── config.py           # SPEL configuration
│       │   ├── unirel/
│       │   │   ├── model.py            # UniREL model
│       │   │   ├── interaction_map.py  # Interaction maps
│       │   │   └── config.py           # UniREL configuration
│       │   ├── base_adapter.py         # Base adapter interface
│       │   ├── retriever.py            # Base retriever
│       │   ├── reader.py               # Base reader
│       │   ├── consensus.py            # Consensus module
│       │   ├── output.py               # Pydantic output models
│       │   └── entity_encoder.py       # Entity-focused encoder
│       ├── database/
│       │   └── vector_store.py         # FAISS knowledge base with versioning
│       ├── telemetry.py                # Structured telemetry & observability
│       ├── unified_system.py           # Main pipeline orchestrator
│       ├── validation.py               # Pydantic config & validation
│       └── run_entity_resolution.py    # CLI interface
├── tests/
│   ├── unit/
│   │   ├── test_atg.py                 # ATG unit tests
│   │   ├── test_relik.py               # ReLiK unit tests
│   │   ├── test_spel.py                # SPEL unit tests
│   │   ├── test_unirel.py              # UniREL unit tests
│   │   └── test_telemetry.py           # Telemetry tests (15 tests, 88% coverage)
│   ├── integration/
│   │   ├── test_e2e_pipeline.py        # E2E pipeline tests
│   │   ├── test_e2e_edge_cases.py      # Edge case tests
│   │   ├── test_golden_fixtures.py     # Golden fixture tests
│   │   └── test_unified_relik_integration.py  # ReLiK integration tests
│   └── fixtures/
│       ├── __init__.py                 # Fixture utilities
│       └── golden_outputs/             # Golden test fixtures
│           ├── sample_1_input.json     # Test input sample 1
│           ├── sample_1_output.json    # Expected output sample 1
│           ├── sample_2_input.json     # Test input sample 2
│           ├── sample_2_output.json    # Expected output sample 2
│           └── tolerance_config.json   # Tolerance configuration
├── pyproject.toml                      # Project dependencies
├── uv.lock                             # Locked dependencies
└── README.md                           # This file
```

## Dependencies

Core dependencies managed with `uv`:

- **PyTorch** 2.6+ - Deep learning framework
- **Transformers** 4.49+ - Hugging Face model library
- **FAISS-CPU** 1.10+ - Fast vector similarity search
- **Pydantic** 2.10+ - Data validation and settings management
- **NumPy**, **Pandas** - Data manipulation
- **scikit-learn** - Machine learning utilities

See `pyproject.toml` for complete dependency list.

## Research Background

This implementation integrates techniques from multiple state-of-the-art research papers:

### Models Implemented

1. **ATG (Autoregressive Text-to-Graph)** - Generates entities and relations autoregressively using encoder-decoder architecture

2. **ReLiK (Retrieval-based Entity Linking)** - Two-stage retrieval-then-read approach for entity linking with dense passage retrieval

3. **SPEL (Structured Prediction for Entity Linking)** - Fixed candidate set approach with structured prediction over mention-entity pairs

4. **UniREL (Unified Representation Learning)** - Joint entity-relation extraction using unified representations and interaction maps

### Key Innovations

- **Multi-Method Consensus**: Combines predictions from multiple models using weighted voting and confidence calibration
- **Pydantic Validation**: Strong typing throughout with comprehensive input/output validation
- **Modular Architecture**: Each model can be enabled/disabled independently
- **Rich Metadata**: Model agreement scores, confidence ranges, and pipeline stage tracking
- **Plugin System**: Extensible adapter pattern allows adding new models without touching core code
- **Structured Telemetry**: Comprehensive performance tracking with per-stage and per-model metrics
- **Knowledge Base Versioning**: Index versioning for reproducibility and compatibility
- **Golden Fixtures**: Regression testing framework for backwards compatibility

## Performance Considerations

- **First Run**: Models are downloaded from Hugging Face (may take several minutes)
- **GPU Recommended**: For production use, GPU acceleration significantly improves speed
- **Memory Usage**: All 4 models enabled requires ~4-8GB GPU memory or ~16GB RAM on CPU
- **Batch Processing**: Use batch processing for multiple documents to amortize model loading overhead

### Optimization Tips

```python
# For faster inference with lower memory
config = SystemConfig(
    # Use smaller models
    retriever_model="sentence-transformers/all-MiniLM-L6-v2",
    reader_model="distilbert-base-uncased",

    # Reduce model count
    use_improved_atg=True,
    use_relik=False,  # Disable some models
    use_spel=False,
    use_unirel=False,

    # Enable optimizations
    gradient_checkpointing=True,
    quantization="int8",  # or "fp16"
)
```

## Troubleshooting

### Common Issues

**Issue**: Models downloading slowly
**Solution**: Models are cached in `~/.cache/huggingface/`. First run may be slow.

**Issue**: Out of memory errors
**Solution**: Reduce `batch_size`, disable some models, or use quantization.

**Issue**: FAISS installation issues
**Solution**: Ensure you have `faiss-cpu` installed: `uv add faiss-cpu`

**Issue**: Test failures
**Solution**: Run `uv run pytest -m "not e2e"` to skip slow E2E tests during development.

## License

GNU General Public License v3.0

## Contributing

Contributions are welcome! Please ensure:

- All tests pass: `uv run pytest`
- Code is formatted: `uv run ruff format .`
- Linting passes: `uv run ruff check .`

## Citation

If you use this system in your research, please cite the relevant papers and this implementation.

## Acknowledgments

This project implements and builds upon techniques from multiple research papers. See Research Background section for details.
