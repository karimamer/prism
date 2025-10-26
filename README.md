# PRISM: Entity Resolution System

**⚠️ Prism is currently in its early experimental phase and should be considered unstable.**

A unified entity resolution system that combines state-of-the-art techniques from multiple research papers (ReLiK, SpEL, UniRel, ATG, and OneNet) to efficiently identify, classify, and link entities within text to a knowledge base.

## Features

- **Multi-Model Ensemble**: Integrates 4 state-of-the-art entity resolution models (ATG, ReLiK, SPEL, UniREL)
- **Unified Pipeline**: Complete retriever-reader-consensus architecture for comprehensive entity resolution
- **Dense Retrieval**: Bi-encoder architecture with FAISS indexing for efficient candidate retrieval
- **Consensus Resolution**: Multi-method consensus mechanism to combine predictions and handle overlapping mentions
- **Pydantic Validation**: Strong typing and validation using Pydantic v2 for configuration and outputs
- **Flexible I/O**: Supports JSON, CSV, and text formats with structured Pydantic models
- **Knowledge Base Integration**: FAISS-based vector database for fast entity similarity search
- **Batch Processing**: Efficient processing of multiple documents with configurable batch sizes
- **Comprehensive Testing**: Full E2E test suite with real models (30 tests covering all functionality)

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
      "confidence_avg": 0.90,
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
  "processing_timestamp": "2025-10-26T10:30:45.123456"
}
```

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
- **Unit Tests**: Individual model components (ATG, ReLiK, SPEL, UniREL)
- **Integration Tests**: Full pipeline integration
- **E2E Tests**: 30 comprehensive end-to-end tests with real models
  - System initialization and model loading
  - Single text and batch processing
  - Knowledge base operations
  - Error handling and edge cases
  - Model ensemble behavior
  - Performance characteristics
  - Output serialization

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
│       │   ├── atg.py                  # ATG model implementation
│       │   ├── relik/
│       │   │   ├── model.py            # ReLiK model
│       │   │   ├── retriever.py        # ReLiK retriever
│       │   │   ├── reader.py           # ReLiK reader
│       │   │   └── config.py           # ReLiK configuration
│       │   ├── spel/
│       │   │   ├── model.py            # SPEL model
│       │   │   ├── candidate_sets.py   # SPEL candidate management
│       │   │   ├── aggregation.py      # SPEL aggregation
│       │   │   └── config.py           # SPEL configuration
│       │   ├── unirel/
│       │   │   ├── model.py            # UniREL model
│       │   │   ├── interaction_map.py  # Interaction maps
│       │   │   └── config.py           # UniREL configuration
│       │   ├── retriever.py            # Base retriever
│       │   ├── reader.py               # Base reader
│       │   ├── consensus.py            # Consensus module
│       │   ├── output.py               # Pydantic output models
│       │   └── entity_encoder.py       # Entity-focused encoder
│       ├── database/
│       │   └── vector_store.py         # FAISS knowledge base
│       ├── unified_system.py           # Main pipeline orchestrator
│       ├── validation.py               # Pydantic config & validation
│       └── run_entity_resolution.py    # CLI interface
├── tests/
│   ├── unit/
│   │   ├── test_atg.py                 # ATG unit tests
│   │   ├── test_relik.py               # ReLiK unit tests
│   │   ├── test_spel.py                # SPEL unit tests
│   │   └── test_unirel.py              # UniREL unit tests
│   └── integration/
│       └── test_e2e_pipeline.py        # 30 E2E tests
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
