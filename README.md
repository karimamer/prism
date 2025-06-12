# Musnad: Entity Resolution System

A unified entity resolution system that combines techniques from state-of-the-art research papers including ReLiK, SpEL, UniRel, ATG, and OneNet to efficiently identify and link entities within text.

## Features

- **Unified Architecture**: Integrates retriever-reader-consensus pipeline for comprehensive entity resolution
- **Dense Retrieval**: Bi-encoder architecture with FAISS indexing for efficient candidate retrieval
- **Conflict Resolution**: Consensus mechanism to handle overlapping entity mentions
- **Flexible Input/Output**: Supports JSON, CSV, and text formats
- **Configurable Models**: Support for different transformer models (DeBERTa, etc.)
- **Vector Database**: FAISS-based entity knowledge base for fast similarity search
- **Batch Processing**: Efficient processing of multiple documents

## Architecture

The system consists of three main components:

### 1. Entity Retriever (`EntityRetriever`)
- Dense bi-encoder architecture for candidate entity retrieval
- FAISS vector database integration for fast similarity search
- Configurable top-k candidate selection
- Shared or separate encoders for text and entities

### 2. Entity Reader (`EntityReader`)
- Processes input text with candidate entities in single forward pass
- Supports mention detection and entity linking
- Based on SpEL and ReLiK techniques
- Configurable sequence length and entity limits

### 3. Consensus Module (`ConsensusModule`)
- Resolves conflicts between overlapping entity mentions
- Confidence calibration from multiple resolution methods
- Threshold-based filtering for high-quality predictions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd musnad

# Install dependencies
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Process a text file with entity resolution
python -m src.entity_resolution.run_entity_resolution \
    --input test_input.txt \
    --output results.json \
    --entities sample_entities.json \
    --format json
```

### Python API

```python
from src.entity_resolution.unified_system import UnifiedEntityResolutionSystem

# Initialize the system
config = {
    "retriever_model": "microsoft/deberta-v3-small",
    "reader_model": "microsoft/deberta-v3-base",
    "top_k_candidates": 50,
    "consensus_threshold": 0.6
}

system = UnifiedEntityResolutionSystem(config)

# Load entity knowledge base
system.load_entities("sample_entities.json")

# Process text
result = system.process_text("Apple Inc. was founded by Steve Jobs in 1976.")
print(result)
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

```json
{
  "text": "Apple Inc. was founded by Steve Jobs in 1976.",
  "entities": [
    {
      "mention": "Apple Inc.",
      "mention_span": [0, 9],
      "entity_id": "Q312",
      "entity_name": "Apple Inc.",
      "entity_type": "ORGANIZATION",
      "confidence": 0.92
    }
  ]
}
```

## Configuration

Key configuration parameters:

```python
config = {
    "retriever_model": "microsoft/deberta-v3-small",  # Retriever model
    "reader_model": "microsoft/deberta-v3-base",      # Reader model
    "entity_dim": 256,                                # Entity embedding dimension
    "max_seq_length": 512,                            # Maximum sequence length
    "top_k_candidates": 50,                           # Candidates to retrieve
    "consensus_threshold": 0.6,                       # Confidence threshold
    "batch_size": 8,                                  # Processing batch size
    "use_gpu": True,                                  # GPU acceleration
    "quantization": None                              # Optional quantization
}
```

## Command Line Interface

```bash
python -m src.entity_resolution.run_entity_resolution [OPTIONS]

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

## Project Structure

```
musnad/
├── src/
│   └── entity_resolution/
│       ├── models/
│       │   ├── retriever.py      # Dense retrieval component
│       │   ├── reader.py         # Entity reading/linking
│       │   ├── consensus.py      # Conflict resolution
│       │   └── output.py         # Output formatting
│       ├── database/
│       │   └── vector_store.py   # FAISS vector database
│       ├── unified_system.py     # Main system orchestrator
│       └── run_entity_resolution.py  # CLI interface
├── sample_entities.json          # Sample entity data
├── test_input.txt                # Sample input text
└── cache/                        # Cached entity data
```

## Dependencies

- PyTorch 2.6+
- Transformers 4.49+
- FAISS (CPU) 1.10+
- NumPy, Pandas, scikit-learn
- Other dependencies listed in `pyproject.toml`

## Research Background

This implementation integrates techniques from:

- **ReLiK**: Dense retrieval and entity linking
- **SpEL**: Structured prediction for entity linking
- **UniRel**: Unified representation learning
- **ATG**: Autoregressive text-to-graph generation
- **OneNet**: End-to-end entity resolution

## License

MIT License

## Citation
If you use this system in your research, please cite the relevant papers and this implementation.
