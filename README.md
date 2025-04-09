# Unified Entity Resolution System

A modular and flexible entity resolution system that combines state-of-the-art approaches to effectively identify and link entities within text to a knowledge base.

## Overview

This system integrates multiple innovative approaches from recent research in entity resolution:

- **ReLiK**: Retrieval-based entity linking with efficient architecture
- **ATG**: Autoregressive Text-to-Graph for structured prediction
- **SpEL**: Structured prediction for entity span identification
- **UniRel**: Unified representation for entity-relation modeling
- **OneNet**: End-to-end architecture for joint extraction

The system is built on a Retriever-Reader architecture, combining the strengths of dense retrieval and contextual encoding to achieve high-quality entity resolution while maintaining computational efficiency.

## Features

- **Unified Architecture**: Combines multiple state-of-the-art techniques in a single framework
- **Flexible Entity Knowledge Base**: Compatible with DuckDB for efficient vector search
- **Modular Components**: Easy to extend or replace individual system components
- **Multi-Method Consensus**: Uses multiple resolution strategies to improve accuracy
- **Efficient Processing**: Optimized for both training and inference performance
- **Gradient Checkpointing**: Support for large language models with memory optimization

## Architecture

The system is structured into the following key components:

### UnifiedEntityResolutionSystem
The main entry point that orchestrates the entity resolution process.

### EntityFocusedEncoder
Utilizes DeBERTa-v3 with entity-specific enhancements to create contextual embeddings.

### EntityCandidateGenerator
Identifies potential entity mentions and retrieves candidates from the knowledge base.

### EntityResolutionProcessor
Processes candidate entities using multiple resolution strategies.

### EntityConsensusModule
Resolves conflicts and reaches consensus on entity predictions.

### EntityOutputFormatter
Formats entity output for downstream applications.

### DuckDBKnowledgeBase
Provides efficient vector storage and retrieval for entity information.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/entity-resolution.git
cd entity-resolution

# Create and activate a virtual environment
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.entity_resolution import UnifiedEntityResolutionSystem

# Initialize the system
system = UnifiedEntityResolutionSystem()

# Process a text document
result = system.process_text("Apple Inc. was founded by Steve Jobs in Cupertino, California.")

# Print resolved entities
for entity in result["entities"]:
    print(f"Entity: {entity['entity_name']} ({entity['entity_id']})")
    print(f"Mention: {entity['mention']} at positions {entity['mention_span']}")
    print(f"Confidence: {entity['confidence']}")
    print()
```

### Command Line Interface

The system can also be used via command line:

```bash
python -m src.entity_resolution.run_entity_resolution \
    --input_file input.txt \
    --output_file output.txt \
    --model_path /path/to/model \
    --encoder microsoft/deberta-v3-base \
    --kb_path entity_kb.duckdb
```

### Training

```python
from src.entity_resolution import UnifiedEntityResolutionSystem

# Initialize the system
system = UnifiedEntityResolutionSystem()

# Train the system
system.train_system(
    train_datasets=train_data,
    val_datasets=val_data,
    num_epochs=5
)

# Save the trained model
system.save_model("entity_resolution_model.pt")
```

## Training Data Format

The training data should be provided as PyTorch DataLoader objects with the following format:

```python
{
    "input_ids": torch.LongTensor,      # Tokenized input text
    "attention_mask": torch.LongTensor, # Attention mask for input text
    "labels": List[Dict]                # Ground truth entity annotations
}
```

Each label in the list should contain:
- `mention_span`: Tuple of (start, end) token positions
- `entity_id`: Knowledge base ID of the entity
- `entity_name`: Human-readable name of the entity
- `entity_type`: Type of the entity (e.g., person, organization)

## Knowledge Base Setup

The system can work with an in-memory knowledge base for testing, but for production use, it's recommended to use DuckDB:

```python
from src.entity_resolution.database.db import DuckDBKnowledgeBase

# Initialize knowledge base
kb = DuckDBKnowledgeBase("entity_kb.duckdb")

# Load entities
kb.load_entities(entities_data)

# Generate and add embeddings
entity_embeddings = generate_entity_embeddings(entities_data)
kb.add_embeddings(entity_embeddings)
```

## Configuration Options

The system accepts the following configuration options:

- `encoder_name`: Name of the pretrained encoder (default: "microsoft/deberta-v3-base")
- `encoder_dim`: Hidden size of the encoder (default: 768 for base models)
- `entity_knowledge_dim`: Dimension for entity knowledge representation (default: 256)
- `max_seq_length`: Maximum input sequence length (default: 512)
- `num_entity_types`: Number of entity types (default: 50)
- `consensus_threshold`: Confidence threshold for entity linking (default: 0.6)
- `top_k_candidates`: Number of candidates to retrieve per mention (default: 50)

## Dependencies

- PyTorch >= 1.10.0
- Transformers >= 4.18.0
- DuckDB >= 0.3.4

## License

[MIT License](LICENSE)

## Acknowledgments

This project builds upon several research contributions in the field of entity resolution:
- ReLiK (Retrieve and Link)
- ATG (Autoregressive Text-to-Graph)
- SpEL (Structured Prediction for Entity Linking)
- UniRel (Unified Representation and Interaction for Joint Relational Triple Extraction)
- OneNet (Joint Entity and Relation Extraction with One Module in One Step)
