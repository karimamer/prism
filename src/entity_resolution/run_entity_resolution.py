#!/usr/bin/env python3
"""
Enhanced Entity Resolution System based on ReLiK, SpEL, UniRel, ATG, and OneNet.
"""

import os
import sys
import argparse
import json
import logging
import time
import torch

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Import system components
from src.entity_resolution.unified_system import UnifiedEntityResolutionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Entity Resolution System"
    )

    # Input/output options
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input file with text to process")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output file for resolved entities")
    parser.add_argument("--entities", "-e", type=str, default=None,
                       help="File with entity data (JSON or CSV)")
    parser.add_argument("--format", "-f", type=str, default="json",
                       choices=["json", "csv", "txt"],
                       help="Output format")

    # Model options
    parser.add_argument("--model_path", "-m", type=str, default=None,
                       help="Path to pretrained model")
    parser.add_argument("--retriever", "-r", type=str,
                       default="microsoft/deberta-v3-small",
                       help="Retriever model name")
    parser.add_argument("--reader", "-d", type=str,
                       default="microsoft/deberta-v3-base",
                       help="Reader model name")
    parser.add_argument("--quantization", "-q", type=str,
                       default=None, choices=[None, "int8", "fp16"],
                       help="Quantization type for faster inference")

    # Processing options
    parser.add_argument("--batch_size", "-b", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--top_k", "-k", type=int, default=50,
                       help="Number of top entities to retrieve")
    parser.add_argument("--threshold", "-t", type=float, default=0.6,
                       help="Confidence threshold for entity linking")
    parser.add_argument("--max_length", "-l", type=int, default=512,
                       help="Maximum sequence length")

    # Other options
    parser.add_argument("--cache_dir", "-c", type=str, default="./cache",
                       help="Cache directory")
    parser.add_argument("--index_path", "-p", type=str, default="./entity_index",
                       help="Entity index path")
    parser.add_argument("--profile", action="store_true",
                       help="Profile performance")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    return parser.parse_args()

def load_input_text(input_file):
    """Load input text from file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        # Check if file is JSON
        if input_file.endswith('.json'):
            try:
                data = json.load(f)

                # Handle different JSON formats
                if isinstance(data, list):
                    if all(isinstance(item, str) for item in data):
                        # List of strings
                        texts = data
                    elif all(isinstance(item, dict) for item in data):
                        # List of dictionaries
                        texts = [item.get('text', '') for item in data]
                    else:
                        # Unknown format
                        raise ValueError("Unsupported JSON format")
                elif isinstance(data, dict):
                    # Dictionary
                    if 'texts' in data and isinstance(data['texts'], list):
                        # List of texts
                        texts = data['texts']
                    elif 'text' in data:
                        # Single text
                        texts = [data['text']]
                    else:
                        # Unknown format
                        raise ValueError("Unsupported JSON format")
                else:
                    # Unknown format
                    raise ValueError("Unsupported JSON format")

                return texts
            except json.JSONDecodeError:
                # Not a valid JSON file, treat as plain text
                pass

        # Plain text file
        lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

def save_output(results, output_file, format="json"):
    """Save output results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        if format == "json":
            # JSON format
            json.dump(results, f, indent=2)
        elif format == "csv":
            # CSV format
            import csv

            # Extract all entities
            all_entities = []
            for result in results:
                text = result['text']
                for entity in result['entities']:
                    entity_copy = entity.copy()
                    entity_copy['text'] = text
                    all_entities.append(entity_copy)

            # Write CSV header
            writer = csv.DictWriter(f, fieldnames=[
                'text', 'mention', 'entity_id', 'entity_name',
                'entity_type', 'confidence'
            ])
            writer.writeheader()

            # Write entities
            for entity in all_entities:
                writer.writerow({
                    'text': entity['text'],
                    'mention': entity['mention'],
                    'entity_id': entity['entity_id'],
                    'entity_name': entity['entity_name'],
                    'entity_type': entity['entity_type'],
                    'confidence': entity['confidence']
                })
        else:
            # Text format
            for result in results:
                f.write(f"TEXT: {result['text']}\n")
                f.write("ENTITIES:\n")

                for entity in result['entities']:
                    f.write(f"  - {entity['mention']} ({entity['entity_name']}, "
                           f"{entity['entity_type']}) [{entity['confidence']:.2f}]\n")
                f.write("\n")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting enhanced entity resolution system")

    # Create configuration
    config = {
        "retriever_model": args.retriever,
        "reader_model": args.reader,
        "entity_dim": 256,
        "max_seq_length": args.max_length,
        "max_entity_length": 100,
        "top_k_candidates": args.top_k,
        "consensus_threshold": args.threshold,
        "batch_size": args.batch_size,
        "index_path": args.index_path,
        "cache_dir": args.cache_dir,
        "use_gpu": torch.cuda.is_available(),
        "quantization": args.quantization
    }

    # Initialize system
    start_time = time.time()

    if args.model_path and os.path.exists(args.model_path):
        # Load pretrained model
        logger.info(f"Loading pretrained model from {args.model_path}")
        system = UnifiedEntityResolutionSystem.load(args.model_path)
    else:
        # Create new model
        logger.info("Creating new model")
        system = UnifiedEntityResolutionSystem(config)

    # Load entity data if provided
    if args.entities and os.path.exists(args.entities):
        logger.info(f"Loading entities from {args.entities}")
        num_entities = system.load_entities(args.entities)
        logger.info(f"Loaded {num_entities} entities")

    # Load input text
    logger.info(f"Loading input from {args.input}")
    texts = load_input_text(args.input)
    logger.info(f"Loaded {len(texts)} texts")

    # Process texts
    logger.info("Processing texts")
    if args.profile:
        # Profile performance
        import cProfile
        import pstats

        # Run with profiling
        profile = cProfile.Profile()
        profile.enable()

    # Process texts
    results = system.process_batch(texts)

    if args.profile:
        # Print profiling results
        profile.disable()
        stats = pstats.Stats(profile).sort_stats('cumtime')
        stats.print_stats(20)

    # Save output
    logger.info(f"Saving output to {args.output}")
    save_output(results, args.output, args.format)

    # Print performance stats
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(texts)

    logger.info(f"Processed {len(texts)} texts in {total_time:.2f} seconds")
    logger.info(f"Average processing time: {avg_time:.2f} seconds per text")

    # Print GPU memory usage if available
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1024**2
        logger.info(f"Peak GPU memory usage: {max_memory:.2f} MB")

    logger.info("Entity resolution completed successfully")


if __name__ == "__main__":
    main()
