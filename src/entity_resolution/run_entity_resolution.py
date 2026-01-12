#!/usr/bin/env python3
"""
Enhanced Entity Resolution System based on ReLiK, SpEL, UniRel, ATG, and OneNet.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import torch

from entity_resolution.unified_system import UnifiedEntityResolutionSystem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Entity Resolution System")

    # Input/output options
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input file with text to process"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output file for resolved entities"
    )
    parser.add_argument(
        "--entities", "-e", type=str, default=None, help="File with entity data (JSON or CSV)"
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="json",
        choices=["json", "csv", "txt"],
        help="Output format",
    )

    # Model options
    parser.add_argument(
        "--model_path", "-m", type=str, default=None, help="Path to pretrained model"
    )
    parser.add_argument(
        "--retriever",
        "-r",
        type=str,
        default="microsoft/deberta-v3-small",
        help="Retriever model name",
    )
    parser.add_argument(
        "--reader", "-d", type=str, default="microsoft/deberta-v3-base", help="Reader model name"
    )
    parser.add_argument(
        "--quantization",
        "-q",
        type=str,
        default=None,
        choices=[None, "int8", "fp16"],
        help="Quantization type for faster inference",
    )

    # Processing options
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size for processing")
    parser.add_argument(
        "--top_k", "-k", type=int, default=50, help="Number of top entities to retrieve"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.6, help="Confidence threshold for entity linking"
    )
    parser.add_argument("--max_length", "-l", type=int, default=512, help="Maximum sequence length")

    # Other options
    parser.add_argument("--cache_dir", "-c", type=str, default="./cache", help="Cache directory")
    parser.add_argument(
        "--index_path", "-p", type=str, default="./entity_index", help="Entity index path"
    )
    parser.add_argument("--profile", action="store_true", help="Profile performance")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


def load_input_text(input_file):
    """Load input text from file"""
    with open(input_file, encoding="utf-8") as f:
        # Check if file is JSON
        if input_file.endswith(".json"):
            try:
                data = json.load(f)

                # Handle different JSON formats
                if isinstance(data, list):
                    if all(isinstance(item, str) for item in data):
                        # List of strings
                        texts = data
                    elif all(isinstance(item, dict) for item in data):
                        # List of dictionaries
                        texts = [item.get("text", "") for item in data]
                    else:
                        # Unknown format
                        raise ValueError("Unsupported JSON format")
                elif isinstance(data, dict):
                    # Dictionary
                    if "texts" in data and isinstance(data["texts"], list):
                        # List of texts
                        texts = data["texts"]
                    elif "text" in data:
                        # Single text
                        texts = [data["text"]]
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


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def save_output(results, output_file, format="json"):
    """Save output results to file"""
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "json":
            # JSON format - convert Pydantic objects to dictionaries
            serializable_results = []
            for result in results:
                if hasattr(result, "model_dump"):
                    # Pydantic v2 method
                    serializable_results.append(result.model_dump())
                elif hasattr(result, "dict"):
                    # Pydantic v1 method
                    serializable_results.append(result.dict())
                elif hasattr(result, "to_dict"):
                    # Custom method
                    serializable_results.append(result.to_dict())
                else:
                    # Fallback to regular dict conversion
                    serializable_results.append(dict(result))
            json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
        elif format == "csv":
            # CSV format - use new to_csv_rows() method
            import csv

            # Collect all CSV rows from all results
            all_rows = []
            for result in results:
                if hasattr(result, "to_csv_rows"):
                    # Use the new Pydantic method
                    all_rows.extend(result.to_csv_rows())
                else:
                    # Fallback for non-Pydantic results
                    text = result.get("text", "")
                    for entity in result.get("entities", []):
                        row = {
                            "text": text,
                            "mention": entity.get("mention", ""),
                            "mention_start": entity.get("mention_span", [0])[0] if isinstance(entity.get("mention_span"), list) else 0,
                            "mention_end": entity.get("mention_span", [0, 0])[1] if isinstance(entity.get("mention_span"), list) else 0,
                            "entity_id": entity.get("entity_id", ""),
                            "entity_name": entity.get("entity_name", ""),
                            "entity_type": entity.get("entity_type", ""),
                            "confidence": entity.get("confidence", 0.0),
                            "source_model": entity.get("source_model", ""),
                            "agreement_score": "",
                            "agreeing_models": "",
                            "total_models": "",
                        }
                        all_rows.append(row)

            if all_rows:
                # Write CSV with all fields
                fieldnames = [
                    "text",
                    "mention",
                    "mention_start",
                    "mention_end",
                    "entity_id",
                    "entity_name",
                    "entity_type",
                    "confidence",
                    "source_model",
                    "agreement_score",
                    "agreeing_models",
                    "total_models",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        else:
            # Text format
            for result in results:
                text = result.text if hasattr(result, "text") else result.get("text", "")
                entities = result.entities if hasattr(result, "entities") else result.get("entities", [])

                f.write(f"TEXT: {text}\n")
                f.write("ENTITIES:\n")

                for entity in entities:
                    mention = entity.mention if hasattr(entity, "mention") else entity.get("mention", "")
                    entity_name = entity.entity_name if hasattr(entity, "entity_name") else entity.get("entity_name", "")
                    entity_type = entity.entity_type.value if hasattr(entity, "entity_type") and hasattr(entity.entity_type, "value") else entity.get("entity_type", "")
                    confidence = entity.confidence if hasattr(entity, "confidence") else entity.get("confidence", 0.0)

                    f.write(
                        f"  - {mention} ({entity_name}, "
                        f"{entity_type}) [{confidence:.2f}]\n"
                    )
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
        "quantization": args.quantization,
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
        stats = pstats.Stats(profile).sort_stats("cumtime")
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
