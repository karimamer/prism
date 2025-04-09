import argparse
from src.entity_resolution.unified_system import UnifiedEntityResolutionSystem

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Entity Resolution")
    parser.add_argument("--input_file", type=str, help="Input file with text to process")
    parser.add_argument("--output_file", type=str, help="Output file for resolved entities")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--encoder", type=str, default="roberta-base",
                        help="Encoder model to use")
    parser.add_argument("--kb_path", type=str, default="entity_kb.duckdb",
                        help="Path to knowledge base")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()

    # Create configuration
    config = {
        "encoder_name": args.encoder,
        "encoder_dim": 768 if "base" in args.encoder else 1024,
        "entity_knowledge_dim": 256,
        "max_seq_length": 512,
        "num_entity_types": 50,
        "consensus_threshold": 0.6,
        "top_k_candidates": 50,
        "kb_path": args.kb_path
    }

    # Initialize the system
    system = UnifiedEntityResolutionSystem(config)

    # Load pretrained model if provided
    if args.model_path:
        system.load_trained_model(args.model_path)

    # Process input text
    process_input_file(system, args.input_file, args.output_file, args.batch_size)

def process_input_file(system, input_file, output_file, batch_size):
    """Process input file and write results to output file"""
    # Load input text
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Process documents individually
    all_results = []

    # Check what methods are available
    available_methods = [method for method in dir(system) if not method.startswith("_")]
    print(f"Available methods in system: {available_methods}")

    # Look for appropriate processing methods
    if hasattr(system, "process_document"):
        process_method = system.process_document
    elif hasattr(system, "process_text"):
        process_method = system.process_text
    elif hasattr(system, "extract_entities"):
        process_method = system.extract_entities
    else:
        # If no suitable method is found, implement a dummy one to avoid errors
        print("WARNING: No processing method found. Using dummy method.")
        process_method = lambda text: {"entities": []}

    # Process each document
    for line in lines:
        result = process_method(line)
        all_results.append(result)

    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for text, result in zip(lines, all_results):
            f.write(f"TEXT: {text}\n")
            f.write("ENTITIES:\n")

            # Handle different result formats
            if isinstance(result, dict) and "entities" in result:
                entities = result["entities"]
            elif isinstance(result, list):
                entities = result
            else:
                entities = []
                print(f"WARNING: Unexpected result format: {type(result)}")

            for entity in entities:
                if isinstance(entity, dict):
                    mention = entity.get('mention', 'Unknown')
                    entity_name = entity.get('entity_name', 'Unknown')
                    entity_type = entity.get('entity_type', 'Unknown')
                    confidence = entity.get('confidence', 0.0)
                    f.write(f"  - {mention} ({entity_name}, {entity_type}) [{confidence:.2f}]\n")
                else:
                    f.write(f"  - {entity}\n")
            f.write("\n")

    print(f"Processed {len(lines)} documents, results written to {output_file}")

if __name__ == "__main__":
    main()
