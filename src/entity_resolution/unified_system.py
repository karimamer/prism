"""
Unified Entity Resolution System with comprehensive input validation.

This module integrates state-of-the-art techniques from ReLiK, SpEL, ATG,
UniRel, and OneNet, with full input validation, error handling, and safety checks.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from src.entity_resolution.database.vector_store import EntityKnowledgeBase
from src.entity_resolution.models.candidate_generator import EntityCandidateGenerator
from src.entity_resolution.models.consensus import ConsensusModule
from src.entity_resolution.models.entity_encoder import EntityFocusedEncoder
from src.entity_resolution.models.reader import EntityReader
from src.entity_resolution.models.resolution_processor import EntityResolutionProcessor
from src.entity_resolution.models.retriever import EntityRetriever
from src.entity_resolution.validation import (
    InputValidator,
    SystemConfig,
    validate_and_load_entities,
    validate_config,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedEntityResolutionSystem(nn.Module):
    """
    Unified entity resolution system with comprehensive validation.

    This system integrates state-of-the-art techniques from ReLiK, SpEL, ATG,
    UniRel, and OneNet, with full input validation and error handling.

    Features:
    - Pydantic-based configuration validation
    - Input text validation
    - Entity data validation
    - File size checks
    - Type safety
    - Clear error messages
    """

    def __init__(self, config: Union[dict[str, Any], SystemConfig, None] = None):
        """
        Initialize the entity resolution system with validation.

        Args:
            config: Configuration dictionary or SystemConfig object

        Raises:
            ValidationError: If configuration is invalid
        """
        super().__init__()

        # Validate and normalize configuration
        if config is None:
            logger.info("No configuration provided, using defaults")
            self.config = SystemConfig()
        else:
            self.config = validate_config(config)

        logger.info("Configuration validated successfully")

        # Set device with validation
        self.device = self._setup_device()

        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)

        # Initialize knowledge base
        self.knowledge_base = self._initialize_knowledge_base()

        # Initialize components
        self._initialize_components()

        # Apply quantization if specified
        if self.config.quantization:
            self._apply_quantization()

        # Initialize cache for entity embeddings
        self.entity_embedding_cache = {}

        logger.info("Entity resolution system initialized successfully")

    def _setup_device(self) -> torch.device:
        """Setup computation device with validation."""
        if self.config.use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

                # Log GPU memory
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU memory: {mem:.2f} GB")
            else:
                device = torch.device("cpu")
                logger.warning("GPU requested but not available, using CPU")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")

        return device

    def _initialize_knowledge_base(self) -> EntityKnowledgeBase:
        """Initialize the entity knowledge base."""
        logger.info("Initializing entity knowledge base")

        try:
            kb = EntityKnowledgeBase(
                index_path=self.config.index_path, cache_dir=self.config.cache_dir
            )
            return kb
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise

    def _initialize_components(self) -> None:
        """Initialize all model components with validation."""
        # Initialize entity-focused encoder (new component)
        if self.config.use_entity_encoder:
            logger.info("Initializing EntityFocusedEncoder")
            try:
                self.entity_encoder = EntityFocusedEncoder(
                    pretrained_model_name=self.config.reader_model,
                    entity_knowledge_dim=self.config.entity_encoder_dim,
                    num_entity_types=self.config.num_entity_types,
                    dropout=self.config.dropout,
                )
                self.entity_encoder.to(self.device)
                logger.info("EntityFocusedEncoder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize entity encoder: {e}")
                raise RuntimeError(f"Entity encoder initialization failed: {e}") from e
        else:
            self.entity_encoder = None

        # Initialize candidate generator (new component)
        if self.config.use_candidate_generator:
            logger.info("Initializing EntityCandidateGenerator")
            try:
                # Get embedding dimension from reader or encoder
                if self.entity_encoder is not None:
                    embedding_dim = self.entity_encoder.hidden_size
                else:
                    embedding_dim = 768  # Default

                self.candidate_generator = EntityCandidateGenerator(
                    embedding_dim=embedding_dim,
                    knowledge_base=self.knowledge_base,
                    dropout=self.config.dropout,
                )
                self.candidate_generator.to(self.device)
                logger.info("EntityCandidateGenerator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize candidate generator: {e}")
                raise RuntimeError(f"Candidate generator initialization failed: {e}") from e
        else:
            self.candidate_generator = None

        # Initialize resolution processor (new component)
        if self.config.use_resolution_processor:
            logger.info("Initializing EntityResolutionProcessor")
            try:
                # Get encoder dimension
                if self.entity_encoder is not None:
                    encoder_dim = self.entity_encoder.hidden_size
                else:
                    encoder_dim = 768  # Default

                self.resolution_processor = EntityResolutionProcessor(
                    encoder_dim=encoder_dim,
                    num_heads=self.config.num_attention_heads,
                    dropout=self.config.dropout,
                )
                self.resolution_processor.to(self.device)
                logger.info("EntityResolutionProcessor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize resolution processor: {e}")
                raise RuntimeError(f"Resolution processor initialization failed: {e}") from e
        else:
            self.resolution_processor = None

        # Initialize retriever
        logger.info(f"Initializing retriever: {self.config.retriever_model}")
        try:
            self.retriever = EntityRetriever(
                model_name=self.config.retriever_model,
                use_faiss=True,
                top_k=self.config.top_k_candidates,
            )
            self.retriever.to(self.device)
            logger.info("Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise RuntimeError(f"Retriever initialization failed: {e}") from e

        # Initialize reader
        logger.info(f"Initializing reader: {self.config.reader_model}")
        try:
            self.reader = EntityReader(
                model_name=self.config.reader_model,
                max_seq_length=self.config.max_seq_length,
                max_entity_length=self.config.max_entity_length,
                gradient_checkpointing=self.config.gradient_checkpointing,
            )
            self.reader.to(self.device)
            logger.info("Reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize reader: {e}")
            raise RuntimeError(f"Reader initialization failed: {e}") from e

        # Initialize consensus module
        logger.info("Initializing consensus module")
        try:
            self.consensus = ConsensusModule(
                hidden_size=self.reader.config.hidden_size,
                threshold=self.config.consensus_threshold,
            )
            self.consensus.to(self.device)
            logger.info("Consensus module initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize consensus: {e}")
            raise RuntimeError(f"Consensus initialization failed: {e}") from e

    def _apply_quantization(self) -> None:
        """Apply quantization to models for faster inference."""
        logger.info(f"Applying {self.config.quantization} quantization")

        try:
            if self.config.quantization == "int8":
                # Quantize retriever
                import torch.quantization

                self.retriever.eval()
                self.retriever = torch.quantization.quantize_dynamic(
                    self.retriever, {nn.Linear}, dtype=torch.qint8
                )

                # Quantize reader
                self.reader.quantize("int8")

            elif self.config.quantization == "fp16":
                # Convert to half precision
                self.retriever = self.retriever.half()
                self.reader.quantize("fp16")
                self.consensus = self.consensus.half()

            logger.info("Quantization applied successfully")
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise RuntimeError(f"Failed to apply quantization: {e}") from e

    def load_entities(
        self, entity_file: Union[str, Path], max_entities: Optional[int] = None
    ) -> int:
        """
        Load entities from a file with validation.

        Args:
            entity_file: Path to entity file (JSON or CSV)
            max_entities: Maximum number of entities to load (optional)

        Returns:
            Number of entities loaded

        Raises:
            FileNotFoundError: If entity file doesn't exist
            ValueError: If entity data is invalid
        """
        logger.info(f"Loading entities from {entity_file}")

        try:
            # Validate and load entities
            entities = validate_and_load_entities(entity_file, max_entities)

            # Convert to knowledge base format
            entity_dict = {entity["id"]: entity for entity in entities}

            # Build retriever index
            logger.info("Building retriever index")
            self.retriever.build_index(entity_dict)

            logger.info(f"Successfully loaded {len(entities)} entities")
            return len(entities)

        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            raise

    def process_text(self, text: str, validate_input: bool = True) -> dict[str, Any]:
        """
        Process a text document for entity resolution with validation.

        Args:
            text: Input text
            validate_input: Whether to validate input text (default: True)

        Returns:
            Dictionary with resolved entities

        Raises:
            ValueError: If input text is invalid
            RuntimeError: If processing fails
        """
        # Validate input text
        if validate_input:
            try:
                text = InputValidator.validate_text_input(
                    text,
                    max_length=self.config.max_seq_length * 10,  # Allow longer input
                )
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid input text: {e}")
                raise

        try:
            # Tokenize text for retriever
            tokenizer = self.retriever.tokenizer
            encoded_text = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
            ).to(self.device)

            # Step 1: Retrieve candidate entities
            logger.debug("Retrieving candidate entities")
            with torch.no_grad():
                candidates = self.retriever.retrieve(
                    encoded_text["input_ids"],
                    encoded_text["attention_mask"],
                    top_k=self.config.top_k_candidates,
                )

            # Format candidates for reader
            candidate_entities = []
            for entity_id, entity_data, score in candidates:
                candidate_entities.append(
                    {
                        "id": entity_id,
                        "name": entity_data.get("name", ""),
                        "description": entity_data.get("description", ""),
                        "type": entity_data.get("type", "UNKNOWN"),
                        "score": float(score),
                    }
                )

            logger.debug(f"Retrieved {len(candidate_entities)} candidates")

            # Step 2: Process with reader
            logger.debug("Processing with reader")
            reader_results = self.reader.process_text(text, candidate_entities)

            # Step 3: Apply consensus
            logger.debug("Applying consensus")
            consensus_results = self.consensus.resolve_entities(reader_results["entities"], text)

            # Format final results
            result = {
                "text": text,
                "entities": consensus_results,
                "num_entities": len(consensus_results),
                "num_candidates": len(candidate_entities),
            }

            logger.debug(f"Resolved {len(consensus_results)} entities")
            return result

        except Exception as e:
            logger.error(f"Failed to process text: {e}")
            raise RuntimeError(f"Text processing failed: {e}") from e

    def process_batch(self, texts: list[str], validate_input: bool = True) -> list[dict[str, Any]]:
        """
        Process a batch of texts with validation.

        Args:
            texts: List of input texts
            validate_input: Whether to validate input texts (default: True)

        Returns:
            List of dictionaries with resolved entities

        Raises:
            ValueError: If batch is invalid
            RuntimeError: If processing fails
        """
        # Validate batch
        if validate_input:
            try:
                texts = InputValidator.validate_batch_texts(
                    texts, max_batch_size=self.config.batch_size * 10
                )
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid batch: {e}")
                raise

        logger.info(f"Processing batch of {len(texts)} texts")

        results = []
        batch_size = self.config.batch_size

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches}")

                # Process each text in the batch
                for text in batch:
                    result = self.process_text(text, validate_input=False)
                    results.append(result)

            logger.info(f"Successfully processed {len(results)} texts")
            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise RuntimeError(f"Failed to process batch: {e}") from e

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the entity resolution system with validation.

        Args:
            path: Path to save the model

        Raises:
            OSError: If save fails
        """
        path = Path(path)
        logger.info(f"Saving model to {path}")

        try:
            # Create directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)

            # Save components
            logger.debug("Saving retriever")
            self.retriever.save(str(path / "retriever"))

            logger.debug("Saving reader")
            self.reader.save(str(path / "reader"))

            logger.debug("Saving consensus module")
            torch.save(self.consensus.state_dict(), path / "consensus.pt")

            # Save configuration
            logger.debug("Saving configuration")
            self.config.save_json(path / "config.json")

            logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise OSError(f"Save failed: {e}") from e

    @classmethod
    def load(
        cls, path: Union[str, Path], validate_path: bool = True
    ) -> "UnifiedEntityResolutionSystem":
        """
        Load the entity resolution system with validation.

        Args:
            path: Path to load the model from
            validate_path: Whether to validate path exists (default: True)

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: If model files not found
            RuntimeError: If loading fails
        """
        path = Path(path)
        logger.info(f"Loading model from {path}")

        # Validate path
        if validate_path:
            if not path.exists():
                raise FileNotFoundError(f"Model path not found: {path}")

            config_path = path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            # Load configuration
            logger.debug("Loading configuration")
            config = SystemConfig.from_json(path / "config.json")

            # Create model instance
            logger.debug("Creating model instance")
            model = cls(config)

            # Load components
            logger.debug("Loading retriever")
            model.retriever = EntityRetriever.load(str(path / "retriever"))

            logger.debug("Loading reader")
            model.reader = EntityReader.load(str(path / "reader"))

            logger.debug("Loading consensus module")
            model.consensus.load_state_dict(torch.load(path / "consensus.pt"))

            # Move models to device
            model.retriever.to(model.device)
            model.reader.to(model.device)
            model.consensus.to(model.device)

            logger.info(f"Model loaded successfully from {path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Load failed: {e}") from e

    def get_system_info(self) -> dict[str, Any]:
        """
        Get system information for diagnostics.

        Returns:
            Dictionary with system information
        """
        info = {
            "device": str(self.device),
            "config": self.config.to_dict(),
            "models": {
                "retriever": self.config.retriever_model,
                "reader": self.config.reader_model,
            },
            "quantization": self.config.quantization,
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
            }

        return info


# Convenience function for creating system
def create_system(
    config: Optional[Union[dict[str, Any], SystemConfig]] = None,
) -> UnifiedEntityResolutionSystem:
    """
    Create an entity resolution system.

    Args:
        config: Configuration dictionary or SystemConfig object

    Returns:
        Initialized system

    Example:
        >>> system = create_system({"batch_size": 16})
        >>> system.load_entities("entities.json")
        >>> results = system.process_text("Apple Inc. was founded by Steve Jobs.")
    """
    return UnifiedEntityResolutionSystem(config)
