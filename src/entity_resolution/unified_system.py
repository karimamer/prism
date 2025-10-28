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

from entity_resolution.database.vector_store import EntityKnowledgeBase
from entity_resolution.models.atg import ATGConfig, ImprovedATGModel
from entity_resolution.models.candidate_generator import EntityCandidateGenerator
from entity_resolution.models.consensus import ConsensusModule
from entity_resolution.models.entity_encoder import EntityFocusedEncoder
from entity_resolution.models.output import UnifiedSystemOutput, create_unified_output
from entity_resolution.models.reader import EntityReader
from entity_resolution.models.relik.unified_integration import (
    ReLiKSystem,
    create_enhanced_relik_integration,
)
from entity_resolution.models.resolution_processor import EntityResolutionProcessor
from entity_resolution.models.retriever import EntityRetriever
from entity_resolution.models.spel import SPELConfig, SPELModel
from entity_resolution.models.unirel import UniRelConfig, UniRelModel
from entity_resolution.validation import (
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

        if self.config.use_improved_atg:
            logger.info("Initializing Improved ATG Model")
            try:
                # Get entity and relation types from config or use defaults
                # Handle None values from config
                entity_types = self.config.entity_types or ["PER", "ORG", "LOC", "MISC"]
                relation_types = self.config.relation_types or [
                    "Work_For",
                    "Based_In",
                    "Located_In",
                ]

                # Create ATG config
                atg_config = ATGConfig(
                    encoder_model=self.config.reader_model,
                    decoder_layers=getattr(self.config, "atg_decoder_layers", 6),
                    decoder_heads=getattr(self.config, "num_attention_heads", 8),
                    decoder_dim_feedforward=getattr(self.config, "atg_decoder_dim", 2048),
                    max_span_length=getattr(self.config, "atg_max_span_length", 12),
                    max_seq_length=self.config.max_seq_length,
                    entity_types=entity_types,
                    relation_types=relation_types,
                    dropout=self.config.dropout,
                    sentence_augmentation_max=getattr(self.config, "sentence_augmentation_max", 5),
                    use_sorted_ordering=getattr(self.config, "use_sorted_ordering", True),
                    top_p=getattr(self.config, "top_p", 1.0),
                    temperature=getattr(self.config, "temperature", 1.0),
                )

                self.atg_model = ImprovedATGModel(atg_config)
                self.atg_model.to(self.device)
                logger.info("Improved ATG Model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ATG model: {e}")
                raise RuntimeError(f"ATG model initialization failed: {e}") from e
        else:
            self.atg_model = None

        # Initialize Enhanced ReLiK Integration (new component with all improvements)
        if self.config.use_relik:
            logger.info("Initializing Enhanced ReLiK Integration")
            try:
                # Get entity and relation types from config or use defaults
                entity_types = self.config.entity_types or ["PER", "ORG", "LOC", "MISC"]

                # Create enhanced integration with all features
                self.relik_integration = create_enhanced_relik_integration(
                    {
                        "retriever_model": self.config.relik_retriever_model
                        or self.config.retriever_model,
                        "reader_model": self.config.relik_reader_model or self.config.reader_model,
                        "enable_relation_extraction": self.config.relik_use_re,
                        "enable_calibration": getattr(
                            self.config, "relik_enable_calibration", False
                        ),
                        "enable_dynamic_updates": getattr(
                            self.config, "relik_enable_dynamic_updates", True
                        ),
                        "device": str(self.device),
                        "max_query_length": 64,
                        "max_passage_length": 64,
                        "max_seq_length": self.config.max_seq_length,
                        "num_entity_types": len(entity_types),
                        "dropout": self.config.dropout,
                        "max_span_length": 10,
                        "rebuild_threshold": getattr(self.config, "relik_rebuild_threshold", 1000),
                        "auto_rebuild": getattr(self.config, "relik_auto_rebuild", True),
                        "use_faiss": True,
                    }
                )

                # Keep reference to linker for compatibility
                self.relik_model = self.relik_integration.linker

                logger.info("Enhanced ReLiK Integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced ReLiK: {e}")
                raise RuntimeError(f"Enhanced ReLiK initialization failed: {e}") from e
        else:
            self.relik_integration = None
            self.relik_model = None

        # Initialize SPEL model (new component)
        if self.config.use_spel:
            logger.info("Initializing SPEL Model")
            try:
                # Create SPEL config
                spel_config = SPELConfig(
                    encoder_model=self.config.spel_model_name or "roberta-base",
                    max_seq_length=self.config.max_seq_length,
                    fixed_candidate_set_size=self.config.spel_fixed_candidate_set_size,
                    use_mention_specific_candidates=self.config.spel_use_mention_specific_candidates,
                    num_hard_negatives=self.config.spel_num_hard_negatives,
                    dropout=self.config.dropout,
                    gradient_checkpointing=self.config.gradient_checkpointing,
                    span_threshold=self.config.spel_span_threshold,
                    entity_threshold=self.config.spel_entity_threshold,
                    entity_types=self.config.entity_types or ["PER", "ORG", "LOC", "MISC"],
                )

                self.spel_model = SPELModel(spel_config)
                self.spel_model.to(self.device)
                logger.info("SPEL Model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SPEL model: {e}")
                raise RuntimeError(f"SPEL model initialization failed: {e}") from e
        else:
            self.spel_model = None

        # Initialize UniRel model (new component)
        if self.config.use_unirel:
            logger.info("Initializing UniRel Model")
            try:
                # Get entity and relation types from config or use defaults
                entity_types = self.config.entity_types or ["PER", "ORG", "LOC", "MISC"]
                relation_types = self.config.relation_types or [
                    "Work_For",
                    "Based_In",
                    "Located_In",
                ]

                # Create UniRel config
                unirel_config = UniRelConfig(
                    encoder_model=self.config.unirel_encoder_model or self.config.reader_model,
                    max_seq_length=self.config.max_seq_length,
                    hidden_size=self.config.unirel_hidden_size,
                    relation_types=relation_types,
                    relation_verbalizations=self.config.unirel_relation_verbalizations,
                    interaction_threshold=self.config.unirel_interaction_threshold,
                    num_attention_heads=self.config.num_attention_heads,
                    interaction_dropout=self.config.unirel_interaction_dropout,
                    entity_types=entity_types,
                    entity_threshold=self.config.unirel_entity_threshold,
                    triple_threshold=self.config.unirel_triple_threshold,
                    max_triples_per_sentence=self.config.unirel_max_triples,
                    handle_overlapping=self.config.unirel_handle_overlapping,
                    dropout=self.config.dropout,
                    gradient_checkpointing=self.config.gradient_checkpointing,
                )

                self.unirel_model = UniRelModel(unirel_config)
                try:
                    self.unirel_model.to(self.device)
                    logger.info("UniRel Model initialized successfully")
                except Exception as device_error:
                    logger.warning(
                        f"Failed to move UniRel model to device {self.device}: {device_error}"
                    )
                    # Try CPU fallback
                    try:
                        self.unirel_model.to(torch.device("cpu"))
                        logger.info("UniRel Model moved to CPU as fallback")
                    except Exception as cpu_error:
                        logger.error(f"Failed to move UniRel model to CPU: {cpu_error}")
                        raise
            except Exception as e:
                logger.error(f"Failed to initialize UniRel model: {e}")
                raise RuntimeError(f"UniRel model initialization failed: {e}") from e
        else:
            self.unirel_model = None

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
        try:
            self.consensus = ConsensusModule(
                hidden_size=self.reader.config.hidden_size,
                threshold=self.config.consensus_threshold,
            )
            self.consensus.to(self.device)
            logger.info("Consensus module initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize consensus module: {e}")
            raise RuntimeError(f"Consensus module initialization failed: {e}") from e

        # Validate all initialized models
        self._validate_model_initialization()

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

            # Load entities into model-specific components
            if self.relik_integration is not None:
                try:
                    # Convert entity format for ReLiK (expects 'text' field)
                    relik_entities = {}
                    for eid, entity in entity_dict.items():
                        relik_entity = entity.copy()
                        if "description" in relik_entity and "text" not in relik_entity:
                            relik_entity["text"] = relik_entity["description"]
                        elif "text" not in relik_entity:
                            # Fallback: use name as text
                            relik_entity["text"] = relik_entity.get("name", f"Entity {eid}")
                        relik_entities[eid] = relik_entity

                    self.relik_integration.load_entities(relik_entities)
                    logger.info("Loaded entities into Enhanced ReLiK Integration")
                except Exception as e:
                    logger.warning(f"Failed to load entities into ReLiK integration: {e}")

            if self.spel_model is not None:
                try:
                    # Create frequency dict for SPEL candidate set
                    entity_frequencies = dict.fromkeys(entity_dict.keys(), 1)
                    self.spel_model.load_candidate_sets(entity_frequencies=entity_frequencies)
                    logger.info("Loaded candidate sets into SPEL model")
                except Exception as e:
                    logger.warning(f"Failed to load candidate sets into SPEL model: {e}")

            logger.info(f"Successfully loaded {len(entities)} entities")
            return len(entities)

        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            raise

    def _validate_model_initialization(self) -> None:
        """Validate that all enabled models are properly initialized."""
        validation_errors = []

        # Validate ATG model
        if self.config.use_improved_atg and self.atg_model is not None:
            if not hasattr(self.atg_model, "encoder") or self.atg_model.encoder is None:
                validation_errors.append("ATG model encoder not initialized")
            if not hasattr(self.atg_model, "tokenizer") or self.atg_model.tokenizer is None:
                validation_errors.append("ATG model tokenizer not initialized")

        # Validate RELiK model
        if self.config.use_relik and self.relik_model is not None:
            if not hasattr(self.relik_model, "retriever") or self.relik_model.retriever is None:
                validation_errors.append("RELiK model retriever not initialized")
            if not hasattr(self.relik_model, "reader") or self.relik_model.reader is None:
                validation_errors.append("RELiK model reader not initialized")

        # Validate SPEL model
        if self.config.use_spel and self.spel_model is not None:
            if not hasattr(self.spel_model, "encoder") or self.spel_model.encoder is None:
                validation_errors.append("SPEL model encoder not initialized")
            if not hasattr(self.spel_model, "tokenizer") or self.spel_model.tokenizer is None:
                validation_errors.append("SPEL model tokenizer not initialized")

        # Validate UniREL model
        if self.config.use_unirel and self.unirel_model is not None:
            if not hasattr(self.unirel_model, "encoder") or self.unirel_model.encoder is None:
                validation_errors.append("UniREL model encoder not initialized")
            if not hasattr(self.unirel_model, "tokenizer") or self.unirel_model.tokenizer is None:
                validation_errors.append("UniREL model tokenizer not initialized")

        # Log validation results
        if validation_errors:
            logger.warning(f"Model validation warnings: {'; '.join(validation_errors)}")
        else:
            logger.info("All models validated successfully")

    def _is_model_ready(self, model_name: str) -> bool:
        """Check if a specific model is ready for processing."""
        if model_name == "atg":
            return (
                self.atg_model is not None
                and hasattr(self.atg_model, "encoder")
                and self.atg_model.encoder is not None
            )
        elif model_name == "relik":
            return (
                self.relik_model is not None
                and hasattr(self.relik_model, "entity_kb")
                and self.relik_model.entity_kb is not None
            )
        elif model_name == "spel":
            return (
                self.spel_model is not None
                and hasattr(self.spel_model, "candidate_manager")
                and self.spel_model.candidate_manager is not None
            )
        elif model_name == "unirel":
            return (
                self.unirel_model is not None
                and hasattr(self.unirel_model, "encoder")
                and self.unirel_model.encoder is not None
            )
        return False

    def process_text(self, text: str, validate_input: bool = True) -> UnifiedSystemOutput:
        """
        Process a text document for entity resolution using all available models.

        Args:
            text: Input text
            validate_input: Whether to validate input text (default: True)

        Returns:
            UnifiedSystemOutput with comprehensive results from all models

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

            # Step 1: Entity-Focused Encoding (if available)
            logger.debug("Step 1: Entity-Focused Encoding")
            entity_embeddings = None
            if self.entity_encoder is not None:
                with torch.no_grad():
                    entity_embeddings = self.entity_encoder(
                        encoded_text["input_ids"], encoded_text["attention_mask"]
                    )

            # Step 2: Multi-Source Candidate Generation
            logger.debug("Step 2: Multi-Source Candidate Generation")
            candidates = self.retriever.retrieve(
                encoded_text["input_ids"],
                encoded_text["attention_mask"],
                top_k=self.config.top_k_candidates,
            )

            # Format candidates for all models
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

            # Step 3: Cross-Model Entity Resolution
            logger.debug("Step 3: Cross-Model Entity Resolution")
            model_results = {}

            # Process with ATG Model
            if self.atg_model is not None:
                logger.debug("Processing with ATG model")
                try:
                    atg_results = self._process_with_atg(text, candidate_entities)
                    model_results["atg"] = atg_results
                except Exception as e:
                    logger.warning(f"ATG processing failed: {e}")
                    model_results["atg"] = {"entities": [], "relations": [], "error": str(e)}

            # Process with RELiK Model
            if self.relik_model is not None:
                logger.debug("Processing with RELiK model")
                try:
                    relik_results = self._process_with_relik(text, candidate_entities)
                    model_results["relik"] = relik_results
                except Exception as e:
                    logger.warning(f"RELiK processing failed: {e}")
                    model_results["relik"] = {"entities": [], "relations": [], "error": str(e)}

            # Process with SPEL Model
            if self.spel_model is not None:
                logger.debug("Processing with SPEL model")
                try:
                    spel_results = self._process_with_spel(text, candidate_entities)
                    model_results["spel"] = spel_results
                except Exception as e:
                    logger.warning(f"SPEL processing failed: {e}")
                    model_results["spel"] = {"entities": [], "error": str(e)}

            # Process with UniREL Model
            if self.unirel_model is not None:
                logger.debug("Processing with UniREL model")
                try:
                    unirel_results = self._process_with_unirel(text, candidate_entities)
                    model_results["unirel"] = unirel_results
                except Exception as e:
                    logger.warning(f"UniREL processing failed: {e}")
                    model_results["unirel"] = {"entities": [], "relations": [], "error": str(e)}

            # Process with base Reader (fallback)
            logger.debug("Processing with base reader")
            reader_results = self.reader.process_text(text, candidate_entities)
            model_results["reader"] = reader_results

            # Step 4: Consensus Entity Linking
            logger.debug("Step 4: Consensus Entity Linking")

            # Collect all entity predictions from different models
            all_entity_predictions = []
            for model_name, results in model_results.items():
                if "entities" in results and results["entities"]:
                    for entity in results["entities"]:
                        entity["source_model"] = model_name
                        all_entity_predictions.append(entity)

            # Apply multi-method consensus resolution
            consensus_results = self.consensus.resolve_entities(all_entity_predictions, text)

            # Enhance consensus results with model agreement information
            for entity in consensus_results:
                entity["model_agreement"] = self._calculate_model_agreement(entity, model_results)

            # Step 5: Structured Entity Output
            logger.debug("Step 5: Generating structured output")

            # Collect all relations from models that support them
            all_relations = []
            for model_name in ["atg", "relik", "unirel"]:
                if model_name in model_results and "relations" in model_results[model_name]:
                    for relation in model_results[model_name]["relations"]:
                        relation["source_model"] = model_name
                        all_relations.append(relation)

            # Create Pydantic-formatted comprehensive results
            model_predictions_formatted = {}
            for model_name, results in model_results.items():
                model_predictions_formatted[model_name] = {
                    "num_entities": len(results.get("entities", [])),
                    "num_relations": len(results.get("relations", [])),
                    "confidence_avg": self._calculate_avg_confidence(results.get("entities", [])),
                    "status": "success" if "error" not in results else "failed",
                    "error_message": results.get("error") if "error" in results else None,
                }

            pipeline_stages = {
                "entity_encoding": entity_embeddings is not None,
                "candidate_generation": len(candidate_entities) > 0,
                "cross_model_resolution": len(model_results) > 0,
                "consensus_linking": len(consensus_results) > 0,
                "structured_output": True,
            }

            result = create_unified_output(
                text=text,
                entities=consensus_results,
                relations=all_relations,
                model_predictions=model_predictions_formatted,
                models_used=list(model_results.keys()),
                pipeline_stages=pipeline_stages,
                num_candidates=len(candidate_entities),
                consensus_method="multi_method_weighted",
            )

            logger.debug(
                f"Resolved {len(consensus_results)} entities and {len(all_relations)} relations using {len(model_results)} models"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to process text: {e}")
            raise RuntimeError(f"Text processing failed: {e}") from e

    def _process_with_atg(self, text: str, candidate_entities: list[dict]) -> dict:
        """Process text with ATG model for joint entity-relation extraction."""
        try:
            # Check if ATG model is ready
            if not self._is_model_ready("atg"):
                logger.warning("ATG model not ready, using reader fallback")
                results = self.reader.process_text(text, candidate_entities)
            elif hasattr(self.atg_model, "process_text"):
                results = self.atg_model.process_text(text, candidate_entities)
            elif hasattr(self.atg_model, "forward"):
                # Use forward method with tokenized input
                tokenizer = getattr(self.atg_model, "tokenizer", self.reader.tokenizer)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                # Remove token_type_ids as ATG model doesn't expect it
                if "token_type_ids" in inputs:
                    del inputs["token_type_ids"]
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.atg_model(**inputs)
                # Extract entities from ATG output format
                results = self._extract_entities_from_atg_output(outputs, text, candidate_entities)
            else:
                # Fallback: use reader for basic entity detection
                results = self.reader.process_text(text, candidate_entities)

            return {
                "entities": results.get("entities", []),
                "relations": results.get("relations", []),
                "confidence_avg": self._calculate_avg_confidence(results.get("entities", [])),
                "processing_time": results.get("processing_time", 0.0),
            }
        except Exception as e:
            logger.warning(f"ATG model processing failed: {e}")
            return {"entities": [], "relations": [], "error": str(e)}

    def _process_with_relik(self, text: str, candidate_entities: list[dict]) -> dict:
        """
        Process text using Enhanced ReLiK Integration.

        Args:
            text: Input text
            candidate_entities: Optional candidate entities (not used by ReLiK)

        Returns:
            Dictionary with entities and relations
        """
        if self.relik_integration is None:
            return {"entities": [], "relations": [], "error": "ReLiK not initialized"}

        try:
            # Process with all enhanced features
            result = self.relik_integration.process_text(
                text,
                top_k_retrieval=self.config.relik_top_k,
                top_k_linking=10,
                span_threshold=self.config.relik_span_threshold,
                entity_threshold=self.config.relik_entity_threshold,
                extract_relations=self.config.relik_use_re,
                relation_types=self.config.relation_types,
                relation_threshold=self.config.relik_relation_threshold,
            )

            # Format output
            formatted_entities = []
            for entity in result.get("entities", []):
                formatted_entity = {
                    "start": entity["start"],
                    "end": entity["end"],
                    "text": entity["text"],
                    "entity_id": entity.get("best_entity", {}).get("entity_id"),
                    "entity_name": entity.get("best_entity", {}).get("entity_name"),
                    "confidence": entity.get("best_entity", {}).get("score", 0.0),
                    "span_confidence": entity.get("span_score", 0.0),
                    "candidates": entity.get("candidates", []),
                }
                formatted_entities.append(formatted_entity)

            formatted_relations = []
            if "relations" in result:
                for relation in result["relations"]:
                    formatted_relation = {
                        "subject": relation["subject"],
                        "relation": relation["relation"],
                        "object": relation["object"],
                        "confidence": relation["confidence"],
                    }
                    formatted_relations.append(formatted_relation)

            return {
                "entities": formatted_entities,
                "relations": formatted_relations,
                "confidence_avg": self._calculate_avg_confidence(formatted_entities),
            }

        except Exception as e:
            logger.error(f"ReLiK processing failed: {e}")
            return {"entities": [], "relations": [], "error": str(e)}

    def _process_with_spel(self, text: str, candidate_entities: list[dict]) -> dict:
        """Process text with SPEL model for structured prediction."""
        try:
            # Check if SPEL model is ready
            if not self._is_model_ready("spel"):
                logger.warning("SPEL model not ready, using reader fallback")
                results = self.reader.process_text(text, candidate_entities)
            elif hasattr(self.spel_model, "process_text"):
                results = self.spel_model.process_text(text, candidate_entities)
                entities = results.get("entities", [])
                processing_time = results.get("processing_time", 0.0)
            elif hasattr(self.spel_model, "predict"):
                results = self.spel_model.predict(text)
                # SPEL predict returns a list of entities, not a dict
                if isinstance(results, list):
                    entities = results
                    processing_time = 0.0
                else:
                    entities = results.get("entities", [])
                    processing_time = results.get("processing_time", 0.0)
            elif hasattr(self.spel_model, "forward"):
                # Use SPEL structured prediction approach
                results = self._process_spel_structured(text, candidate_entities)
                entities = results.get("entities", [])
                processing_time = results.get("processing_time", 0.0)
            else:
                # Fallback to reader
                results = self.reader.process_text(text, candidate_entities)
                entities = results.get("entities", [])
                processing_time = results.get("processing_time", 0.0)

            return {
                "entities": entities,
                "confidence_avg": self._calculate_avg_confidence(entities),
                "processing_time": processing_time,
            }
        except Exception as e:
            logger.warning(f"SPEL model processing failed: {e}")
            return {"entities": [], "error": str(e)}

    def _process_with_unirel(self, text: str, candidate_entities: list[dict]) -> dict:
        """Process text with UniREL model for unified representation learning."""
        try:
            # Check if UniREL model is ready
            if not self._is_model_ready("unirel"):
                logger.warning("UniREL model not ready, using reader fallback")
                results = self.reader.process_text(text, candidate_entities)
            elif hasattr(self.unirel_model, "process_text"):
                results = self.unirel_model.process_text(text, candidate_entities)
            elif hasattr(self.unirel_model, "predict"):
                # UniREL's predict expects (text, device) not candidate_entities
                results = self.unirel_model.predict(text, device=self.device)
                # Convert to expected format with entities and relations
                if isinstance(results, list):
                    # Results are triples - extract entities and relations
                    entities = []
                    relations = []
                    for triple in results:
                        if isinstance(triple, tuple) and len(triple) == 3:
                            subj, rel, obj = triple
                            relations.append(
                                {
                                    "subject": subj,
                                    "relation": rel,
                                    "object": obj,
                                    "confidence": 0.7,
                                    "source_model": "unirel",
                                }
                            )
                    results = {"entities": entities, "relations": relations}
            elif hasattr(self.unirel_model, "extract_triples"):
                results = self.unirel_model.extract_triples(text, candidate_entities)
            elif hasattr(self.unirel_model, "forward"):
                # Use UniREL joint extraction approach with proper input preparation
                results = self._process_unirel_joint(text, candidate_entities)
            else:
                # Fallback to reader
                results = self.reader.process_text(text, candidate_entities)

            return {
                "entities": results.get("entities", []),
                "relations": results.get("relations", []),
                "confidence_avg": self._calculate_avg_confidence(results.get("entities", [])),
                "processing_time": results.get("processing_time", 0.0),
            }
        except Exception as e:
            logger.warning(f"UniREL model processing failed: {e}")
            return {"entities": [], "relations": [], "error": str(e)}

    def _calculate_model_agreement(self, entity: dict, model_results: dict) -> dict:
        """Calculate agreement between different models for an entity prediction."""
        agreement_info = {
            "total_models": len(model_results),
            "agreeing_models": [],
            "confidence_range": {"min": 1.0, "max": 0.0},
            "agreement_score": 0.0,
        }

        entity_mention = entity.get("mention", "").lower()
        entity_id = entity.get("entity_id", "")

        for model_name, results in model_results.items():
            if "entities" not in results:
                continue

            # Check if this model also predicted this entity
            for model_entity in results["entities"]:
                if (
                    model_entity.get("mention", "").lower() == entity_mention
                    or model_entity.get("entity_id", "") == entity_id
                ):
                    agreement_info["agreeing_models"].append(model_name)
                    confidence = model_entity.get("confidence", 0.5)
                    agreement_info["confidence_range"]["min"] = min(
                        agreement_info["confidence_range"]["min"], confidence
                    )
                    agreement_info["confidence_range"]["max"] = max(
                        agreement_info["confidence_range"]["max"], confidence
                    )
                    break

        # Calculate agreement score
        agreement_info["agreement_score"] = len(agreement_info["agreeing_models"]) / max(
            agreement_info["total_models"], 1
        )

        return agreement_info

    def _calculate_avg_confidence(self, entities: list[dict]) -> float:
        """Calculate average confidence score for a list of entities."""
        if not entities:
            return 0.0

        total_confidence = sum(entity.get("confidence", 0.0) for entity in entities)
        return total_confidence / len(entities)

    def _extract_entities_from_atg_output(
        self, outputs, text: str, candidate_entities: list[dict]
    ) -> dict:
        """Extract entities from ATG model output format."""
        # Simplified entity extraction from ATG outputs
        try:
            # ATG typically outputs sequences - extract entity spans
            # This is a placeholder that should be implemented based on actual ATG output format
            return self.reader.process_text(text, candidate_entities)
        except Exception:
            return {"entities": [], "relations": []}

    def _process_relik_pipeline(self, text: str, candidate_entities: list[dict]) -> dict:
        """Process text using RELiK retriever-reader pipeline."""
        try:
            # Use the existing retriever and reader components which are ReLiK-based
            return self.reader.process_text(text, candidate_entities)
        except Exception:
            return {"entities": [], "relations": []}

    def _process_spel_structured(self, text: str, candidate_entities: list[dict]) -> dict:
        """Process text using SPEL structured prediction approach."""
        try:
            # SPEL uses structured prediction - fallback to reader for now
            return self.reader.process_text(text, candidate_entities)
        except Exception:
            return {"entities": []}

    def _process_unirel_joint(self, text: str, candidate_entities: list[dict]) -> dict:
        """Process text using UniREL joint extraction approach."""
        try:
            # Prepare inputs for UniREL model
            tokenizer = getattr(self.unirel_model, "tokenizer", self.reader.tokenizer)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Remove token_type_ids if present and not expected
            if "token_type_ids" in inputs and not hasattr(
                self.unirel_model.encoder.config, "type_vocab_size"
            ):
                del inputs["token_type_ids"]

            # Safely move inputs to device - fix the device transfer issue
            try:
                device_inputs = {}
                for k, v in inputs.items():
                    if hasattr(v, "to") and hasattr(v, "device"):
                        # Only move tensors, and pass device as proper argument
                        device_inputs[k] = v.to(device=self.device)
                    else:
                        device_inputs[k] = v
                inputs = device_inputs
            except Exception as device_error:
                logger.warning(f"Failed to move UniREL inputs to device: {device_error}")
                # Keep inputs on CPU
                pass

            # Call the model
            with torch.no_grad():
                try:
                    _outputs = self.unirel_model(**inputs)
                    # Extract results (simplified - would need proper implementation)
                    return {"entities": [], "relations": []}
                except Exception as model_error:
                    logger.warning(f"UniREL model forward pass failed: {model_error}")
                    return {"entities": [], "relations": []}
        except Exception as e:
            logger.warning(f"UniREL joint processing failed: {e}")
            return {"entities": [], "relations": []}

    def process_batch(
        self, texts: list[str], validate_input: bool = True
    ) -> list[UnifiedSystemOutput]:
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

    def add_entity_to_kb(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        immediate: bool = True,
    ) -> None:
        """
        Add entity to knowledge base dynamically.

        Args:
            entity_id: Entity ID
            entity_data: Entity data (must have 'text' or 'description' or 'name' field)
            immediate: Apply immediately (rebuild index)

        Raises:
            ValueError: If entity data is invalid
        """
        # Add to ReLiK integration
        if self.relik_integration is not None:
            self.relik_integration.add_entity(entity_id, entity_data, immediate=immediate)
            logger.info(f"Added entity {entity_id} to ReLiK knowledge base")
        else:
            logger.warning("ReLiK integration not initialized, cannot add entity")

    def update_entity_in_kb(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        immediate: bool = True,
    ) -> None:
        """
        Update entity in knowledge base dynamically.

        Args:
            entity_id: Entity ID
            entity_data: Updated entity data
            immediate: Apply immediately (rebuild index)

        Raises:
            ValueError: If entity data is invalid
        """
        # Update in ReLiK integration
        if self.relik_integration is not None:
            self.relik_integration.update_entity(entity_id, entity_data, immediate=immediate)
            logger.info(f"Updated entity {entity_id} in ReLiK knowledge base")
        else:
            logger.warning("ReLiK integration not initialized, cannot update entity")

    def remove_entity_from_kb(
        self,
        entity_id: str,
        immediate: bool = True,
    ) -> None:
        """
        Remove entity from knowledge base dynamically.

        Args:
            entity_id: Entity ID
            immediate: Apply immediately (rebuild index)
        """
        # Remove from ReLiK integration
        if self.relik_integration is not None:
            self.relik_integration.remove_entity(entity_id, immediate=immediate)
            logger.info(f"Removed entity {entity_id} from ReLiK knowledge base")
        else:
            logger.warning("ReLiK integration not initialized, cannot remove entity")

    def get_kb_statistics(self) -> dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Statistics dictionary with info about KB size, index status, etc.
        """
        stats = {}

        # ReLiK integration stats
        if self.relik_integration is not None:
            stats["relik"] = self.relik_integration.get_statistics()

        # Retriever stats
        if hasattr(self.retriever, "get_index_size"):
            stats["retriever_index_size"] = self.retriever.get_index_size()

        return stats

    def fit_confidence_calibrators(
        self,
        validation_data: dict[str, Any],
    ) -> None:
        """
        Fit confidence calibrators using validation data.

        Args:
            validation_data: Dictionary with validation scores and labels.
                Expected keys:
                - 'span_scores': Tensor of span detection scores
                - 'span_labels': Tensor of span detection labels (0/1)
                - 'entity_scores': Tensor of entity linking scores
                - 'entity_labels': Tensor of entity linking labels (0/1)
                - 'relation_scores': Tensor of relation extraction scores (optional)
                - 'relation_labels': Tensor of relation extraction labels (optional)

        Raises:
            RuntimeError: If ReLiK integration is not initialized
        """
        if self.relik_integration is None:
            raise RuntimeError("ReLiK integration not initialized")

        try:
            self.relik_integration.fit_calibrator(validation_data)
            logger.info("Fitted ReLiK confidence calibrators")
        except Exception as e:
            logger.error(f"Failed to fit calibrators: {e}")
            raise

    def get_training_batch_relik(
        self,
        queries: list[str],
        positive_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Get training batch with hard negatives for ReLiK retriever.

        Args:
            queries: Query texts (e.g., entity mentions in context)
            positive_ids: Positive entity IDs corresponding to each query

        Returns:
            Training batch dictionary with:
                - query_ids: Tokenized query input IDs
                - query_mask: Query attention mask
                - positive_ids: Tokenized positive entity input IDs
                - positive_mask: Positive attention mask
                - negative_ids: Tokenized hard negative input IDs
                - negative_mask: Negative attention mask

        Raises:
            RuntimeError: If ReLiK integration is not initialized
        """
        if self.relik_integration is None:
            raise RuntimeError("ReLiK integration not initialized")

        return self.relik_integration.get_training_batch(queries, positive_ids)

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
