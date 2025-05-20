import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import logging

from src.entity_resolution.models.retriever import EntityRetriever
from src.entity_resolution.models.reader import EntityReader
from src.entity_resolution.models.consensus import ConsensusModule
from src.entity_resolution.database.vector_store import EntityKnowledgeBase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedEntityResolutionSystem(nn.Module):
    """
    Unified entity resolution system that integrates state-of-the-art
    techniques from ReLiK, SpEL, ATG, UniRel, and OneNet.
    """
    def __init__(self, config=None):
        super().__init__()

        # Default configuration if none provided
        if config is None:
            config = {
                "retriever_model": "microsoft/deberta-v3-small",
                "reader_model": "microsoft/deberta-v3-base",
                "entity_dim": 256,
                "max_seq_length": 512,
                "max_entity_length": 100,
                "top_k_candidates": 50,
                "consensus_threshold": 0.6,
                "batch_size": 8,
                "index_path": "./entity_index",
                "cache_dir": "./cache",
                "use_gpu": torch.cuda.is_available(),
                "quantization": None  # None, "int8", or "fp16"
            }

        self.config = config
        self.device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

        # Create cache directory if it doesn't exist
        os.makedirs(config["cache_dir"], exist_ok=True)

        # Initialize knowledge base
        self.knowledge_base = self._initialize_knowledge_base()

        # Initialize retriever
        logger.info(f"Initializing retriever with model {config['retriever_model']}")
        self.retriever = EntityRetriever(
            model_name=config["retriever_model"],
            entity_dim=config["entity_dim"],
            shared_encoder=True,  # Parameter efficient
            use_faiss=True,       # Use FAISS for efficient retrieval
            top_k=config["top_k_candidates"]
        )

        # Move retriever to device
        self.retriever.to(self.device)

        # Initialize reader
        logger.info(f"Initializing reader with model {config['reader_model']}")
        self.reader = EntityReader(
            model_name=config["reader_model"],
            max_seq_length=config["max_seq_length"],
            max_entity_length=config["max_entity_length"],
            gradient_checkpointing=True  # Enable gradient checkpointing for memory efficiency
        )

        # Move reader to device
        self.reader.to(self.device)

        # Initialize consensus module
        self.consensus = ConsensusModule(
            hidden_size=self.reader.config.hidden_size,
            threshold=config["consensus_threshold"]
        )

        # Move consensus module to device
        self.consensus.to(self.device)

        # Apply quantization if specified
        if config["quantization"]:
            self._apply_quantization(config["quantization"])

        # Initialize cache for entity embeddings
        self.entity_embedding_cache = {}

        logger.info("Entity resolution system initialized")

    def _initialize_knowledge_base(self):
        """Initialize the entity knowledge base"""
        kb = EntityKnowledgeBase(
            index_path=self.config["index_path"],
            cache_dir=self.config["cache_dir"]
        )
        return kb

    def _apply_quantization(self, quantization_type):
        """Apply quantization to models for faster inference"""
        logger.info(f"Applying {quantization_type} quantization")

        if quantization_type == "int8":
            # Quantize retriever
            import torch.quantization
            self.retriever.eval()
            self.retriever = torch.quantization.quantize_dynamic(
                self.retriever,
                {nn.Linear},
                dtype=torch.qint8
            )

            # Quantize reader
            self.reader.quantize("int8")

        elif quantization_type == "fp16":
            # Convert to half precision
            self.retriever = self.retriever.half()
            self.reader.quantize("fp16")
            self.consensus = self.consensus.half()

        logger.info(f"Models quantized to {quantization_type}")

    def load_entities(self, entity_file):
        """
        Load entities from a file into the knowledge base.

        Args:
            entity_file: Path to entity file (JSON or CSV)
        """
        logger.info(f"Loading entities from {entity_file}")

        # Load entities
        entities = self.knowledge_base.load_entities(entity_file)

        # Build retriever index
        self.retriever.build_index(entities)

        logger.info(f"Loaded {len(entities)} entities into knowledge base")

        return len(entities)

    def process_text(self, text):
        """
        Process a text document for entity resolution.

        Args:
            text: Input text

        Returns:
            Dictionary with resolved entities
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        # Tokenize text for retriever
        tokenizer = self.retriever.tokenizer
        encoded_text = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config["max_seq_length"]
        ).to(self.device)

        # Step 1: Retrieve candidate entities (ReLiK approach)
        logger.info("Retrieving candidate entities")
        with torch.no_grad():
            candidates = self.retriever.retrieve(
                encoded_text["input_ids"],
                encoded_text["attention_mask"],
                top_k=self.config["top_k_candidates"]
            )

        # Format candidates for reader
        candidate_entities = []
        for entity_id, entity_data, score in candidates:
            candidate_entities.append({
                "id": entity_id,
                "name": entity_data["name"],
                "description": entity_data.get("description", ""),
                "type": entity_data.get("type", "UNKNOWN"),
                "score": score
            })

        logger.info(f"Retrieved {len(candidate_entities)} candidate entities")

        # Step 2: Process with reader to link entities (SpEL approach)
        logger.info("Processing with reader")
        reader_results = self.reader.process_text(text, candidate_entities)

        # Step 3: Apply consensus to resolve conflicts (OneNet approach)
        logger.info("Applying consensus")
        consensus_results = self.consensus.resolve_entities(reader_results["entities"], text)

        # Format final results
        result = {
            "text": text,
            "entities": consensus_results
        }

        return result

    def process_batch(self, texts):
        """
        Process a batch of texts for entity resolution.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries with resolved entities
        """
        results = []

        # Process in batches of the configured size
        batch_size = self.config["batch_size"]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            # Process each text in the batch
            batch_results = []
            for text in batch:
                result = self.process_text(text)
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def train(self, train_data, val_data=None, learning_rate=1e-5, num_epochs=5):
        """
        Train the entity resolution system.

        Args:
            train_data: Training data
            val_data: Validation data
            learning_rate: Learning rate
            num_epochs: Number of training epochs

        Returns:
            Training history
        """
        logger.info("Starting training")

        # Set models to training mode
        self.retriever.train()
        self.reader.train()
        self.consensus.train()

        # Create optimizer
        optimizer = torch.optim.AdamW([
            {'params': self.retriever.parameters(), 'lr': learning_rate},
            {'params': self.reader.parameters(), 'lr': learning_rate},
            {'params': self.consensus.parameters(), 'lr': learning_rate}
        ], lr=learning_rate)

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": []
        }

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Training
            total_loss = 0
            for batch in train_data:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                loss = self._training_step(batch)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.item()

            # Average training loss
            avg_train_loss = total_loss / len(train_data)
            history["train_loss"].append(avg_train_loss)

            logger.info(f"Training loss: {avg_train_loss:.4f}")

            # Validation
            if val_data is not None:
                val_loss, val_f1 = self._validate(val_data)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)

                logger.info(f"Validation loss: {val_loss:.4f}, F1: {val_f1:.4f}")

        logger.info("Training complete")

        return history

    def _training_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch: Batch of training data

        Returns:
            Loss value
        """
        # Process batch with retriever
        retriever_outputs = self.retriever(
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            entity_ids=batch["entity_ids"]
        )

        # Calculate retriever loss
        retriever_loss = self.retriever.contrastive_loss(
            retriever_outputs["text_embeddings"],
            retriever_outputs["entity_embeddings"],
            batch["positive_pairs"]
        )

        # Process batch with reader
        reader_outputs = self.reader(batch)

        # Calculate reader loss
        reader_loss = F.binary_cross_entropy_with_logits(
            reader_outputs["entity_scores"],
            batch["entity_labels"].to(self.device)
        )

        # Process batch with consensus
        consensus_loss = self.consensus.loss(
            reader_outputs["entities"],
            batch["entity_labels"].to(self.device)
        )

        # Combine losses
        total_loss = 0.3 * retriever_loss + 0.5 * reader_loss + 0.2 * consensus_loss

        return total_loss

    def _validate(self, val_data):
        """
        Validate the entity resolution system.

        Args:
            val_data: Validation data

        Returns:
            Tuple of (validation loss, F1 score)
        """
        # Set models to evaluation mode
        self.retriever.eval()
        self.reader.eval()
        self.consensus.eval()

        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_data:
                # Forward pass
                loss = self._training_step(batch)
                total_loss += loss.item()

                # Get predictions
                predictions = self.process_batch(batch["texts"])

                # Collect predictions and labels
                all_predictions.extend(predictions)
                all_labels.extend(batch["labels"])

        # Calculate average loss
        avg_loss = total_loss / len(val_data)

        # Calculate F1 score
        f1 = self._calculate_f1(all_predictions, all_labels)

        # Set models back to training mode
        self.retriever.train()
        self.reader.train()
        self.consensus.train()

        return avg_loss, f1

    def _calculate_f1(self, predictions, labels):
        """
        Calculate F1 score for entity resolution.

        Args:
            predictions: Predicted entities
            labels: Ground truth entities

        Returns:
            F1 score
        """
        tp = fp = fn = 0

        for pred_dict, label_dict in zip(predictions, labels):
            pred_entities = pred_dict["entities"]
            label_entities = label_dict["entities"]

            # Convert to sets for easy comparison
            pred_set = {(e["mention"], e["entity_id"]) for e in pred_entities}
            label_set = {(e["mention"], e["entity_id"]) for e in label_entities}

            # Calculate TP, FP, FN
            tp += len(pred_set.intersection(label_set))
            fp += len(pred_set - label_set)
            fn += len(label_set - pred_set)

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return f1

    def save(self, path):
        """
        Save the entity resolution system.

        Args:
            path: Path to save the model
        """
        logger.info(f"Saving model to {path}")

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save retriever
        self.retriever.save(f"{path}/retriever")

        # Save reader
        self.reader.save(f"{path}/reader")

        # Save consensus module
        torch.save(self.consensus.state_dict(), f"{path}/consensus.pt")

        # Save configuration
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """
        Load the entity resolution system.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {path}")

        # Load configuration
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)

        # Create model instance
        model = cls(config)

        # Load retriever
        model.retriever = EntityRetriever.load(f"{path}/retriever")

        # Load reader
        model.reader = EntityReader.load(f"{path}/reader")

        # Load consensus module
        model.consensus.load_state_dict(torch.load(f"{path}/consensus.pt"))

        # Move models to device
        model.retriever.to(model.device)
        model.reader.to(model.device)
        model.consensus.to(model.device)

        logger.info(f"Model loaded from {path}")

        return model
