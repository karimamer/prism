import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from .models.candidate_generation import EntityCandidateGenerator
from .models.encoder import EntityFocusedEncoder
from .models.processor import EntityResolutionProcessor
from .models.consensus import EntityConsensusModule
from .models.output import EntityOutputFormatter

class UnifiedEntityResolutionSystem(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        # Default configuration if none provided
        if config is None:
            config = {
                "encoder_name": "microsoft/deberta-v3-base",
                "encoder_dim": 768,  # DeBERTa-v3-base hidden size
                "entity_knowledge_dim": 256,
                "max_seq_length": 512,
                "num_entity_types": 50,
                "consensus_threshold": 0.6,
                "top_k_candidates": 50
            }

        self.config = config

        # Initialize tokenizer with use_fast=False to avoid conversion issues
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["encoder_name"],
                use_fast=False
            )
            print(f"Loaded tokenizer for {config['encoder_name']}")
        except Exception as e:
            print(f"Error loading DeBERTa tokenizer: {e}")
            print("Falling back to RoBERTa tokenizer")
            # Fallback to RoBERTa if DeBERTa has issues
            self.tokenizer = AutoTokenizer.from_pretrained(
                "roberta-base",
                use_fast=True
            )

        # Initialize a simple base encoder directly instead of using EntityFocusedEncoder
        try:
            self.base_encoder = AutoModel.from_pretrained(config["encoder_name"])
            print(f"Loaded model for {config['encoder_name']}")
        except Exception as e:
            print(f"Error loading DeBERTa model: {e}")
            print("Falling back to RoBERTa model")
            # Fallback to RoBERTa if DeBERTa has issues
            self.base_encoder = AutoModel.from_pretrained("roberta-base")
            config["encoder_name"] = "roberta-base"

        # Initialize entity encoder as a wrapper around base encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(self.base_encoder.config.hidden_size, config["entity_knowledge_dim"]),
            nn.ReLU(),
            nn.Linear(config["entity_knowledge_dim"], config["entity_knowledge_dim"])
        )

        # Initialize knowledge base
        self.knowledge_base = self._create_knowledge_base()

        # Other components
        self.candidate_generator = EntityCandidateGenerator(
            knowledge_base=self.knowledge_base,
            embedding_dim=config["encoder_dim"]
        )

        self.entity_processor = EntityResolutionProcessor(
            encoder_dim=config["encoder_dim"]
        )

        self.consensus_module = EntityConsensusModule(
            encoder_dim=config["encoder_dim"],
            threshold=config["consensus_threshold"]
        )

        self.output_formatter = EntityOutputFormatter(self.tokenizer)

        # Enable gradient checkpointing for large models
        if "large" in config["encoder_name"]:
            self.base_encoder.gradient_checkpointing_enable()

    def _create_knowledge_base(self):
        """Create or load knowledge base - placeholder implementation"""
        # In a real implementation, this would load entity data from disk
        # or connect to an external knowledge base service
        return InMemoryKnowledgeBase()

    def forward(self, input_ids, attention_mask, entity_knowledge=None):
        """
        Main forward pass for entity resolution

        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for input text
            entity_knowledge: Optional external entity knowledge

        Returns:
            Dictionary containing resolved entities and metadata
        """
        # 1. Encode input text with base encoder
        outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = outputs.last_hidden_state

        # 2. Generate entity candidates from multiple sources
        entity_candidates = self.candidate_generator(
            text_embeddings=text_embeddings,
            top_k=self.config["top_k_candidates"]
        )

        # 3. Process entity candidates with multiple methods
        entity_results = self.entity_processor(
            text_embeddings=text_embeddings,
            entity_candidates=entity_candidates
        )

        # 4. Reach consensus on entity resolution
        linked_entities = self.consensus_module(
            entity_results=entity_results,
            text_embeddings=text_embeddings
        )

        # 5. Format structured entity output
        entity_output = self.output_formatter(
            linked_entities=linked_entities,
            input_ids=input_ids
        )

        return entity_output

    def process_text(self, text):
        """
        Process raw text for entity resolution

        Args:
            text: Raw input text

        Returns:
            Dictionary containing resolved entities and metadata
        """
        # Tokenize the input text
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_seq_length"],
            return_tensors="pt",
            return_offsets_mapping=True  # For span mapping
        )

        # Store offset mapping for later use in span conversion
        offset_mapping = encoded.pop("offset_mapping", None)

        # If offset_mapping is None (some tokenizers don't support it),
        # create a simple approximation
        if offset_mapping is None:
            # Create a simple approximation of offset_mapping
            tokens = self.tokenizer.tokenize(text)
            offset_mapping = torch.zeros((1, len(tokens), 2), dtype=torch.long)

            current_pos = 0
            for i, token in enumerate(tokens):
                # Skip special tokens
                if token in self.tokenizer.all_special_tokens:
                    offset_mapping[0, i, 0] = 0
                    offset_mapping[0, i, 1] = 0
                    continue

                # Find position of token in original text
                token_text = token.replace("##", "").replace("Ġ", "")
                token_pos = text[current_pos:].find(token_text)
                if token_pos >= 0:
                    start_pos = current_pos + token_pos
                    end_pos = start_pos + len(token_text)
                    offset_mapping[0, i, 0] = start_pos
                    offset_mapping[0, i, 1] = end_pos
                    current_pos = end_pos
                else:
                    # Fallback if token not found
                    offset_mapping[0, i, 0] = current_pos
                    offset_mapping[0, i, 1] = current_pos

        # Simplified placeholder implementation during development
        # Instead of running the full model, we'll use the fallback entity extractor
        from src.entity_resolution.run_entity_resolution import extract_placeholder_entities
        entities = extract_placeholder_entities(text)

        # Format the output to match the expected structure
        output = {
            "entities": entities,
            "text": text
        }

        return output

        # Comment out the full model forward pass for now until the model is fully implemented
        # Forward pass
        # with torch.no_grad():
        #     outputs = self(
        #         input_ids=encoded["input_ids"],
        #         attention_mask=encoded["attention_mask"]
        #     )

        # Map spans back to original text using offset mapping
        # final_outputs = self._map_spans_to_original(outputs, offset_mapping)
        # final_outputs["text"] = text

        # return final_outputs

    def _map_spans_to_original(self, outputs, offset_mapping):
        """
        Maps token-level spans to character-level spans in original text

        Args:
            outputs: Model outputs with token-level spans
            offset_mapping: Mapping from tokens to character positions

        Returns:
            Updated outputs with character-level spans
        """
        entities = outputs["entities"]
        updated_entities = []

        for entity in entities:
            token_start, token_end = entity["mention_span"]

            # Convert token positions to character positions
            char_start = offset_mapping[0][token_start][0].item()
            char_end = offset_mapping[0][token_end][1].item()

            # Create updated entity with character-level spans
            updated_entity = {**entity}
            updated_entity["char_span"] = (char_start, char_end)
            updated_entities.append(updated_entity)

        return {"entities": updated_entities}

    def train_system(self, train_datasets, val_datasets, num_epochs=5):
        """
        End-to-end training of the entity resolution system

        Args:
            train_datasets: Training datasets for different components
            val_datasets: Validation datasets
            num_epochs: Number of training epochs
        """
        # Create optimizers with different learning rates for different components
        optimizer = torch.optim.AdamW([
            {'params': self.base_encoder.parameters(), 'lr': 5e-6},
            {'params': self.entity_encoder.parameters(), 'lr': 1e-5},
            {'params': self.candidate_generator.parameters(), 'lr': 2e-5},
            {'params': self.entity_processor.parameters(), 'lr': 2e-5},
            {'params': self.consensus_module.parameters(), 'lr': 2e-5}
        ], weight_decay=0.01)

        # Create scheduler
        total_steps = len(train_datasets["dataloader"]) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        # Training loop
        best_val_score = 0
        for epoch in range(num_epochs):
            # Train
            self.train()
            train_loss = 0

            for batch in train_datasets["dataloader"]:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                # Calculate loss
                loss = self._calculate_loss(outputs, batch["labels"])

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_datasets["dataloader"])
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

            # Validate
            val_score = self._validate(val_datasets)
            print(f"Validation Score: {val_score:.4f}")

            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(self.state_dict(), f"best_entity_resolution_model.pt")

        print(f"Training complete. Best validation score: {best_val_score:.4f}")

    def _calculate_loss(self, outputs, labels):
        """
        Calculate combined loss for entity resolution

        Args:
            outputs: Model outputs
            labels: Ground truth labels

        Returns:
            Combined loss value
        """
        # Extract different aspects of the task
        mention_loss = self._calculate_mention_detection_loss(outputs, labels)
        linking_loss = self._calculate_entity_linking_loss(outputs, labels)

        # Combine losses with weighting
        total_loss = 0.4 * mention_loss + 0.6 * linking_loss

        return total_loss

    def _calculate_mention_detection_loss(self, outputs, labels):
        """Calculate loss for mention detection component"""
        # Implementation depends on specific label format
        # Placeholder implementation
        return torch.tensor(0.0, requires_grad=True)

    def _calculate_entity_linking_loss(self, outputs, labels):
        """Calculate loss for entity linking component"""
        # Implementation depends on specific label format
        # Placeholder implementation
        return torch.tensor(0.0, requires_grad=True)

    def _validate(self, val_datasets):
        """
        Validate the model on validation data

        Args:
            val_datasets: Validation datasets

        Returns:
            Validation score (F1)
        """
        self.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_datasets["dataloader"]:
                outputs = self(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                # Convert outputs to predictions
                predictions = self._convert_outputs_to_predictions(outputs)

                all_predictions.extend(predictions)
                all_labels.extend(batch["labels"])

        # Calculate F1 score
        f1 = calculate_entity_f1(all_predictions, all_labels)

        return f1

    def _convert_outputs_to_predictions(self, outputs):
        """Convert model outputs to prediction format for evaluation"""
        # Implementation depends on output and evaluation format
        # Placeholder implementation
        return []

    def load_trained_model(self, model_path):
        """Load a trained model from disk"""
        self.load_state_dict(torch.load(model_path))
        self.eval()
        return self

    def save_model(self, model_path):
        """Save the current model to disk"""
        torch.save(self.state_dict(), model_path)


# Placeholder for knowledge base - would be implemented with actual data store
class InMemoryKnowledgeBase:
    def __init__(self):
        # Initialize with empty entity store
        self.entities = {}
        self.entity_embeddings = {}

    def retrieve(self, query_vector, top_k=100):
        """
        Retrieve most similar entities to query vector

        Args:
            query_vector: Query embedding
            top_k: Number of results to return

        Returns:
            Tuple of (entities, embeddings)
        """
        # In a real implementation, this would search a vector database
        # For now, return empty results
        return [], []

    def load_entities(self, entity_data):
        """Load entities into the knowledge base"""
        self.entities = entity_data
        # Generate embeddings for entities
        # This is a placeholder - would actually compute embeddings
        self.entity_embeddings = {eid: torch.randn(768) for eid in self.entities}

    def get_entity_by_id(self, entity_id):
        """Get entity by ID"""
        return self.entities.get(entity_id, None)


# Helper function for scheduler
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Helper function for F1 calculation
def calculate_entity_f1(predictions, gold_labels):
    """
    Calculate F1 score for entity resolution

    Args:
        predictions: Predicted entities
        gold_labels: Gold standard entities

    Returns:
        F1 score
    """
    # Count true positives, false positives, and false negatives
    tp = fp = fn = 0

    # This is a simplified implementation - would need to be adapted to actual data format
    for pred, gold in zip(predictions, gold_labels):
        if pred == gold and pred is not None:
            tp += 1
        elif pred is not None and gold is None:
            fp += 1
        elif pred is None and gold is not None:
            fn += 1

    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1
