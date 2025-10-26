import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np


class EntityReader(nn.Module):
    """
    ReLiK-based Reader component that performs entity linking and relation extraction.

    This reader follows the ReLiK approach:
    - Single forward pass for query + retrieved passages
    - Span-based mention detection with start/end token prediction
    - Entity linking through shared dense space projection
    - Relation extraction with Hadamard product for triplets
    """

    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        max_seq_length=512,
        max_entity_length=100,
        gradient_checkpointing=True,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_entity_length = max_entity_length

        # Load model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens for ReLiK format
        special_tokens = {
            "additional_special_tokens": [
                "[SEP]",  # Separator between query and passages
                "<ST0>",
                "<ST1>",
                "<ST2>",
                "<ST3>",
                "<ST4>",
                "<ST5>",
                "<ST6>",
                "<ST7>",
                "<ST8>",
                "<ST9>",
                "<ST10>",
                "<ST11>",
                "<ST12>",
                "<ST13>",
                "<ST14>",
                "<ST15>",
                "<ST16>",
                "<ST17>",
                "<ST18>",
                "<ST19>",
            ]
        }

        # Handle case where tokenizer doesn't have pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add markers to tokenizer
        num_added = self.tokenizer.add_special_tokens(special_tokens)

        # Resize token embeddings to account for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Get indices of special tokens for easy access
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.st_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ST{i}>") for i in range(20)]

        # Define entity and candidate marker token IDs
        self.entity_start_id = self.tokenizer.convert_tokens_to_ids("<ST0>")
        self.entity_end_id = self.tokenizer.convert_tokens_to_ids("<ST1>")
        self.candidate_start_id = self.tokenizer.convert_tokens_to_ids("<ST2>")
        self.candidate_end_id = self.tokenizer.convert_tokens_to_ids("<ST3>")

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Mention detection layers (start/end token prediction)
        self.mention_start_classifier = nn.Linear(
            self.config.hidden_size, 2
        )  # Binary classification
        self.mention_end_classifier = nn.Linear(
            self.config.hidden_size * 2, 2
        )  # Binary classification

        # Entity linking projection layer
        self.entity_projection = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size), nn.GELU()
        )

        # Relation extraction projection layers
        self.subject_projection = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size), nn.GELU()
        )
        self.object_projection = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size), nn.GELU()
        )
        self.relation_projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.GELU()
        )

        # Relation classification head
        self.relation_classifier = nn.Linear(self.config.hidden_size, 2)

        # Span classifier for mention detection (BIO tagging)
        self.span_classifier = nn.Linear(self.config.hidden_size, 3)  # B, I, O tags

        # Interaction layer for entity interactions
        self.interaction_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.GELU(), nn.Dropout(0.1)
        )

        # Entity linker for linking mentions to entities
        self.entity_linker = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 1),
            nn.Sigmoid(),
        )

        # Initialize special input format requirements
        self.requires_query_passage_format = True

    def encode_text_with_candidates(
        self,
        input_text: str,
        candidate_entities: List[Dict],
        detected_mentions: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict:
        """
        Encode input text with candidate entities in a single forward pass.

        Args:
            input_text: Raw input text
            candidate_entities: List of candidate entities with their information
            detected_mentions: Optional list of detected mention spans (start, end)

        Returns:
            Dictionary with model outputs
        """
        # Tokenize the formatted text
        encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        # Extract relevant outputs
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)
        offset_mapping = encoding["offset_mapping"]
        special_tokens_mask = encoding["special_tokens_mask"]

        # Track positions of entity mentions and candidates
        mention_positions = []
        candidate_positions = []

        # Find all entity start/end markers and candidate markers
        entity_start_positions = []
        entity_end_positions = []

        # Find positions of special tokens
        for i, input_id in enumerate(input_ids[0]):
            if input_id.item() == self.entity_start_id:
                entity_start_positions.append(i)
            elif input_id.item() == self.entity_end_id:
                entity_end_positions.append(i)
            elif input_id.item() == self.candidate_start_id:
                candidate_start = i
            elif input_id.item() == self.candidate_end_id:
                candidate_positions.append((candidate_start, i))

        # Match entity start/end positions to create mention spans
        if len(entity_start_positions) == len(entity_end_positions):
            for start, end in zip(entity_start_positions, entity_end_positions):
                mention_positions.append((start, end))

        # Forward pass through model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get contextualized token representations
        hidden_states = outputs.last_hidden_state

        return {
            "hidden_states": hidden_states,
            "mention_positions": mention_positions,
            "candidate_positions": candidate_positions,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
        }

    def detect_mentions(self, text):
        """
        Detect potential entity mentions in text.

        Args:
            text: Input text

        Returns:
            List of mention spans (start_token, end_token)
        """
        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=False,
        )

        input_ids = tokenized["input_ids"].to(self.model.device)
        attention_mask = tokenized["attention_mask"].to(self.model.device)
        offset_mapping = tokenized["offset_mapping"]

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Hidden states
        hidden_states = outputs.last_hidden_state

        # Apply span classifier
        tag_logits = self.span_classifier(hidden_states)
        tag_probs = F.softmax(tag_logits, dim=-1)

        # Get predictions (B-I-O tagging)
        predictions = torch.argmax(tag_probs, dim=-1)[0].cpu().numpy()

        # Extract mentions from BIO tags
        mentions = []
        current_mention = None

        for i, tag in enumerate(predictions):
            if tag == 1:  # B - Beginning of entity
                if current_mention is not None:
                    mentions.append((current_mention[0], i - 1))
                current_mention = (i, None)
            elif tag == 2:  # I - Inside entity
                continue
            elif tag == 0:  # O - Outside entity
                if current_mention is not None:
                    mentions.append((current_mention[0], i - 1))
                    current_mention = None

        # Handle case where mention continues until the end
        if current_mention is not None:
            mentions.append((current_mention[0], len(predictions) - 1))

        return mentions

    def link_entities(self, hidden_states, mention_positions, candidate_positions):
        """
        Link mentions to candidate entities using ReLiK approach.

        Args:
            hidden_states: Contextualized token representations
            mention_positions: List of mention span positions (start, end)
            candidate_positions: List of candidate entity positions (start, end)

        Returns:
            Dictionary with entity linking outputs
        """
        # Get mention representations
        mention_embeddings = []
        for start, end in mention_positions:
            # Average token embeddings in the span
            span_embedding = hidden_states[0, start : end + 1].mean(dim=0)
            mention_embeddings.append(span_embedding)

        # If no mentions, return empty results
        if not mention_embeddings:
            return {"mention_entity_scores": [], "best_entities": []}

        mention_embeddings = torch.stack(mention_embeddings)

        # Get candidate entity representations
        candidate_embeddings = []
        for start, end in candidate_positions:
            # Average token embeddings in the span
            span_embedding = hidden_states[0, start : end + 1].mean(dim=0)
            candidate_embeddings.append(span_embedding)

        # If no candidates, return empty results
        if not candidate_embeddings:
            return {"mention_entity_scores": [], "best_entities": []}

        candidate_embeddings = torch.stack(candidate_embeddings)

        # Apply interaction layer (UniRel approach)
        mention_embeddings = self.interaction_layer(mention_embeddings)
        candidate_embeddings = self.interaction_layer(candidate_embeddings)

        # Calculate scores for all mention-entity pairs
        mention_entity_scores = []

        for i, mention_emb in enumerate(mention_embeddings):
            scores = []
            for j, candidate_emb in enumerate(candidate_embeddings):
                # Concatenate mention and entity embeddings
                pair_emb = torch.cat([mention_emb, candidate_emb])

                # Get linking score
                score = self.entity_linker(pair_emb).item()
                scores.append((j, score))

            # Sort by score in descending order
            scores.sort(key=lambda x: x[1], reverse=True)
            mention_entity_scores.append(scores)

        # Get best entity for each mention
        best_entities = [scores[0][0] if scores else None for scores in mention_entity_scores]

        return {"mention_entity_scores": mention_entity_scores, "best_entities": best_entities}

    def compute_interaction_map(self, hidden_states, attention_mask):
        """
        Compute interaction map for entities using UniRel approach.

        Args:
            hidden_states: Contextualized token representations
            attention_mask: Attention mask for input tokens

        Returns:
            Interaction map between tokens
        """
        # Project hidden states
        projected_states = self.interaction_layer(hidden_states)

        # Compute attention scores between all tokens
        # Scaling factor for numerical stability
        scaling_factor = torch.sqrt(torch.tensor(projected_states.size(-1), dtype=torch.float))

        # Compute raw attention scores
        attention_scores = (
            torch.matmul(projected_states, projected_states.transpose(-1, -2)) / scaling_factor
        )

        # Apply attention mask
        mask = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
        attention_scores = attention_scores * mask + -1e9 * (1 - mask)

        # Apply sigmoid to get interaction scores
        interaction_map = torch.sigmoid(attention_scores)

        return interaction_map

    def forward(self, batch):
        """
        Forward pass for the entity reader.

        Args:
            batch: Dictionary with input data
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - mention_positions: (Optional) Positions of mentions
                - candidate_positions: (Optional) Positions of candidate entities

        Returns:
            Dictionary with model outputs
        """
        # Get inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        mention_positions = batch.get("mention_positions", None)
        candidate_positions = batch.get("candidate_positions", None)

        # Forward pass through model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get contextualized token representations
        hidden_states = outputs.last_hidden_state

        # Detect mentions if not provided
        if mention_positions is None:
            mention_detection = self.detect_mentions(hidden_states, attention_mask)
            mention_positions = mention_detection["mentions"][0]  # Take first batch example

        # Link entities
        entity_linking = self.link_entities(hidden_states, mention_positions, candidate_positions)

        # Compute interaction map
        interaction_map = self.compute_interaction_map(hidden_states, attention_mask)

        return {
            "hidden_states": hidden_states,
            "mention_positions": mention_positions,
            "mention_entity_scores": entity_linking["mention_entity_scores"],
            "best_entities": entity_linking["best_entities"],
            "interaction_map": interaction_map,
        }

    def process_text(
        self, query: str, candidate_entities: List[Dict], task_type: str = "entity_linking"
    ) -> Dict:
        """
        Process query with candidate entities for entity linking or relation extraction.

        Args:
            query: Input query text
            candidate_entities: List of candidate entity dictionaries
            task_type: "entity_linking", "relation_extraction", or "joint"

        Returns:
            Dictionary with processing results
        """
        with torch.no_grad():
            # Detect mentions in the query text
            mentions = self.detect_mentions(query)

            # Link mentions to candidate entities
            entities = []
            for mention_span in mentions:
                start, end = mention_span
                query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
                if start < len(query_tokens) and end < len(query_tokens):
                    mention_text = self.tokenizer.decode(query_tokens[start : end + 1]).strip()

                    # Find best matching entity
                    best_entity = None
                    best_score = 0.0

                    for entity in candidate_entities:
                        # Calculate similarity score based on entity name matching
                        entity_name = entity.get("name", "")
                        mention_lower = mention_text.lower()
                        entity_lower = entity_name.lower()

                        # Calculate score based on match quality
                        if mention_lower == entity_lower:
                            # Exact match
                            score = 1.0
                        elif mention_lower in entity_lower and entity_lower in mention_lower:
                            # Bidirectional substring (one contains the other both ways - exact match)
                            score = 1.0
                        elif mention_lower in entity_lower:
                            # Mention is substring of entity name
                            score = 0.7 + 0.2 * (len(mention_lower) / len(entity_lower))
                        elif entity_lower in mention_lower:
                            # Entity name is substring of mention
                            score = 0.6 + 0.3 * (len(entity_lower) / len(mention_lower))
                        else:
                            # No match
                            score = 0.0

                        if score > best_score:
                            best_score = score
                            best_entity = entity

                    if best_entity and best_score > 0.5:
                        entities.append(
                            {
                                "mention": mention_text,
                                "mention_span": mention_span,
                                "entity_id": best_entity.get("id", ""),
                                "entity_name": best_entity.get("name", ""),
                                "entity_type": best_entity.get("type", "UNKNOWN"),
                                "confidence": best_score,
                            }
                        )

        # Extract relations if doing relation extraction (simplified for now)
        relations = []
        if task_type in ["relation_extraction", "joint"]:
            # For now, return empty relations - would need more sophisticated relation extraction
            pass

        return {
            "query": query,
            "mentions": mentions,
            "entities": entities,
            "relations": relations,
            "task_type": task_type,
        }

    def save(self, path):
        """Save ReLiK reader model"""
        torch.save(
            {
                "model": self.model.state_dict(),
                "mention_start_classifier": self.mention_start_classifier.state_dict(),
                "mention_end_classifier": self.mention_end_classifier.state_dict(),
                "entity_projection": self.entity_projection.state_dict(),
                "subject_projection": self.subject_projection.state_dict(),
                "object_projection": self.object_projection.state_dict(),
                "relation_projection": self.relation_projection.state_dict(),
                "relation_classifier": self.relation_classifier.state_dict(),
                "span_classifier": self.span_classifier.state_dict(),
                "interaction_layer": self.interaction_layer.state_dict(),
                "entity_linker": self.entity_linker.state_dict(),
                "config": {
                    "model_name": self.model_name,
                    "max_seq_length": self.max_seq_length,
                    "max_entity_length": self.max_entity_length,
                },
            },
            f"{path}/reader_model.pt",
        )

    @classmethod
    def load(cls, path):
        """Load ReLiK reader model"""
        state_dict = torch.load(f"{path}/reader_model.pt")
        config = state_dict["config"]

        # Create model instance
        reader = cls(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"],
            max_entity_length=config["max_entity_length"],
        )

        # Load model weights
        reader.model.load_state_dict(state_dict["model"])
        reader.mention_start_classifier.load_state_dict(state_dict["mention_start_classifier"])
        reader.mention_end_classifier.load_state_dict(state_dict["mention_end_classifier"])
        reader.entity_projection.load_state_dict(state_dict["entity_projection"])
        reader.subject_projection.load_state_dict(state_dict["subject_projection"])
        reader.object_projection.load_state_dict(state_dict["object_projection"])
        reader.relation_projection.load_state_dict(state_dict["relation_projection"])
        reader.relation_classifier.load_state_dict(state_dict["relation_classifier"])

        # Load new components if they exist
        if "span_classifier" in state_dict:
            reader.span_classifier.load_state_dict(state_dict["span_classifier"])
        if "interaction_layer" in state_dict:
            reader.interaction_layer.load_state_dict(state_dict["interaction_layer"])
        if "entity_linker" in state_dict:
            reader.entity_linker.load_state_dict(state_dict["entity_linker"])

        return reader

    def quantize(self, quantization_type="int8"):
        """
        Quantize the model for faster inference.

        Args:
            quantization_type: Type of quantization (int8, fp16)

        Returns:
            Quantized model
        """
        if quantization_type == "int8":
            # Int8 quantization
            import torch.quantization

            # Prepare model for quantization
            self.model.eval()

            # Define quantization configuration
            qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(self.model, qconfig, inplace=True)

            # Calibrate with some sample data (dummy data here)
            dummy_input = torch.randint(0, 1000, (1, 128)).to(self.model.device)
            dummy_mask = torch.ones_like(dummy_input).to(self.model.device)
            self.model(dummy_input, dummy_mask)

            # Convert to quantized model
            torch.quantization.convert(self.model, inplace=True)

            print("Model quantized to INT8")

        elif quantization_type == "fp16":
            # Float16 quantization
            self.model = self.model.half()
            self.mention_start_classifier = self.mention_start_classifier.half()
            self.mention_end_classifier = self.mention_end_classifier.half()
            self.entity_projection = self.entity_projection.half()
            self.subject_projection = self.subject_projection.half()
            self.object_projection = self.object_projection.half()
            self.relation_projection = self.relation_projection.half()
            self.relation_classifier = self.relation_classifier.half()
            self.span_classifier = self.span_classifier.half()
            self.interaction_layer = self.interaction_layer.half()
            self.entity_linker = self.entity_linker.half()

            print("Model quantized to FP16")

        return self
