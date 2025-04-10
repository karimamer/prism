import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Union

class EntityReader(nn.Module):
    """
    Enhanced Reader component based on SpEL and ReLiK techniques.

    This reader encodes the input text and all retrieved candidate
    entities in a single forward pass, making it much more efficient
    than processing each candidate separately.
    """
    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        max_seq_length=512,
        max_entity_length=100,
        gradient_checkpointing=True
    ):
        super().__init__()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_entity_length = max_entity_length

        # Load model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens for entity linking
        special_tokens = {
            "additional_special_tokens": [
                "<e>", "</e>",  # Entity span markers
                "<c>", "</c>",  # Candidate entity markers
                "<s>", "</s>",  # Segment markers
                "<r>", "</r>"   # Relation markers (for joint entity-relation extraction)
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
        self.entity_start_id = self.tokenizer.convert_tokens_to_ids("<e>")
        self.entity_end_id = self.tokenizer.convert_tokens_to_ids("</e>")
        self.candidate_start_id = self.tokenizer.convert_tokens_to_ids("<c>")
        self.candidate_end_id = self.tokenizer.convert_tokens_to_ids("</c>")
        self.segment_start_id = self.tokenizer.convert_tokens_to_ids("<s>")
        self.segment_end_id = self.tokenizer.convert_tokens_to_ids("</s>")

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Mention detection (SpEL approach)
        self.span_classifier = nn.Linear(self.config.hidden_size, 3)  # B-I-O tagging

        # Entity linking
        self.entity_linker = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 1)
        )

        # Additional interaction layer (UniRel approach)
        self.interaction_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def encode_text_with_candidates(
        self,
        input_text: str,
        candidate_entities: List[Dict],
        detected_mentions: Optional[List[Tuple[int, int]]] = None
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
            return_special_tokens_mask=True
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get contextualized token representations
        hidden_states = outputs.last_hidden_state

        return {
            "hidden_states": hidden_states,
            "mention_positions": mention_positions,
            "candidate_positions": candidate_positions,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping
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
            return_special_tokens_mask=False
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
                    mentions.append((current_mention[0], i-1))
                current_mention = (i, None)
            elif tag == 2:  # I - Inside entity
                continue
            elif tag == 0:  # O - Outside entity
                if current_mention is not None:
                    mentions.append((current_mention[0], i-1))
                    current_mention = None

        # Handle case where mention continues until the end
        if current_mention is not None:
            mentions.append((current_mention[0], len(predictions)-1))

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
            span_embedding = hidden_states[0, start:end+1].mean(dim=0)
            mention_embeddings.append(span_embedding)

        # If no mentions, return empty results
        if not mention_embeddings:
            return {
                "mention_entity_scores": [],
                "best_entities": []
            }

        mention_embeddings = torch.stack(mention_embeddings)

        # Get candidate entity representations
        candidate_embeddings = []
        for start, end in candidate_positions:
            # Average token embeddings in the span
            span_embedding = hidden_states[0, start:end+1].mean(dim=0)
            candidate_embeddings.append(span_embedding)

        # If no candidates, return empty results
        if not candidate_embeddings:
            return {
                "mention_entity_scores": [],
                "best_entities": []
            }

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

        return {
            "mention_entity_scores": mention_entity_scores,
            "best_entities": best_entities
        }

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
        attention_scores = torch.matmul(projected_states, projected_states.transpose(-1, -2)) / scaling_factor

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

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
            "interaction_map": interaction_map
        }

    def process_text(self, text, candidate_entities):
        """
        Process a text and link entity mentions to candidates.

        Args:
            text: Input text
            candidate_entities: List of candidate entities

        Returns:
            Dictionary with entity linking results
        """
        # Tokenize the input text
        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # Simple rule-based detection of potential entities (capitalized words)
        mentions = []

        # Split text into words
        words = text.split()

        # Find potential entity mentions (capitalized words)
        start_idx = 0
        while start_idx < len(words):
            # Skip non-capitalized words
            if not words[start_idx] or not words[start_idx][0].isupper() or words[start_idx].lower() in ['the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'and', 'or', 'but']:
                start_idx += 1
                continue

            # Found potential start of entity
            end_idx = start_idx

            # Look for multi-word entities
            while end_idx + 1 < len(words) and words[end_idx + 1][0].isupper() and words[end_idx + 1].lower() not in ['the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'and', 'or', 'but']:
                end_idx += 1

            # Found a potential entity from start_idx to end_idx
            mention_text = " ".join(words[start_idx:end_idx+1])

            # Find character positions
            char_start = text.find(mention_text)
            if char_start >= 0:
                char_end = char_start + len(mention_text) - 1

                # Convert to token positions (approximate)
                start_token = None
                end_token = None

                # Find tokens corresponding to character positions
                for i, (start, end) in enumerate(tokenized["offset_mapping"][0]):
                    if start_token is None and start <= char_start < end:
                        start_token = i
                    if end_token is None and start <= char_end < end:
                        end_token = i

                if start_token is not None and end_token is not None:
                    mentions.append((start_token, end_token))

            # Move to next potential entity
            start_idx = end_idx + 1

        # If no mentions found, return empty results
        if not mentions:
            return {"entities": [], "text": text}

        # Format the input text with mentions and candidate entities
        input_text_with_markers = text
        for start, end in sorted(mentions, reverse=True):
            # Get character positions
            char_start = tokenized["offset_mapping"][0][start][0].item()
            char_end = tokenized["offset_mapping"][0][end][1].item()

            # Add entity markers
            input_text_with_markers = (
                input_text_with_markers[:char_start] +
                " <e> " + input_text_with_markers[char_start:char_end] + " </e> " +
                input_text_with_markers[char_end:]
            )

        # Encode text with candidate entities
        encoding = self.encode_text_with_candidates(input_text_with_markers, candidate_entities)

        # If no mention positions found in the encoding, return empty results
        if not encoding["mention_positions"]:
            return {"entities": [], "text": text}

        # Forward pass through model
        with torch.no_grad():
            results = self.forward({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "mention_positions": encoding["mention_positions"],
                "candidate_positions": encoding["candidate_positions"]
            })

        # Extract linked entities
        linked_entities = []

        for i, (mention_pos, best_entity_idx) in enumerate(zip(results["mention_positions"], results["best_entities"])):
            if best_entity_idx is None:
                continue

            # Get mention text
            mention_start, mention_end = mention_pos
            mention_tokens = encoding["input_ids"][0, mention_start+1:mention_end]
            mention_text = self.tokenizer.decode(mention_tokens)

            # Get entity
            entity = candidate_entities[best_entity_idx]

            # Get score
            score = results["mention_entity_scores"][i][0][1] if results["mention_entity_scores"][i] else 0.0

            linked_entities.append({
                "mention": mention_text,
                "mention_span": (mention_start, mention_end),
                "entity_id": entity.get("id", ""),
                "entity_name": entity.get("name", ""),
                "entity_type": entity.get("type", "UNKNOWN"),
                "confidence": score
            })

        return {
            "entities": linked_entities,
            "text": text
        }

    def save(self, path):
        """Save reader model"""
        torch.save({
            "model": self.model.state_dict(),
            "span_classifier": self.span_classifier.state_dict(),
            "entity_linker": self.entity_linker.state_dict(),
            "interaction_layer": self.interaction_layer.state_dict(),
            "config": {
                "model_name": self.model_name,
                "max_seq_length": self.max_seq_length,
                "max_entity_length": self.max_entity_length
            }
        }, f"{path}/reader_model.pt")

    @classmethod
    def load(cls, path):
        """Load reader model"""
        state_dict = torch.load(f"{path}/reader_model.pt")
        config = state_dict["config"]

        # Create model instance
        reader = cls(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"],
            max_entity_length=config["max_entity_length"]
        )

        # Load model weights
        reader.model.load_state_dict(state_dict["model"])
        reader.span_classifier.load_state_dict(state_dict["span_classifier"])
        reader.entity_linker.load_state_dict(state_dict["entity_linker"])
        reader.interaction_layer.load_state_dict(state_dict["interaction_layer"])

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
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
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
            self.span_classifier = self.span_classifier.half()
            self.entity_linker = self.entity_linker.half()
            self.interaction_layer = self.interaction_layer.half()

            print("Model quantized to FP16")

        return self
