import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

class GenerationState(Enum):
    """States for constrained decoding in ATG."""
    START = "start"
    ENTITY = "entity" 
    SEPARATOR = "separator"
    RELATION_HEAD = "relation_head"
    RELATION_TAIL = "relation_tail"
    RELATION_TYPE = "relation_type"
    END = "end"

@dataclass
class SpanRepresentation:
    """Representation of a text span with boundaries and type."""
    start: int
    end: int
    entity_type: str
    text: str
    embedding: Optional[torch.Tensor] = None

@dataclass 
class ATGOutput:
    """Output from ATG model generation."""
    entities: List[SpanRepresentation]
    relations: List[Tuple[SpanRepresentation, SpanRepresentation, str]]
    linearized_sequence: List[str]
    generation_scores: List[float]

class DynamicVocabulary:
    """Dynamic vocabulary for ATG with spans, relations, and special tokens."""
    
    def __init__(self, max_span_length: int = 12, relation_types: List[str] = None):
        self.max_span_length = max_span_length
        self.relation_types = relation_types or []
        self.special_tokens = ["<START>", "<SEP>", "<END>"]
        self.entity_types = ["PER", "ORG", "LOC", "MISC"]  # Default entity types
        
    def construct_vocabulary(
        self, 
        sequence_length: int, 
        entity_types: List[str] = None
    ) -> Dict[str, int]:
        """
        Construct dynamic vocabulary for given sequence.
        
        Args:
            sequence_length: Length of input sequence
            entity_types: Entity types for this instance
            
        Returns:
            Vocabulary mapping from tokens to indices
        """
        if entity_types:
            self.entity_types = entity_types
            
        vocab = {}
        idx = 0
        
        # Add special tokens
        for token in self.special_tokens:
            vocab[token] = idx
            idx += 1
            
        # Add span tokens: (start, end, type)
        for start in range(sequence_length):
            for end in range(start, min(start + self.max_span_length, sequence_length)):
                for entity_type in self.entity_types:
                    span_token = f"({start},{end},{entity_type})"
                    vocab[span_token] = idx
                    idx += 1
                    
        # Add relation types
        for rel_type in self.relation_types:
            vocab[rel_type] = idx
            idx += 1
            
        return vocab
        
    def get_vocabulary_size(self, sequence_length: int) -> int:
        """Get size of dynamic vocabulary for given sequence length."""
        special_count = len(self.special_tokens)
        span_count = 0
        
        for start in range(sequence_length):
            for end in range(start, min(start + self.max_span_length, sequence_length)):
                span_count += len(self.entity_types)
                
        relation_count = len(self.relation_types)
        
        return special_count + span_count + relation_count

class SpanRepresentationLayer(nn.Module):
    """Layer for computing span representations from token embeddings."""
    
    def __init__(self, hidden_size: int, entity_types: List[str]):
        super().__init__()
        self.hidden_size = hidden_size
        self.entity_types = entity_types
        
        # Projection matrices for each entity type
        self.type_projections = nn.ModuleDict({
            entity_type: nn.Linear(hidden_size * 2, hidden_size)
            for entity_type in entity_types
        })
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        max_span_length: int = 12
    ) -> torch.Tensor:
        """
        Compute span representations for all possible spans.
        
        Args:
            hidden_states: Token representations (batch_size, seq_len, hidden_size)
            max_span_length: Maximum span length to consider
            
        Returns:
            Span embeddings (batch_size, num_spans, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        span_embeddings = []
        
        for start in range(seq_len):
            for end in range(start, min(start + max_span_length, seq_len)):
                for entity_type in self.entity_types:
                    # Concatenate start and end token representations
                    start_repr = hidden_states[:, start, :]  # (batch_size, hidden_size)
                    end_repr = hidden_states[:, end, :]      # (batch_size, hidden_size)
                    
                    # Concatenate start and end representations
                    concat_repr = torch.cat([start_repr, end_repr], dim=-1)  # (batch_size, 2*hidden_size)
                    
                    # Project through type-specific layer
                    span_emb = self.type_projections[entity_type](concat_repr)  # (batch_size, hidden_size)
                    span_embeddings.append(span_emb)
                    
        # Stack all span embeddings
        span_embeddings = torch.stack(span_embeddings, dim=1)  # (batch_size, num_spans, hidden_size)
        
        return span_embeddings

class ATGDecoder(nn.Module):
    """Autoregressive decoder for ATG with pointing mechanism."""
    
    def __init__(
        self, 
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Position embedding
        self.position_embedding = nn.Embedding(512, hidden_size)
        
        # Structural embeddings for different generation phases
        self.structural_embeddings = nn.ModuleDict({
            "node": nn.Parameter(torch.randn(hidden_size)),
            "head": nn.Parameter(torch.randn(hidden_size)),
            "tail": nn.Parameter(torch.randn(hidden_size)),
            "relation": nn.Parameter(torch.randn(hidden_size))
        })
        
    def forward(
        self,
        input_embeddings: torch.Tensor,
        encoder_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through ATG decoder.
        
        Args:
            input_embeddings: Embedded input tokens (batch_size, tgt_len, hidden_size)
            encoder_outputs: Encoder outputs (batch_size, src_len, hidden_size)
            attention_mask: Source attention mask
            tgt_mask: Target causal mask
            
        Returns:
            Decoder outputs (batch_size, tgt_len, hidden_size)
        """
        return self.transformer_decoder(
            tgt=input_embeddings,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=attention_mask
        )

class ATGModel(nn.Module):
    """
    Autoregressive Text-to-Graph (ATG) model for joint entity and relation extraction.
    
    This model generates a linearized graph representation where:
    - Nodes represent entity spans (start, end, type)
    - Edges represent relation triplets (head, tail, relation_type)
    
    The model uses a transformer encoder-decoder architecture with:
    - Dynamic vocabulary of spans and relation types
    - Pointing mechanism for grounding in original text
    - Constrained decoding for well-formed output
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_span_length: int = 12,
        entity_types: List[str] = None,
        relation_types: List[str] = None,
        max_seq_length: int = 512,
        decoder_layers: int = 6
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_span_length = max_span_length
        self.max_seq_length = max_seq_length
        self.entity_types = entity_types or ["PER", "ORG", "LOC", "MISC"]
        self.relation_types = relation_types or ["Work_For", "Located_In", "Born_In", "Live_In", "Based_In"]
        
        # Load encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": ["<START>", "<SEP>", "<END>"]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Get special token IDs
        self.start_token_id = self.tokenizer.convert_tokens_to_ids("<START>")
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids("<SEP>")
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<END>")
        
        # Span representation layer
        self.span_layer = SpanRepresentationLayer(self.config.hidden_size, self.entity_types)
        
        # Dynamic vocabulary
        self.dynamic_vocab = DynamicVocabulary(max_span_length, self.relation_types)
        
        # Decoder
        self.decoder = ATGDecoder(
            hidden_size=self.config.hidden_size,
            num_layers=decoder_layers
        )
        
        # Relation type embeddings (learned during training)
        self.relation_embeddings = nn.Embedding(
            len(self.relation_types), self.config.hidden_size
        )
        
        # Special token embeddings
        self.special_token_embeddings = nn.Embedding(3, self.config.hidden_size)  # START, SEP, END
        
    def encode_input(self, input_text: str) -> Dict[str, torch.Tensor]:
        """
        Encode input text and compute span representations.
        
        Args:
            input_text: Input text string
            
        Returns:
            Dictionary with encoder outputs and span representations
        """
        # Tokenize input
        encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.encoder.device)
        attention_mask = encoding["attention_mask"].to(self.encoder.device)
        
        # Encode with transformer
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Compute span representations
        span_embeddings = self.span_layer(hidden_states, self.max_span_length)
        
        # Get actual sequence length (excluding padding)
        seq_len = attention_mask.sum(dim=1).item()
        
        return {
            "hidden_states": hidden_states,
            "span_embeddings": span_embeddings,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "sequence_length": seq_len
        }
        
    def construct_dynamic_vocabulary_matrix(
        self, 
        span_embeddings: torch.Tensor, 
        sequence_length: int
    ) -> torch.Tensor:
        """
        Construct dynamic vocabulary matrix for decoding.
        
        Args:
            span_embeddings: Precomputed span embeddings
            sequence_length: Actual sequence length
            
        Returns:
            Vocabulary matrix (vocab_size, hidden_size)
        """
        vocab_embeddings = []
        
        # Add special token embeddings
        vocab_embeddings.append(self.special_token_embeddings.weight)  # (3, hidden_size)
        
        # Add span embeddings (filter by actual sequence length)
        span_idx = 0
        for start in range(sequence_length):
            for end in range(start, min(start + self.max_span_length, sequence_length)):
                for _ in self.entity_types:
                    if span_idx < span_embeddings.shape[1]:
                        vocab_embeddings.append(span_embeddings[0, span_idx, :].unsqueeze(0))
                        span_idx += 1
                        
        # Add relation type embeddings
        vocab_embeddings.append(self.relation_embeddings.weight)  # (num_relations, hidden_size)
        
        # Concatenate all embeddings
        vocab_matrix = torch.cat(vocab_embeddings, dim=0)  # (vocab_size, hidden_size)
        
        return vocab_matrix
        
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.encoder.device)
        
    def generate_linearized_sequence(
        self,
        input_text: str,
        max_length: int = 50,
        use_constrained_decoding: bool = True
    ) -> ATGOutput:
        """
        Generate linearized graph sequence from input text.
        
        Args:
            input_text: Input text string
            max_length: Maximum generation length
            use_constrained_decoding: Whether to use state-based constraints
            
        Returns:
            ATGOutput with entities, relations, and generation info
        """
        # Encode input
        encoding_result = self.encode_input(input_text)
        encoder_outputs = encoding_result["hidden_states"]
        span_embeddings = encoding_result["span_embeddings"]
        sequence_length = encoding_result["sequence_length"]
        
        # Construct dynamic vocabulary
        vocab_matrix = self.construct_dynamic_vocabulary_matrix(span_embeddings, sequence_length)
        vocab_mapping = self.dynamic_vocab.construct_vocabulary(sequence_length, self.entity_types)
        reverse_vocab = {v: k for k, v in vocab_mapping.items()}
        
        # Initialize generation
        generated_tokens = [self.start_token_id]
        generated_scores = []
        current_state = GenerationState.START
        
        # Generation loop
        for step in range(max_length):
            # Prepare decoder input
            if len(generated_tokens) == 1:
                # First step: just start token
                decoder_input = torch.tensor([[0]], device=self.encoder.device)  # Index of <START> 
            else:
                decoder_input = torch.tensor([generated_tokens[:-1]], device=self.encoder.device)
                
            # Embed decoder input using vocabulary matrix
            if decoder_input.numel() > 0:
                input_embeddings = vocab_matrix[decoder_input]  # (1, seq_len, hidden_size)
                
                # Add positional embeddings
                positions = torch.arange(input_embeddings.size(1), device=self.encoder.device)
                pos_embeddings = self.decoder.position_embedding(positions).unsqueeze(0)
                input_embeddings = input_embeddings + pos_embeddings
                
                # Create causal mask
                tgt_mask = self.create_causal_mask(input_embeddings.size(1))
                
                # Decoder forward pass
                decoder_outputs = self.decoder(
                    input_embeddings=input_embeddings,
                    encoder_outputs=encoder_outputs,
                    tgt_mask=tgt_mask
                )
                
                # Get last position output
                last_hidden = decoder_outputs[0, -1, :]  # (hidden_size,)
                
                # Compute logits using vocabulary matrix
                logits = torch.matmul(last_hidden, vocab_matrix.T)  # (vocab_size,)
                
                # Apply constraints if enabled
                if use_constrained_decoding:
                    logits = self.apply_generation_constraints(logits, current_state, vocab_mapping)
                    
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token_idx = torch.argmax(probs).item()
                next_token_score = probs[next_token_idx].item()
                
                generated_tokens.append(next_token_idx)
                generated_scores.append(next_token_score)
                
                # Update state and check for end
                next_token_str = reverse_vocab.get(next_token_idx, "<UNK>")
                current_state = self.update_generation_state(current_state, next_token_str)
                
                if next_token_str == "<END>" or step >= max_length - 1:
                    break
            else:
                break
                
        # Parse generated sequence into entities and relations
        generated_sequence = [reverse_vocab.get(idx, "<UNK>") for idx in generated_tokens[1:]]  # Skip <START>
        entities, relations = self.parse_generated_sequence(generated_sequence, input_text)
        
        return ATGOutput(
            entities=entities,
            relations=relations,
            linearized_sequence=generated_sequence,
            generation_scores=generated_scores
        )
        
    def apply_generation_constraints(
        self, 
        logits: torch.Tensor, 
        current_state: GenerationState,
        vocab_mapping: Dict[str, int]
    ) -> torch.Tensor:
        """
        Apply state-based constraints to generation logits.
        
        Args:
            logits: Raw logits (vocab_size,)
            current_state: Current generation state
            vocab_mapping: Vocabulary mapping
            
        Returns:
            Constrained logits
        """
        # Create mask for invalid tokens
        mask = torch.full_like(logits, float('-inf'))
        
        if current_state == GenerationState.START:
            # Can only generate entity spans
            for token, idx in vocab_mapping.items():
                if token.startswith('(') and ',' in token and token.endswith(')'):
                    mask[idx] = 0
                    
        elif current_state == GenerationState.ENTITY:
            # Can generate more entities or separator
            for token, idx in vocab_mapping.items():
                if token.startswith('(') and ',' in token and token.endswith(')'):
                    mask[idx] = 0
                elif token == "<SEP>":
                    mask[idx] = 0
                    
        elif current_state == GenerationState.SEPARATOR:
            # Can only generate entity spans (for relations)
            for token, idx in vocab_mapping.items():
                if token.startswith('(') and ',' in token and token.endswith(')'):
                    mask[idx] = 0
                    
        elif current_state == GenerationState.RELATION_HEAD:
            # Can only generate entity spans (tail entities)
            for token, idx in vocab_mapping.items():
                if token.startswith('(') and ',' in token and token.endswith(')'):
                    mask[idx] = 0
                    
        elif current_state == GenerationState.RELATION_TAIL:
            # Can only generate relation types
            for token, idx in vocab_mapping.items():
                if token in self.relation_types:
                    mask[idx] = 0
                    
        elif current_state == GenerationState.RELATION_TYPE:
            # Can generate more relations, entity spans, or end
            for token, idx in vocab_mapping.items():
                if token.startswith('(') and ',' in token and token.endswith(')'):
                    mask[idx] = 0
                elif token == "<END>":
                    mask[idx] = 0
                    
        return logits + mask
        
    def update_generation_state(
        self, 
        current_state: GenerationState, 
        token: str
    ) -> GenerationState:
        """Update generation state based on current token."""
        if current_state == GenerationState.START:
            if token.startswith('('):
                return GenerationState.ENTITY
                
        elif current_state == GenerationState.ENTITY:
            if token == "<SEP>":
                return GenerationState.SEPARATOR
            elif token.startswith('('):
                return GenerationState.ENTITY
                
        elif current_state == GenerationState.SEPARATOR:
            if token.startswith('('):
                return GenerationState.RELATION_HEAD
                
        elif current_state == GenerationState.RELATION_HEAD:
            if token.startswith('('):
                return GenerationState.RELATION_TAIL
                
        elif current_state == GenerationState.RELATION_TAIL:
            if token in self.relation_types:
                return GenerationState.RELATION_TYPE
                
        elif current_state == GenerationState.RELATION_TYPE:
            if token.startswith('('):
                return GenerationState.RELATION_HEAD
            elif token == "<END>":
                return GenerationState.END
                
        return current_state
        
    def parse_generated_sequence(
        self, 
        sequence: List[str], 
        input_text: str
    ) -> Tuple[List[SpanRepresentation], List[Tuple[SpanRepresentation, SpanRepresentation, str]]]:
        """
        Parse generated sequence into entities and relations.
        
        Args:
            sequence: Generated token sequence
            input_text: Original input text
            
        Returns:
            Tuple of (entities, relations)
        """
        entities = []
        relations = []
        
        # Find separator
        sep_idx = -1
        for i, token in enumerate(sequence):
            if token == "<SEP>":
                sep_idx = i
                break
                
        if sep_idx == -1:
            # No separator found, treat all as entities
            entity_tokens = sequence
            relation_tokens = []
        else:
            entity_tokens = sequence[:sep_idx]
            relation_tokens = sequence[sep_idx + 1:]
            
        # Parse entities
        entity_map = {}
        for token in entity_tokens:
            if token.startswith('(') and token.endswith(')'):
                try:
                    # Parse (start,end,type) format
                    content = token[1:-1]  # Remove parentheses
                    parts = content.split(',')
                    if len(parts) == 3:
                        start, end, entity_type = parts
                        start, end = int(start), int(end)
                        
                        # Extract text span
                        words = input_text.split()
                        if start < len(words) and end < len(words):
                            span_text = ' '.join(words[start:end+1])
                            
                            span_repr = SpanRepresentation(
                                start=start,
                                end=end,
                                entity_type=entity_type,
                                text=span_text
                            )
                            entities.append(span_repr)
                            entity_map[token] = span_repr
                            
                except (ValueError, IndexError):
                    continue
                    
        # Parse relations
        i = 0
        while i < len(relation_tokens):
            if i + 2 < len(relation_tokens):
                head_token = relation_tokens[i]
                tail_token = relation_tokens[i + 1]
                relation_type = relation_tokens[i + 2]
                
                if (head_token in entity_map and 
                    tail_token in entity_map and 
                    relation_type in self.relation_types):
                    
                    relations.append((
                        entity_map[head_token],
                        entity_map[tail_token],
                        relation_type
                    ))
                    i += 3
                else:
                    i += 1
            else:
                i += 1
                
        return entities, relations
        
    def compute_loss(
        self,
        input_text: str,
        target_sequence: List[str]
    ) -> torch.Tensor:
        """
        Compute training loss for given input and target sequence.
        
        Args:
            input_text: Input text string
            target_sequence: Target linearized sequence
            
        Returns:
            Cross-entropy loss
        """
        # Encode input
        encoding_result = self.encode_input(input_text)
        encoder_outputs = encoding_result["hidden_states"]
        span_embeddings = encoding_result["span_embeddings"]
        sequence_length = encoding_result["sequence_length"]
        
        # Construct dynamic vocabulary
        vocab_matrix = self.construct_dynamic_vocabulary_matrix(span_embeddings, sequence_length)
        vocab_mapping = self.dynamic_vocab.construct_vocabulary(sequence_length, self.entity_types)
        
        # Convert target sequence to indices
        target_indices = []
        for token in target_sequence:
            if token in vocab_mapping:
                target_indices.append(vocab_mapping[token])
            else:
                # Handle unknown tokens (shouldn't happen in well-formed training data)
                target_indices.append(vocab_mapping.get("<END>", 0))
                
        target_tensor = torch.tensor(target_indices, device=self.encoder.device)
        
        # Prepare decoder input (shifted right)
        decoder_input_indices = [vocab_mapping["<START>"]] + target_indices[:-1]
        decoder_input = torch.tensor([decoder_input_indices], device=self.encoder.device)
        
        # Embed decoder input
        input_embeddings = vocab_matrix[decoder_input]  # (1, seq_len, hidden_size)
        
        # Add positional embeddings
        positions = torch.arange(input_embeddings.size(1), device=self.encoder.device)
        pos_embeddings = self.decoder.position_embedding(positions).unsqueeze(0)
        input_embeddings = input_embeddings + pos_embeddings
        
        # Create causal mask
        tgt_mask = self.create_causal_mask(input_embeddings.size(1))
        
        # Decoder forward pass
        decoder_outputs = self.decoder(
            input_embeddings=input_embeddings,
            encoder_outputs=encoder_outputs,
            tgt_mask=tgt_mask
        )
        
        # Compute logits
        logits = torch.matmul(decoder_outputs[0], vocab_matrix.T)  # (seq_len, vocab_size)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_tensor.view(-1)
        )
        
        return loss
        
    def forward(
        self,
        input_text: str,
        target_sequence: Optional[List[str]] = None,
        max_length: int = 50
    ) -> Union[torch.Tensor, ATGOutput]:
        """
        Forward pass - training or inference mode.
        
        Args:
            input_text: Input text string
            target_sequence: Target sequence for training (optional)
            max_length: Maximum generation length for inference
            
        Returns:
            Loss tensor (training) or ATGOutput (inference)
        """
        if target_sequence is not None:
            # Training mode
            return self.compute_loss(input_text, target_sequence)
        else:
            # Inference mode
            return self.generate_linearized_sequence(input_text, max_length)