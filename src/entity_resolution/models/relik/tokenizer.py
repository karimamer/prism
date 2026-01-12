"""
ReLiK Tokenizer: Enhanced tokenizer with special token support.

Handles special tokens for marking candidates (<ST0>, <ST1>) and relations (<R0>, <R1>)
in the input sequence for proper candidate encoding.
"""

from typing import Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


class ReLiKTokenizer:
    """
    Enhanced tokenizer with special token support for ReLiK.

    Adds special tokens for:
    - Entity/passage markers: <ST0> to <ST99> (Span Token markers)
    - Relation markers: <R0> to <R99>

    These markers are used to encode candidates in the same sequence as the text,
    following the ReLiK paper's approach.
    """

    def __init__(
        self,
        base_tokenizer_name: str,
        num_special_tokens: int = 200,
        max_candidates: int = 100,
    ):
        """
        Initialize ReLiK tokenizer.

        Args:
            base_tokenizer_name: Base transformer tokenizer to use
            num_special_tokens: Total number of special tokens to add
            max_candidates: Maximum number of candidates to support
        """
        self.base_tokenizer_name = base_tokenizer_name
        self.max_candidates = max_candidates

        # Load base tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

        # Add special tokens for candidates and relations
        special_tokens = []

        # Entity/passage markers: <ST0> to <ST99>
        special_tokens.extend([f"<ST{i}>" for i in range(max_candidates)])

        # Relation markers: <R0> to <R99>
        special_tokens.extend([f"<R{i}>" for i in range(max_candidates)])

        # Add to tokenizer
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        print(f"Added {num_added} special tokens to tokenizer")

        # Create lookup dictionaries for efficient access
        self.entity_marker_ids = {
            i: self.tokenizer.convert_tokens_to_ids(f"<ST{i}>") for i in range(max_candidates)
        }

        self.relation_marker_ids = {
            i: self.tokenizer.convert_tokens_to_ids(f"<R{i}>") for i in range(max_candidates)
        }

        # Store special token strings
        self.entity_markers = [f"<ST{i}>" for i in range(max_candidates)]
        self.relation_markers = [f"<R{i}>" for i in range(max_candidates)]

    def encode_with_candidates(
        self,
        text: str,
        candidates: list[str],
        max_length: int = 1024,
        return_tensors: str = "pt",
        return_offsets_mapping: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text with candidate markers.

        Format: text [SEP] <ST0> cand0 <ST1> cand1 ...

        Args:
            text: Input text
            candidates: List of candidate descriptions
            max_length: Maximum sequence length
            return_tensors: Return format ('pt' for PyTorch)
            return_offsets_mapping: Whether to return character offset mappings

        Returns:
            Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - marker_positions: Positions of <STi> markers
                - text_end: Position where text ends (SEP token)
                - num_candidates: Number of candidates
                - offset_mapping: Character offsets (if requested)
        """
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"Too many candidates ({len(candidates)}), max is {self.max_candidates}"
            )

        # Build input string with markers
        parts = [text, " [SEP]"]

        for i, cand in enumerate(candidates):
            parts.append(f" <ST{i}> {cand}")

        full_text = "".join(parts)

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors,
            return_offsets_mapping=return_offsets_mapping,
        )

        # Find marker positions
        input_ids = encoded["input_ids"][0] if return_tensors == "pt" else encoded["input_ids"]
        marker_positions = []

        for i in range(len(candidates)):
            marker_id = self.entity_marker_ids[i]
            # Find position of this marker
            if return_tensors == "pt":
                positions = (input_ids == marker_id).nonzero(as_tuple=True)[0]
            else:
                positions = [j for j, token_id in enumerate(input_ids) if token_id == marker_id]

            if len(positions) > 0:
                marker_positions.append(
                    int(positions[0]) if return_tensors == "pt" else positions[0]
                )
            else:
                marker_positions.append(-1)  # Marker not found (likely truncated)

        # Find SEP position (end of text)
        sep_id = self.tokenizer.sep_token_id
        if return_tensors == "pt":
            sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0]
        else:
            sep_positions = [j for j, token_id in enumerate(input_ids) if token_id == sep_id]

        text_end = int(sep_positions[0]) if len(sep_positions) > 0 else len(input_ids)

        # Add metadata
        encoded["marker_positions"] = (
            torch.tensor(marker_positions) if return_tensors == "pt" else marker_positions
        )
        encoded["text_end"] = text_end
        encoded["num_candidates"] = len(candidates)

        return encoded

    def encode_with_relations(
        self,
        text: str,
        relation_types: list[str],
        max_length: int = 1024,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """
        Encode text with relation type markers.

        Format: text [SEP] <R0> relation0 <R1> relation1 ...

        Args:
            text: Input text
            relation_types: List of relation type descriptions
            max_length: Maximum sequence length
            return_tensors: Return format

        Returns:
            Dictionary with encoded data and metadata
        """
        if len(relation_types) > self.max_candidates:
            raise ValueError(
                f"Too many relations ({len(relation_types)}), max is {self.max_candidates}"
            )

        # Build input string with relation markers
        parts = [text, " [SEP]"]

        for i, rel in enumerate(relation_types):
            parts.append(f" <R{i}> {rel}")

        full_text = "".join(parts)

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        # Find relation marker positions
        input_ids = encoded["input_ids"][0] if return_tensors == "pt" else encoded["input_ids"]
        marker_positions = []

        for i in range(len(relation_types)):
            marker_id = self.relation_marker_ids[i]
            if return_tensors == "pt":
                positions = (input_ids == marker_id).nonzero(as_tuple=True)[0]
            else:
                positions = [j for j, token_id in enumerate(input_ids) if token_id == marker_id]

            if len(positions) > 0:
                marker_positions.append(
                    int(positions[0]) if return_tensors == "pt" else positions[0]
                )
            else:
                marker_positions.append(-1)

        # Find SEP position
        sep_id = self.tokenizer.sep_token_id
        if return_tensors == "pt":
            sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0]
        else:
            sep_positions = [j for j, token_id in enumerate(input_ids) if token_id == sep_id]

        text_end = int(sep_positions[0]) if len(sep_positions) > 0 else len(input_ids)

        encoded["marker_positions"] = (
            torch.tensor(marker_positions) if return_tensors == "pt" else marker_positions
        )
        encoded["text_end"] = text_end
        encoded["num_relations"] = len(relation_types)

        return encoded

    def decode(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size including special tokens."""
        return len(self.tokenizer)

    def __call__(self, *args, **kwargs):
        """Forward calls to underlying tokenizer."""
        return self.tokenizer(*args, **kwargs)

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id

    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID."""
        return self.tokenizer.sep_token_id

    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID."""
        return self.tokenizer.cls_token_id

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)


__all__ = ["ReLiKTokenizer"]
