import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

class EntityFocusedEncoder(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", entity_knowledge_dim=256):
        super().__init__()
        # Load DeBERTa-v3 configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Entity-specific projections
        self.entity_projector = nn.Linear(
            self.config.hidden_size,
            entity_knowledge_dim
        )

        # Entity type embedding layer
        self.entity_type_embedder = nn.Embedding(
            num_embeddings=50,
            embedding_dim=entity_knowledge_dim
        )

        # Knowledge integration
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=8
        )

    def forward(self, input_ids, attention_mask, entity_knowledge=None):
        # DeBERTa encoding
        outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = outputs.last_hidden_state

        # Knowledge enhancement
        if entity_knowledge is not None:
            entity_proj = self.entity_projector(entity_knowledge)
            text_embeddings, _ = self.knowledge_attention(
                text_embeddings, entity_proj, entity_proj
            )

        return text_embeddings
