import torch
import torch.nn as nn

class EntityResolutionProcessor(nn.Module):
    def __init__(self, encoder_dim=768):
        super().__init__()
        # ReLiK reader component
        self.relik_reader = nn.Sequential(
            nn.Linear(encoder_dim*2, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 1)
        )

        # ATG entity representation component
        self.atg_entity_encoder = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=8,
            dim_feedforward=encoder_dim*4
        )

        # UniRel entity correlation modeling
        self.unirel_interaction = nn.Sequential(
            nn.Linear(encoder_dim*2, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        # Cross-model attention for information exchange
        self.cross_model_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=8
        )

    def forward(self, text_embeddings, entity_candidates):
        results = []

        for candidate_set in entity_candidates:
            mention_start, mention_end = candidate_set['mention']
            mention_embedding = text_embeddings[:, mention_start:mention_end+1].mean(dim=1)

            candidate_scores = {}

            # Process each candidate with multiple methods
            for candidate in candidate_set['candidates']:
                candidate_id = candidate['id']
                candidate_embedding = candidate['embedding']

                # 1. ReLiK reader scoring
                relik_score = self.relik_reader(
                    torch.cat([mention_embedding, candidate_embedding], dim=-1)
                )

                # 2. ATG entity representation enhancement
                enhanced_embedding = self.atg_entity_encoder(
                    torch.stack([mention_embedding, candidate_embedding])
                )
                atg_score = torch.cosine_similarity(
                    enhanced_embedding[0], enhanced_embedding[1], dim=-1
                )

                # 3. UniRel entity correlation
                context_size = 10  # Consider 10 tokens around mention
                context_start = max(0, mention_start - context_size)
                context_end = min(text_embeddings.size(1) - 1, mention_end + context_size)
                context_embedding = text_embeddings[:, context_start:context_end].mean(dim=1)

                interaction = self.unirel_interaction(
                    torch.cat([context_embedding, candidate_embedding], dim=-1)
                )
                unirel_score = torch.sigmoid(interaction.mean(dim=-1))

                # Store all scores
                candidate_scores[candidate_id] = {
                    'relik_score': relik_score.item(),
                    'atg_score': atg_score.item(),
                    'unirel_score': unirel_score.item(),
                    'candidate': candidate
                }

            results.append({
                'mention': (mention_start, mention_end),
                'candidate_scores': candidate_scores
            })

        return results
