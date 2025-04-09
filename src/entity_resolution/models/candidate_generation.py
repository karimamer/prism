import torch
import torch.nn as nn

class EntityCandidateGenerator(nn.Module):
    def __init__(self, knowledge_base, embedding_dim=768):
        super().__init__()
        self.knowledge_base = knowledge_base

        # ReLiK-style dense retrieval
        self.entity_retriever = nn.Linear(embedding_dim, embedding_dim)

        # SpEL-style mention detection
        self.mention_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)  # B-I-O tagging
        )

        # OneNet-style candidate filtering
        self.candidate_scorer = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def detect_mentions(self, text_embeddings):
        """Extract entity mentions using SpEL approach"""
        tags = self.mention_detector(text_embeddings)
        tag_ids = torch.argmax(tags, dim=-1)

        # Extract mentions from B-I-O tags
        mentions = []
        for i, sent_tags in enumerate(tag_ids):
            current_mention = None
            for j, tag in enumerate(sent_tags):
                if tag == 1:  # B tag
                    if current_mention is not None:
                        mentions.append((i, current_mention[0], j-1))
                    current_mention = (j, None)
                elif tag == 2:  # I tag
                    continue
                elif tag == 0:  # O tag
                    if current_mention is not None:
                        mentions.append((i, current_mention[0], j-1))
                        current_mention = None
            if current_mention is not None:
                mentions.append((i, current_mention[0], len(sent_tags)-1))

        return mentions

    def forward(self, text_embeddings, top_k=100):
        # 1. Detect potential entity mentions (SpEL approach)
        mentions = self.detect_mentions(text_embeddings)

        # 2. Generate query vectors for each mention (ReLiK approach)
        mention_embeddings = []
        for sent_idx, start, end in mentions:
            # Span representation
            span_emb = text_embeddings[sent_idx, start:end+1].mean(dim=0)
            mention_embeddings.append(span_emb)

        mention_embeddings = torch.stack(mention_embeddings) if mention_embeddings else \
                             torch.zeros((0, text_embeddings.size(-1)), device=text_embeddings.device)

        # 3. Transform for retrieval
        query_vectors = self.entity_retriever(mention_embeddings)

        # 4. Retrieve candidates from knowledge base (ReLiK)
        all_candidates = []
        all_candidate_embeddings = []

        for query in query_vectors:
            candidates, candidate_embeddings = self.knowledge_base.retrieve(
                query, top_k=top_k*2
            )
            all_candidates.append(candidates)
            all_candidate_embeddings.append(candidate_embeddings)

        # 5. Filter candidates (OneNet)
        filtered_candidates = []
        for i, (mention_emb, candidates, candidate_embs) in enumerate(
            zip(mention_embeddings, all_candidates, all_candidate_embeddings)):

            # Score each candidate
            scores = []
            for cand_emb in candidate_embs:
                combined = torch.cat([mention_emb, cand_emb])
                score = self.candidate_scorer(combined)
                scores.append(score)

            scores = torch.cat(scores)

            # Keep top-k highest scoring candidates
            if len(scores) > 0:
                _, indices = scores.topk(min(top_k, len(scores)))
                filtered_candidates.append({
                    'mention': (mentions[i][1], mentions[i][2]),
                    'candidates': [candidates[idx] for idx in indices]
                })

        return filtered_candidates
