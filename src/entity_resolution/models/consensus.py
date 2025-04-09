import torch
import torch.nn as nn

class EntityConsensusModule(nn.Module):
    def __init__(self, encoder_dim=768, threshold=0.6):
        super().__init__()
        # Model confidence calibration
        self.confidence_calibration = nn.Sequential(
            nn.Linear(3, 16),  # 3 score types
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

        # OneNet-inspired consensus judger
        self.consensus_threshold = threshold
        self.conflict_resolver = nn.Linear(encoder_dim*2, 1)

    def resolve_conflicts(self, entity_results, text_embeddings):
        """Resolve entity conflicts using OneNet approach"""
        # Group entities by overlapping spans
        span_groups = self._group_overlapping_spans(entity_results)

        resolved_entities = []
        for group in span_groups:
            if len(group) == 1:
                # No conflict
                resolved_entities.append(group[0])
            else:
                # Resolve conflict using context and entity embeddings
                context_start = min(result['mention'][0] for result in group) - 5
                context_end = max(result['mention'][1] for result in group) + 5
                context_start = max(0, context_start)
                context_end = min(text_embeddings.size(1) - 1, context_end)

                context_embedding = text_embeddings[:, context_start:context_end].mean(dim=1)

                best_score = -1
                best_entity = None

                for result in group:
                    top_candidate_id = max(
                        result['candidate_scores'],
                        key=lambda x: result['candidate_scores'][x]['final_score']
                    )
                    candidate_embedding = result['candidate_scores'][top_candidate_id]['candidate']['embedding']

                    conflict_score = torch.sigmoid(self.conflict_resolver(
                        torch.cat([context_embedding, candidate_embedding], dim=-1)
                    ))

                    if conflict_score.item() > best_score:
                        best_score = conflict_score.item()
                        best_entity = result

                resolved_entities.append(best_entity)

        return resolved_entities

    def _group_overlapping_spans(self, entity_results):
        """Group entities with overlapping spans"""
        groups = []
        for result in entity_results:
            start, end = result['mention']

            # Check if this span overlaps with any existing group
            found_group = False
            for group in groups:
                for existing in group:
                    e_start, e_end = existing['mention']
                    # Check overlap condition
                    if (start <= e_end and end >= e_start):
                        group.append(result)
                        found_group = True
                        break

                if found_group:
                    break

            # If no overlapping group found, create a new one
            if not found_group:
                groups.append([result])

        return groups

    def forward(self, entity_results, text_embeddings):
        # Calculate final scores using weighted combination
        for result in entity_results:
            for candidate_id, scores in result['candidate_scores'].items():
                # Collect scores from different methods
                method_scores = torch.tensor([
                    scores['relik_score'],
                    scores['atg_score'],
                    scores['unirel_score']
                ])

                # Apply confidence calibration
                weights = self.confidence_calibration(method_scores)

                # Calculate weighted final score
                final_score = (weights * method_scores).sum().item()
                scores['final_score'] = final_score

        # Resolve conflicts between overlapping entity mentions
        resolved_entities = self.resolve_conflicts(entity_results, text_embeddings)

        # Filter entities based on confidence
        linked_entities = []
        for result in resolved_entities:
            # Find best candidate
            best_candidate_id = max(
                result['candidate_scores'],
                key=lambda x: result['candidate_scores'][x]['final_score']
            )
            best_score = result['candidate_scores'][best_candidate_id]['final_score']

            if best_score >= self.consensus_threshold:
                linked_entities.append({
                    'mention': result['mention'],
                    'entity': result['candidate_scores'][best_candidate_id]['candidate'],
                    'confidence': best_score
                })

        return linked_entities
