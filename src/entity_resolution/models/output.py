import torch.nn as nn

class EntityOutputFormatter(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, linked_entities, input_ids):
        """Format the final entity output"""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        formatted_entities = []
        for entity in linked_entities:
            start, end = entity['mention']
            mention_text = self.tokenizer.convert_tokens_to_string(
                tokens[start:end+1]
            )

            formatted_entity = {
                'mention': mention_text,
                'mention_span': (start, end),
                'entity_id': entity['entity']['id'],
                'entity_name': entity['entity']['name'],
                'entity_type': entity['entity']['type'],
                'confidence': entity['confidence'],
                'source': 'integrated_resolution'
            }

            formatted_entities.append(formatted_entity)

        return {
            'entities': formatted_entities,
            'text': self.tokenizer.decode(input_ids[0]),
            'token_map': tokens
        }
