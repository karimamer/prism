"""
Tests for improved ReLiK implementation.

Tests the new features:
1. Special token handling with ReLiKTokenizer
2. Improved span detection with candidate encoding
3. Complete entity linking pipeline
4. Contrastive loss for retriever training
"""

import torch

from src.entity_resolution.models.relik import (
    CompleteEntityLinker,
    ImprovedReLiKReader,
    ReLiKRetriever,
    ReLiKTokenizer,
)


def test_relik_tokenizer():
    """Test ReLiKTokenizer with special tokens."""
    print("\n=== Testing ReLiKTokenizer ===")

    tokenizer = ReLiKTokenizer("bert-base-uncased")

    # Test basic properties
    assert tokenizer.max_candidates == 100
    assert len(tokenizer.entity_markers) == 100
    assert len(tokenizer.relation_markers) == 100

    # Test encoding with candidates
    text = "Apple was founded by Steve Jobs"
    candidates = ["Apple Inc. is a technology company", "Steve Jobs was an entrepreneur"]

    encoded = tokenizer.encode_with_candidates(text, candidates, max_length=128)

    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert "marker_positions" in encoded
    assert "text_end" in encoded
    assert "num_candidates" in encoded

    assert encoded["num_candidates"] == 2
    assert len(encoded["marker_positions"]) == 2

    # Check that markers are found
    marker_positions = encoded["marker_positions"]
    if isinstance(marker_positions, torch.Tensor):
        marker_positions = marker_positions.tolist()

    # At least one marker should be found
    assert any(pos >= 0 for pos in marker_positions)

    print(f"✓ Text end position: {encoded['text_end']}")
    print(f"✓ Marker positions: {marker_positions}")
    print(f"✓ Vocabulary size: {tokenizer.get_vocab_size()}")

    # Test encoding with relations
    relation_types = ["founded_by", "located_in"]
    encoded_rel = tokenizer.encode_with_relations(text, relation_types, max_length=128)

    assert "marker_positions" in encoded_rel
    assert encoded_rel["num_relations"] == 2

    print("✓ ReLiKTokenizer tests passed")


def test_retriever_contrastive_loss():
    """Test retriever with contrastive loss."""
    print("\n=== Testing Retriever Contrastive Loss ===")

    # Create retriever with NLL loss
    retriever = ReLiKRetriever(
        model_name="bert-base-uncased",
        max_query_length=32,
        max_passage_length=32,
        loss_type="nll",
        temperature=1.0,
    )

    batch_size = 4
    seq_len = 32

    # Create dummy batch
    query_ids = torch.randint(0, 1000, (batch_size, seq_len))
    query_mask = torch.ones(batch_size, seq_len)
    positive_ids = torch.randint(0, 1000, (batch_size, seq_len))
    positive_mask = torch.ones(batch_size, seq_len)

    # Test with in-batch negatives
    outputs = retriever.forward_train(
        query_ids,
        query_mask,
        positive_ids,
        positive_mask,
    )

    assert "loss" in outputs
    assert "logits" in outputs
    assert "query_emb" in outputs
    assert "pos_emb" in outputs

    assert outputs["loss"].requires_grad
    assert outputs["logits"].shape == (batch_size, batch_size)
    assert outputs["query_emb"].shape == (batch_size, retriever.hidden_size)

    print(f"✓ Loss value: {outputs['loss'].item():.4f}")
    print(f"✓ Logits shape: {outputs['logits'].shape}")

    # Test with explicit negatives
    num_negatives = 3
    negative_ids = torch.randint(0, 1000, (batch_size, num_negatives, seq_len))
    negative_mask = torch.ones(batch_size, num_negatives, seq_len)

    outputs_neg = retriever.forward_train(
        query_ids,
        query_mask,
        positive_ids,
        positive_mask,
        negative_ids,
        negative_mask,
    )

    assert outputs_neg["loss"].requires_grad
    assert outputs_neg["logits"].shape == (batch_size, 1 + num_negatives)

    print(f"✓ Loss with negatives: {outputs_neg['loss'].item():.4f}")
    print(f"✓ Logits with negatives shape: {outputs_neg['logits'].shape}")

    # Test BCE loss
    retriever_bce = ReLiKRetriever(
        model_name="bert-base-uncased",
        loss_type="bce",
    )

    outputs_bce = retriever_bce.forward_train(
        query_ids,
        query_mask,
        positive_ids,
        positive_mask,
    )

    assert outputs_bce["loss"].requires_grad
    print(f"✓ BCE loss value: {outputs_bce['loss'].item():.4f}")

    print("✓ Retriever contrastive loss tests passed")


def test_improved_reader_span_detection():
    """Test improved reader with proper span detection."""
    print("\n=== Testing Improved Reader Span Detection ===")

    reader = ImprovedReLiKReader(
        model_name="bert-base-uncased",
        max_seq_length=128,
        num_entity_types=4,
        use_entity_linking=True,
        max_span_length=10,
    )

    # Test span detection without candidates
    text = "Apple was founded by Steve Jobs in California"
    spans = reader.predict_spans(text, span_threshold=0.3)

    print(f"✓ Detected {len(spans)} spans")
    for start, end, span_text in spans:
        print(f"  - [{start}:{end}] '{span_text}'")

    # Test with candidates
    candidates = [
        {"id": "Q312", "text": "Apple Inc. is a technology company", "name": "Apple Inc."},
        {
            "id": "Q7564",
            "text": "Steve Jobs was an entrepreneur",
            "name": "Steve Jobs",
        },
        {
            "id": "Q99",
            "text": "California is a U.S. state",
            "name": "California",
        },
    ]

    linked_spans = reader.predict_spans_with_linking(
        text,
        candidates,
        span_threshold=0.3,
        entity_threshold=0.1,
        top_k=3,
    )

    print(f"✓ Detected {len(linked_spans)} spans with linking")
    for span in linked_spans:
        print(f"  - [{span['start']}:{span['end']}] '{span['text']}'")
        print(f"    Span score: {span['span_score']:.3f}")
        if span["best_entity"]:
            print(
                f"    Best entity: {span['best_entity']['entity_name']} "
                f"(score: {span['best_entity']['score']:.3f})"
            )
        print(f"    Total candidates: {len(span['candidates'])}")

    print("✓ Improved reader span detection tests passed")


def test_complete_entity_linker():
    """Test complete entity linking pipeline."""
    print("\n=== Testing Complete Entity Linker ===")

    # Create components
    retriever = ReLiKRetriever(
        model_name="bert-base-uncased",
        max_query_length=64,
        max_passage_length=64,
    )

    reader = ImprovedReLiKReader(
        model_name="bert-base-uncased",
        max_seq_length=256,
        use_entity_linking=True,
    )

    linker = CompleteEntityLinker(retriever, reader, device="cpu")

    # Create knowledge base
    knowledge_base = {
        "Q312": {
            "text": "Apple Inc. is an American multinational technology company",
            "name": "Apple Inc.",
            "type": "ORG",
        },
        "Q7564": {
            "text": "Steve Jobs was an American entrepreneur and co-founder of Apple",
            "name": "Steve Jobs",
            "type": "PER",
        },
        "Q99": {
            "text": "California is a state in the Western United States",
            "name": "California",
            "type": "LOC",
        },
        "Q19860": {
            "text": "Stanford University is a private research university",
            "name": "Stanford University",
            "type": "ORG",
        },
    }

    # Build retriever index
    retriever.build_index(knowledge_base, batch_size=2)
    print("✓ Built retriever index")

    # Test entity linking
    text = "Apple was founded by Steve Jobs in California"

    linked_entities = linker.link_entities_end_to_end(
        text,
        knowledge_base,
        top_k_retrieval=4,
        top_k_linking=3,
        span_threshold=0.3,
        entity_threshold=0.1,
    )

    print(f"✓ Linked {len(linked_entities)} entities")

    for entity in linked_entities:
        print(f"\n  Span: '{entity['text']}' [{entity['start']}:{entity['end']}]")
        print(f"  Span confidence: {entity['span_score']:.3f}")

        if entity["best_entity"]:
            best = entity["best_entity"]
            print(f"  Best entity: {best['entity_name']} ({best['entity_id']})")
            print(f"  Linking confidence: {best['score']:.3f}")

        print(f"  Top candidates ({len(entity['candidates'])}):")
        for cand in entity["candidates"][:3]:
            print(f"    - {cand['entity_name']}: {cand['score']:.3f}")

    # Test batch processing
    texts = [
        "Apple was founded by Steve Jobs",
        "California has many universities including Stanford",
    ]

    batch_results = linker.link_entities_batch(
        texts,
        knowledge_base,
        batch_size=2,
        top_k_retrieval=4,
        span_threshold=0.3,
    )

    print(f"\n✓ Processed batch of {len(texts)} texts")
    print(f"  Results: {len(batch_results)} outputs")
    for i, result in enumerate(batch_results):
        print(f"  Text {i + 1}: {len(result)} entities linked")

    print("\n✓ Complete entity linker tests passed")


def test_tokenizer_special_tokens():
    """Test that special tokens are properly added and used."""
    print("\n=== Testing Special Token Properties ===")

    tokenizer = ReLiKTokenizer("bert-base-uncased")

    # Check entity markers
    for i in range(5):
        marker = f"<ST{i}>"
        token_id = tokenizer.entity_marker_ids[i]
        assert token_id > 0
        decoded = tokenizer.tokenizer.decode([token_id])
        print(f"✓ Entity marker {marker}: ID={token_id}, decoded='{decoded}'")

    # Check relation markers
    for i in range(3):
        marker = f"<R{i}>"
        token_id = tokenizer.relation_marker_ids[i]
        assert token_id > 0
        decoded = tokenizer.tokenizer.decode([token_id])
        print(f"✓ Relation marker {marker}: ID={token_id}, decoded='{decoded}'")

    print("✓ Special token properties tests passed")


def test_forward_pass_dimensions():
    """Test that forward pass produces correct dimensions."""
    print("\n=== Testing Forward Pass Dimensions ===")

    reader = ImprovedReLiKReader(
        model_name="bert-base-uncased",
        max_seq_length=128,
        use_entity_linking=True,
    )

    tokenizer = reader.tokenizer

    text = "Apple was founded by Steve Jobs"
    candidates = ["Apple Inc.", "Steve Jobs"]

    encoded = tokenizer.encode_with_candidates(text, candidates, max_length=128)

    # Forward pass
    with torch.no_grad():
        outputs = reader.forward(
            encoded["input_ids"],
            encoded["attention_mask"],
            text_end=encoded["text_end"],
            marker_positions=encoded["marker_positions"].unsqueeze(0),
        )

    assert "span_start_logits" in outputs
    assert "hidden_states" in outputs
    assert "text_length" in outputs

    print(f"✓ Span start logits shape: {outputs['span_start_logits'].shape}")
    print(f"✓ Hidden states shape: {outputs['hidden_states'].shape}")
    print(f"✓ Text length: {outputs['text_length']}")

    if "candidate_embeddings" in outputs:
        print(f"✓ Candidate embeddings: {len(outputs['candidate_embeddings'])} batches")

    print("✓ Forward pass dimension tests passed")


def run_all_tests():
    """Run all improved ReLiK tests."""
    print("\n" + "=" * 60)
    print("Running Improved ReLiK Tests")
    print("=" * 60)

    try:
        test_relik_tokenizer()
        test_tokenizer_special_tokens()
        test_retriever_contrastive_loss()
        test_forward_pass_dimensions()
        test_improved_reader_span_detection()
        test_complete_entity_linker()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
