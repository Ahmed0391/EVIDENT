"""
RAG tests. Use the zero-dependency 'hashing' embedding backend so they're
deterministic and need neither torch nor a network.
"""
from evident.rag.retrieve import KBIndex, reciprocal_rank_fusion
from evident.rag.ground import explain_anomaly, quote_supported, citations_valid


def test_rrf_fusion_prefers_agreed_top():
    a = ["x", "y", "z"]
    b = ["x", "z", "y"]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0] == "x"          # ranked first in both


def _index():
    return KBIndex.from_kb(backend="hashing")


def test_retrieval_finds_relevant_rule():
    idx = _index()
    hits = idx.search("national identity card number must have eight digits", top_k=3)
    ids = [h.chunk.chunk_id for h in hits]
    assert "id-format-cin-001" in ids


def test_metadata_filter_keeps_applicable():
    idx = _index()
    hits = idx.search("proof of address recent bill", applies_to="proof_of_address", top_k=3)
    assert any(h.chunk.chunk_id == "poa-recency-001" for h in hits)


def test_explain_anomaly_is_grounded_and_cited():
    idx = _index()
    exp = explain_anomaly("expired_document", idx, applies_to="passport")
    assert exp["citations"], "must return at least one citation"
    assert exp["supported"] is True
    # the quote must literally appear in the cited chunk (defence #4)
    cid = exp["citations"][0]["chunk_id"]
    assert quote_supported(exp["citations"][0]["quote"], idx.by_id[cid].text)


def test_citation_checks():
    assert citations_valid(["a", "b"], {"a", "b", "c"}) is True
    assert citations_valid(["a", "z"], {"a", "b"}) is False
    assert quote_supported("hello", "well, hello there") is True
    assert quote_supported("absent", "nothing here") is False
