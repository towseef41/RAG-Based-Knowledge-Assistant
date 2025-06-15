import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base, Chunk
from app.db.vector.in_memory_vector_store import InMemoryVectorStore

import numpy as np

# Create an in-memory SQLite database for testing
@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

def test_store_and_query_basic(db_session):
    store = InMemoryVectorStore(db_session)

    chunks = [
        {"text": "chunk one", "metadata": {"section": "intro"}},
        {"text": "chunk two", "metadata": {"section": "body"}},
        "chunk three"  # without metadata
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]

    store.store_chunks(document_id=1, chunks=chunks, embeddings=embeddings)

    # Query using vector close to first chunk embedding
    query_emb = [0.9, 0.1, 0.0]
    results = store.query(query_embedding=query_emb, top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "chunk one"
    assert results[0]["chunk_metadata"] == {"section": "intro"}
    assert results[1]["text"] == "chunk two" or results[1]["text"] == "chunk three"

def test_query_with_knowledge_base_filter(db_session):
    store = InMemoryVectorStore(db_session)

    # Add chunks for two different documents
    store.store_chunks(1, ["chunk1"], [[1, 0, 0]])
    store.store_chunks(2, ["chunk2"], [[0, 1, 0]])

    # Query filtered by knowledge_base_id = 1 should only return chunks from doc 1
    results = store.query([1, 0, 0], knowledge_base_id=1)
    assert all(r["document_id"] == 1 for r in results)

    # Query filtered by knowledge_base_id = 2 should only return chunks from doc 2
    results = store.query([0, 1, 0], knowledge_base_id=2)
    assert all(r["document_id"] == 2 for r in results)

def test_query_with_metadata_filter(db_session):
    store = InMemoryVectorStore(db_session)

    chunks = [
        {"text": "chunk1", "metadata": {"type": "A"}},
        {"text": "chunk2", "metadata": {"type": "B"}},
        {"text": "chunk3", "metadata": {"type": "A"}}
    ]
    embeddings = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    store.store_chunks(1, chunks, embeddings)

    # Query with metadata filter type=A should only return chunks with that metadata
    results = store.query([1, 0, 0], filters={"type": "A"})
    assert all(r["chunk_metadata"].get("type") == "A" for r in results)

def test_query_min_score_filter(db_session):
    store = InMemoryVectorStore(db_session)

    chunks = [
        "chunk1",
        "chunk2"
    ]
    embeddings = [
        [1, 0, 0],
        [0.1, 0.9, 0]
    ]
    store.store_chunks(1, chunks, embeddings)

    # Query embedding similar to chunk1
    query_emb = [1, 0, 0]

    # min_score higher than cosine similarity to second chunk should exclude it
    results = store.query(query_emb, min_score=0.95)
    assert all(r["similarity"] >= 0.95 for r in results)
    assert len(results) == 1
    assert results[0]["text"] == "chunk1"

def test_query_respects_top_k(db_session):
    store = InMemoryVectorStore(db_session)

    chunks = ["chunk1", "chunk2", "chunk3"]
    embeddings = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    store.store_chunks(1, chunks, embeddings)

    query_emb = [1, 0, 0]

    results = store.query(query_emb, top_k=1)
    assert len(results) == 1

    results = store.query(query_emb, top_k=2)
    assert len(results) == 2

def test_store_chunks_accepts_strings_and_dicts(db_session):
    store = InMemoryVectorStore(db_session)

    chunks = [
        "plain text chunk",
        {"text": "dict chunk", "metadata": {"key": "value"}}
    ]
    embeddings = [
        [0.5, 0.5, 0],
        [0, 0.5, 0.5]
    ]

    store.store_chunks(42, chunks, embeddings)

    results = db_session.query(Chunk).filter_by(document_id=42).all()
    assert len(results) == 2
    assert results[0].text == "plain text chunk"
    assert results[0].chunk_metadata == {}
    assert results[1].text == "dict chunk"
    assert results[1].chunk_metadata == {"key": "value"}

def test_query_returns_empty_list_for_no_matches(db_session):
    store = InMemoryVectorStore(db_session)
    results = store.query([1, 0, 0])
    assert results == []

def test_query_filters_metadata_missing_key(db_session):
    store = InMemoryVectorStore(db_session)

    chunks = [
        {"text": "chunk1", "metadata": {"type": "A"}},
        {"text": "chunk2", "metadata": {}}
    ]
    embeddings = [
        [1, 0, 0],
        [0, 1, 0]
    ]
    store.store_chunks(1, chunks, embeddings)

    # Filter by metadata key that some chunks don't have should not raise error and filter properly
    results = store.query([1, 0, 0], filters={"type": "A"})
    assert all("type" in r["chunk_metadata"] and r["chunk_metadata"]["type"] == "A" for r in results)
