import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from sqlalchemy.sql.elements import ColumnElement

from app.db.models import Chunk
from app.db.vector.hybrid_vector_store import HybridVectorStore
from app.db.vector.base_vector_store import BaseVectorStore


@pytest.fixture
def mock_db_session():
    # Mock the SQLAlchemy session and query
    session = MagicMock()
    query = MagicMock()
    session.query.return_value = query
    query.filter.return_value = query
    query.limit.return_value = query
    query.all.return_value = []
    return session


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=BaseVectorStore)
    return store


@pytest.fixture
def hybrid_vector_store(mock_db_session, mock_vector_store):
    return HybridVectorStore(db_session=mock_db_session, vector_store=mock_vector_store)


def test_init(hybrid_vector_store):
    assert hybrid_vector_store.alpha == 0.7
    assert hybrid_vector_store.beta == 0.3
    assert hybrid_vector_store.vector_store is not None


def test_store_chunks_delegates_to_vector_store(hybrid_vector_store):
    chunks = ["chunk1", {"text": "chunk2", "metadata": {"key": "value"}}]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    doc_id = 42

    hybrid_vector_store.store_chunks(doc_id, chunks, embeddings)

    hybrid_vector_store.vector_store.store_chunks.assert_called_once_with(doc_id, chunks, embeddings)


def test_keyword_search_basic(mock_db_session, hybrid_vector_store):
    # Setup fake chunks returned from DB query
    fake_chunks = [
        MagicMock(id=1, text="hello world", chunk_metadata={"section": "intro"}, document_id=42),
        MagicMock(id=2, text="another chunk", chunk_metadata={"section": "body"}, document_id=42),
    ]

    query_mock = mock_db_session.query.return_value
    query_mock.filter.return_value.filter.return_value.filter.return_value.limit.return_value.all.return_value = fake_chunks

    results = hybrid_vector_store.keyword_search("hello", top_k=2, knowledge_base_id="42")

    # Verify query/filter calls
    mock_db_session.query.assert_called_once_with(Chunk)
    # It should filter on document_id and ilike
    # Also chunk.metadata[key].astext should be called in filters (none here)
    assert len(results) == 2
    assert results[0]["chunk_id"] == 1
    assert results[0]["text"] == "hello world"
    assert results[0]["similarity"] == 1.0
    assert results[0]["chunk_metadata"] == {"section": "intro"}
    assert results[0]["document_id"] == 42

def test_keyword_search_with_filters(mock_db_session, hybrid_vector_store):
    fake_chunk = MagicMock(
        id=1,
        text="some text",
        chunk_metadata={"author": "alice"},
        document_id=10
    )

    query_mock = MagicMock()
    mock_db_session.query.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.limit.return_value.all.return_value = [fake_chunk]

    filters = {"author": "alice"}

    # Create a mock bind object with dialect.name property
    mock_dialect = MagicMock()
    type(mock_dialect).name = PropertyMock(return_value="sqlite")
    mock_bind = MagicMock()
    mock_bind.dialect = mock_dialect

    # Patch get_bind to return this mock_bind
    with patch.object(hybrid_vector_store.db, "get_bind", return_value=mock_bind):

        with patch("app.db.models.Chunk.chunk_metadata", create=True) as mock_chunk_metadata:
            mock_chunk_metadata.__getitem__.side_effect = lambda key: f"chunk_metadata['{key}']"

            results = hybrid_vector_store.keyword_search("some", filters=filters)

    assert len(results) == 1
    assert results[0]["chunk_metadata"] == {"author": "alice"}
    assert results[0]["text"] == "some text"

def test_keyword_search_without_knowledge_base(mock_db_session, hybrid_vector_store):
    # Should not filter by document_id
    query_mock = mock_db_session.query.return_value
    query_mock.filter.return_value.limit.return_value.all.return_value = []

    results = hybrid_vector_store.keyword_search("anytext")
    assert results == []


def test_query_merges_vector_and_keyword_results(hybrid_vector_store):
    # Setup vector_store.query return value
    vector_results = [
        {"chunk_id": 1, "similarity": 0.9, "text": "vec1", "chunk_metadata": {}, "document_id": 1},
        {"chunk_id": 2, "similarity": 0.8, "text": "vec2", "chunk_metadata": {}, "document_id": 1},
    ]
    hybrid_vector_store.vector_store.query.return_value = vector_results

    # Setup keyword_search return value with overlapping and new chunk
    keyword_results = [
        {"chunk_id": 2, "similarity": 1.0, "text": "key2", "chunk_metadata": {}, "document_id": 1},
        {"chunk_id": 3, "similarity": 1.0, "text": "key3", "chunk_metadata": {}, "document_id": 1},
    ]
    hybrid_vector_store.keyword_search = MagicMock(return_value=keyword_results)

    combined_results = hybrid_vector_store.query(
        query_embedding=[0.1, 0.2, 0.3], top_k=3, query_text="some query"
    )

    # chunk_id 2 is in both; similarity should be weighted: 0.7*0.8 + 0.3*1.0 = 0.86
    # chunk_id 1 only in vector results with 0.9 similarity
    # chunk_id 3 only in keyword results with 1.0 similarity
    # So sorted by similarity: chunk 1 (0.9), chunk 3 (1.0), chunk 2 (0.86) => actually chunk 3 highest similarity 1.0, then 1 (0.9), then 2 (0.86)
    # But keyword-only chunks get similarity=1.0 by default, so order: 3, 1, 2

    # Let's check combined results are sorted by similarity descending and only top_k returned
    similarities = [res["similarity"] for res in combined_results]
    chunk_ids = [res["chunk_id"] for res in combined_results]

    assert chunk_ids == [3, 1, 2]
    assert similarities[0] == 1.0  # chunk 3 keyword only
    # Confirm weighted similarity for chunk 2
    expected_weighted_similarity = 0.7 * 0.8 + 0.3 * 1.0
    assert abs(combined_results[2]["similarity"] - expected_weighted_similarity) < 1e-6


def test_query_when_vector_results_empty(hybrid_vector_store):
    # Vector store returns empty, keyword search returns some results
    hybrid_vector_store.vector_store.query.return_value = []
    hybrid_vector_store.keyword_search = MagicMock(
        return_value=[
            {"chunk_id": 5, "similarity": 1.0, "text": "kw1", "chunk_metadata": {}, "document_id": 1}
        ]
    )

    results = hybrid_vector_store.query([0.0, 0.0], top_k=1, query_text="test")

    assert len(results) == 1
    assert results[0]["chunk_id"] == 5
    assert results[0]["similarity"] == 1.0


def test_query_when_keyword_results_empty(hybrid_vector_store):
    # Keyword search returns empty, vector store returns results
    hybrid_vector_store.vector_store.query.return_value = [
        {"chunk_id": 10, "similarity": 0.5, "text": "vec1", "chunk_metadata": {}, "document_id": 1}
    ]
    hybrid_vector_store.keyword_search = MagicMock(return_value=[])

    results = hybrid_vector_store.query([0.0, 0.0], top_k=1, query_text="test")

    assert len(results) == 1
    assert results[0]["chunk_id"] == 10
    assert results[0]["similarity"] == 0.5


def test_query_top_k_limit(hybrid_vector_store):
    # Return more than top_k results, ensure only top_k returned after sorting
    hybrid_vector_store.vector_store.query.return_value = [
        {"chunk_id": 1, "similarity": 0.2, "text": "vec1", "chunk_metadata": {}, "document_id": 1},
        {"chunk_id": 2, "similarity": 0.3, "text": "vec2", "chunk_metadata": {}, "document_id": 1},
        {"chunk_id": 3, "similarity": 0.1, "text": "vec3", "chunk_metadata": {}, "document_id": 1},
    ]
    hybrid_vector_store.keyword_search = MagicMock(return_value=[])

    results = hybrid_vector_store.query([0, 0, 0], top_k=2)

    assert len(results) == 2
    assert results[0]["similarity"] >= results[1]["similarity"]
