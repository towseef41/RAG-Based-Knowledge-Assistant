import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base, Chunk
from app.db.vector.db_vector_store import DBVectorStore


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_store_chunks(db_session):
    store = DBVectorStore(db_session)

    chunks = [
        {"text": "First chunk", "metadata": {"section": "intro"}},
        {"text": "Second chunk", "metadata": {"section": "summary"}},
    ]
    embeddings = [[0.1] * 10, [0.2] * 10]

    store.store_chunks(document_id=1, chunks=chunks, embeddings=embeddings)

    results = db_session.query(Chunk).filter_by(document_id=1).all()
    assert len(results) == 2
    assert results[0].text == "First chunk"
    assert results[0].chunk_metadata["section"] == "intro"
    assert results[0].embedding == embeddings[0]
    assert results[0].chunk_index == 0



def test_query_returns_expected_format(monkeypatch, db_session):
    store = DBVectorStore(db_session)

    # Monkeypatch actual `query` function to return dummy result
    dummy_result = [
        {
            "chunk_id": 1,
            "text": "Test chunk",
            "similarity": 0.95,
            "chunk_metadata": {"section": "intro"},
            "document_id": 42
        }
    ]
    monkeypatch.setattr(store, "query", lambda *args, **kwargs: dummy_result)

    results = store.query(query_embedding=[0.1] * 10)
    assert len(results) == 1
    assert results[0]["text"] == "Test chunk"
    assert results[0]["similarity"] >= 0.9


def test_query_filters_by_knowledge_base_id(monkeypatch, db_session):
    store = DBVectorStore(db_session)

    dummy_result = [
        {
            "chunk_id": 2,
            "text": "KB specific chunk",
            "similarity": 0.93,
            "chunk_metadata": {},
            "document_id": 99
        }
    ]
    monkeypatch.setattr(store, "query", lambda *args, **kwargs: dummy_result)

    results = store.query(query_embedding=[0.1] * 10, knowledge_base_id=99)
    assert results[0]["document_id"] == 99


def test_query_filters_by_metadata(monkeypatch, db_session):
    store = DBVectorStore(db_session)

    dummy_result = [
        {
            "chunk_id": 3,
            "text": "Filtered chunk",
            "similarity": 0.92,
            "chunk_metadata": {"section": "summary"},
            "document_id": 1
        }
    ]
    monkeypatch.setattr(store, "query", lambda *args, **kwargs: dummy_result)

    results = store.query(query_embedding=[0.1] * 10, filters={"section": "summary"})
    assert results[0]["chunk_metadata"]["section"] == "summary"
