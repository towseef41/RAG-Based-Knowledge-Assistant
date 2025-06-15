import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base  # adjust if path differs
from app.db.models import Document, Chunk, Conversation, Message

from datetime import datetime

# In-memory SQLite DB for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture(scope="module")
def db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_document_and_chunks_relationship(db):
    doc = Document(
        name="Test Document",
        path="/fake/path/test.txt",
        document_metadata={"author": "Alice"}
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    chunk = Chunk(
        document_id=doc.id,
        chunk_index=0,
        text="This is a test chunk",
        embedding=[0.1, 0.2, 0.3],
        chunk_metadata={"page": 1}
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)

    fetched_doc = db.query(Document).filter_by(id=doc.id).first()
    assert fetched_doc.name == "Test Document"
    assert fetched_doc.chunks[0].text == "This is a test chunk"
    assert chunk.document.name == "Test Document"


def test_conversation_and_messages_relationship(db):
    convo = Conversation(knowledge_base_id="kb1")
    db.add(convo)
    db.commit()
    db.refresh(convo)

    msg1 = Message(
        conversation_id=convo.id,
        role="user",
        content="Hello!"
    )
    msg2 = Message(
        conversation_id=convo.id,
        role="assistant",
        content="Hi! How can I help?"
    )

    db.add_all([msg1, msg2])
    db.commit()

    convo_in_db = db.query(Conversation).filter_by(id=convo.id).first()
    assert len(convo_in_db.messages) == 2
    assert convo_in_db.messages[0].role == "user"
    assert convo_in_db.messages[1].role == "assistant"
    assert msg1.conversation.id == convo.id


def test_cascade_delete_conversation_and_messages(db):
    convo = Conversation()
    db.add(convo)
    db.commit()
    db.refresh(convo)

    message = Message(
        conversation_id=convo.id,
        role="user",
        content="Will be deleted"
    )
    db.add(message)
    db.commit()

    db.delete(convo)
    db.commit()

    assert db.query(Conversation).filter_by(id=convo.id).first() is None
    assert db.query(Message).filter_by(conversation_id=convo.id).first() is None
