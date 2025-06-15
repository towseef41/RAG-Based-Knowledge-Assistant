from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy import Index
from sqlalchemy.types import JSON
import uuid
from datetime import datetime
from .database import Base


def generate_uuid():
    """Generates a UUID string."""
    return str(uuid.uuid4())


class Document(Base):
    """
    SQLAlchemy model representing a document that has been ingested into the system.

    Each document corresponds to a source file and contains high-level metadata and a
    list of associated chunks.

    Attributes:
        id (int): Primary key identifier for the document.
        name (str): Name of the document (usually the filename).
        path (str): File system or storage path of the document.
        created_at (datetime): Timestamp when the document was ingested.
        document_metadata (dict): Optional metadata about the document (e.g., source, tags, format).
        chunks (List[Chunk]): Relationship to associated text chunks split from this document.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    document_metadata = Column(JSON, nullable=True)

    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    """
    SQLAlchemy model representing a semantic chunk of a document.

    Each chunk stores a portion of the original document’s content along with its
    vector embedding and optional metadata such as section title, page number, etc.

    Attributes:
        id (int): Primary key for the chunk.
        document_id (int): Foreign key referencing the parent document.
        chunk_index (int): Order/index of the chunk in the parent document.
        text (str): The raw text content of the chunk.
        embedding (List[float]): The vector embedding of the chunk (stored as JSON).
        created_at (datetime): Timestamp when the chunk was created.
        chunk_metadata (dict): Additional metadata about the chunk (e.g., source, position).
        document (Document): SQLAlchemy relationship back to the parent document.
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True)
    chunk_index = Column(Integer, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    chunk_metadata = Column(JSON, nullable=True)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_docid_chunkindex", "document_id", "chunk_index"),
    )


class Conversation(Base):
    """
    Represents a conversation session between a user and the assistant.

    Attributes:
        id (str): UUID as the primary key. Indexed automatically.
        knowledge_base_id (str): Optional ID of the associated knowledge base.
        created_at (datetime): Timestamp when the conversation was started.
        messages (List[Message]): Relationship to messages in the conversation.

    Indexes:
        - Primary Key on `id`
    """
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=generate_uuid)
    knowledge_base_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """
    Represents a single message exchanged during a conversation.

    Attributes:
        id (int): Auto-incremented primary key.
        conversation_id (str): Foreign key to the parent conversation. Indexed.
        role (str): Role of the sender ('user', 'assistant', or 'system').
        content (str): The text content of the message.
        created_at (datetime): Timestamp when the message was created.
        conversation (Conversation): Relationship to the parent conversation.

    Indexes:
        - Index on `conversation_id` for quick filtering.
        - Composite index on (`conversation_id`, `created_at`) for ordered conversation retrieval.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("idx_conversation_created_at", "conversation_id", "created_at"),
    )