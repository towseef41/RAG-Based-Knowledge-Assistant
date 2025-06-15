from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

# NOTE: These are example models. You can change them as you see fit.


class ChatMessage(BaseModel):
    """Representation of a message in a chat conversation."""

    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")

    @validator("role")
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v


class ChatRequest(BaseModel):
    """Simplified request model for chat completions."""

    query: str = Field(..., description="User's input question or message")
    conversation_id: Optional[str] = Field(
        None, description="ID of the conversation (if continuing)"
    )
    knowledge_base_id: Optional[str] = Field(
        None, description="ID of the knowledge base to query"
    )
    use_knowledge_base: bool = Field(
        True, description="Whether to use the knowledge base"
    )
    stream: bool = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(
        0.7, description="Temperature for response generation", ge=0, le=2
    )



class ChatResponse(BaseModel):
    """Response model for chat completions."""

    message: ChatMessage = Field(..., description="Generated assistant message")
    conversation_id: str = Field(..., description="ID of the conversation")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of response creation"
    )
    sources: Optional[List[Dict[str, Any]]] = Field(
        None, description="Sources used for the response"
    )
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")


class SearchRequest(BaseModel):
    """Request model for vector search."""

    query: str = Field(..., description="Search query text")
    limit: int = Field(5, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters to apply"
    )
    min_score: Optional[float] = Field(
        0.0, description="Minimum similarity score threshold", ge=0, le=1
    )

class DocumentChunk(BaseModel):
    """Representation of a document chunk with its metadata."""

    chunk_id: int = Field(..., description="Unique ID of the chunk")
    text: str = Field(..., description="Text content of the chunk")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata associated with the chunk"
    )
    similarity_score: Optional[float] = Field(
        None, description="Similarity score if retrieved as a search result"
    )


class SearchResponse(BaseModel):
    """Response model for vector search results."""

    query: str = Field(..., description="Original search query")
    results: List[DocumentChunk] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of matching documents")


# TODO: Add more models as needed for the assignment
