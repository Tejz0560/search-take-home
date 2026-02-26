from datetime import datetime

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    document: Document
    score: float = Field(..., ge=0)
    reason: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class SearchEntry(BaseModel):
    query: str = Field(..., min_length=1)
    timestamp: datetime


class CypherQuery(BaseModel):
    """Represents a Cypher query generated from natural language.

    The model is intentionally minimal for the prototype: a single field
    containing the actual query string.  Converters use ``__str__`` to obtain
    the text form when constructing or executing the query.
    """

    cypher: str

    def __str__(self) -> str:
        """Return the contained Cypher query verbatim.

        This allows code that works with ``CypherQuery`` instances to treat them
        as strings when needed, for example when appending to log messages or
        passing to downstream search functions.
        """
        return self.cypher
