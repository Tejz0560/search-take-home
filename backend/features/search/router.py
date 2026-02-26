from datetime import datetime

from fastapi import APIRouter, HTTPException

from .data import DOCUMENTS  # noqa: F401
from .models import SearchEntry, SearchRequest, SearchResult
from .integrations import search_documents

router = APIRouter(prefix="/search", tags=["search"])
SEARCH_HISTORY: list[SearchEntry] = []


@router.post("", response_model=list[SearchResult])
async def search(request: SearchRequest) -> list[SearchResult]:
    """
    Search over the in-memory DOCUMENTS collection.

    The implementation is intentionally straightforward:
    1. normalize the query and ensure it is not empty
    2. record the query in our history list
    3. call ``search_documents`` to score every document
    4. return the top ``request.top_k`` results, ordered by score (descending)

    This keeps the API layer clean while allowing the underlying ranking logic
    to be improved later without touching the route handler.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    # keep track of what people have asked for (could be exposed via another
    # endpoint or persisted later)
    SEARCH_HISTORY.append(SearchEntry(query=query, timestamp=datetime.utcnow()))

    # ask the integration to return only the requested number of hits
    results = await search_documents(query, DOCUMENTS, top_k=request.top_k)
    return results
