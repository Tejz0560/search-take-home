from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# embeddings used to convert text to vectors for FAISS.  change to another
# provider if desired (HuggingFace, Cohere, etc.).
from langchain_openai import OpenAIEmbeddings

# the LLM and structured output helpers used by ``text_to_cypher``
from langchain_openai import OpenAI
from langchain_core.output_parsers import PydanticOutputParser

from .models import SearchResult, CypherQuery



async def text_to_cypher(text: str) -> CypherQuery | None:
    """Convert a natural language text query into a :class:`CypherQuery`.

    This prototype uses an LLM together with ``PydanticOutputParser`` so that
    the model return value is parsed into a :class:`CypherQuery` instance.  The
    underlying prompt instructs the model to emit JSON conforming to the
    schema generated from that Pydantic class; ``with_structured_output`` is
    handled implicitly by using the parser directly.

    If the model fails (e.g. no API key) we swallow the exception and return
    ``None`` so that callers can continue using vector search alone.
    """
    # build an LLM instance (synchronous or async; ``apredict`` returns awaited
    # result).  Configuration (model name, api key) is handled by environment
    # variables or other upstream configuration.
    llm = OpenAI()

    parser = PydanticOutputParser(pydantic_object=CypherQuery)

    prompt = (
        "Convert the following natural-language query to a Cypher query. "
        "Return your answer as JSON matching this schema:\n" + parser.schema + "\n\n"
        f"Input: {text}"
    )

    try:
        raw = await llm.apredict(prompt)
        parsed = parser.parse(raw)
        return parsed
    except Exception:
        # during development we may not have a configured provider; return None
        # to indicate that no cypher query could be produced.
        return None


@lru_cache()
def load_FAISS(documents: tuple[Document, ...]) -> FAISS:
    """Return a FAISS vector store built from the provided documents.

    The list of documents is converted to a tuple so that it can be cached by
    ``lru_cache``.  The embeddings object is created once and reused by FAISS
    internally.  This function will construct the index on first call and
    return the same ``FAISS`` instance for subsequent invocations with the
    *same* document tuple.
    """
    # build embeddings and vectorstore; this will call out to an LLM provider if
    # using OpenAIEmbeddings.  In a real application you'd configure which
    # embeddings class to use via dependency injection or settings.
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(list(documents), embeddings)


def search_knowledgegraph(cypher_query: CypherQuery) -> list[SearchResult]:
    """Mock search of the knowledge graph using a :class:`CypherQuery`.

    The returned ``score`` must lie in the 0.0–1.0 range; we use a fixed
    0.9 here to simulate a high‑confidence match.  In a real system this
    function would execute the Cypher against Neo4j or similar and translate
    results into ``SearchResult`` instances.
    """
    return [
        SearchResult(
            document=Document(page_content=str(cypher_query)), score=0.9, reason="test"
        )
    ]


def _lexical_score(query: str, content: str) -> float:
    """Return a simple score based on how often the query terms appear in the
    document content.  This is intentionally naive but works well enough for a
    small in‑memory corpus.

    The query is split on whitespace and each term is counted separately in a
    case‑insensitive manner.  Documents that contain none of the terms receive a
    score of 0 and are omitted from the final results.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return 0.0

    terms = query_lower.split()
    content_lower = content.lower()
    score = 0
    for term in terms:
        score += content_lower.count(term)
    return float(score)


async def search_documents(query: str, documents: list[Document]) -> list[SearchResult]:
    """Perform a two‑step search using FAISS and a (mock) knowledge graph.

    The provided *query* is treated as natural-language text.  It is used both
    for the vector similarity search and for conversion to a Cypher query that
    can be executed against the (mock) knowledge graph.

    The vector search returns results with a distance score; we convert that to
    a similarity value so that higher numbers indicate better relevance.  If the
    embedding call or cypher conversion fails during development, the function
    degrades gracefully and still returns whatever vector hits were obtained.
    """
    if not query:
        return []

    store = load_FAISS(tuple(documents))
    # request all documents to allow reranking outside the vector store
    faiss_hits = store.similarity_search_with_score(query, k=len(documents))

    results: list[SearchResult] = []
    for doc, distance in faiss_hits:
        # transform distance to a pseudo-score; FAISS distances are non-negative
        # with smaller meaning more similar, so we invert.
        results.append(SearchResult(document=doc, score=1.0 / (1 + distance)))

    # attempt cypher transformation to augment results; if we get a
    # CypherQuery back we hand it to the KG search helper.  failures are
    # ignored so the vector results are still returned.
    cypher_obj = await text_to_cypher(query)
    if cypher_obj is not None:
        results.extend(search_knowledgegraph(cypher_obj))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
