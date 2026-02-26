export type SearchDocumentMetadata = {
  id: number;
  title: string;
  // Allow additional metadata fields without losing type safety.
  [key: string]: unknown;
};

export type SearchDocument = {
  page_content: string;
  metadata: SearchDocumentMetadata;
};

export type SearchResult = {
  document: SearchDocument;
  score: number;
  reason?: string;
};

export class SearchError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "SearchError";
    this.status = status;
  }
}

/**
 * Perform a search request against the backend.
 *
 * TODO (candidate):
 * - Call the `/api/search` endpoint via `fetch`.
 * - Send `{ query, top_k: topK }` as a JSON body.
 * - On non-OK responses, throw a `SearchError` with a helpful message and
 *   the HTTP status code.
 * - Parse and return the JSON response typed as `SearchResult[]`.
 */
export async function search(query: string, topK = 5): Promise<SearchResult[]> {
  // ensure caller provided something useful
  const trimmed = query.trim();
  if (!trimmed) {
    return [];
  }

  const resp = await fetch("/api/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query: trimmed, top_k: topK }),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    const message =
      text || `search request failed with status ${resp.status}`;
    throw new SearchError(message, resp.status);
  }

  const data = (await resp.json()) as SearchResult[];
  return data;
}
