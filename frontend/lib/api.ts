const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Citation {
  index: number;
  title: string;
  authors: string[];
  url: string;
  published: string;
}

export interface SearchResult {
  query: string;
  answer: string;
  citations: Citation[];
  total_retrieved: number;
  reranking_applied: boolean;
}

export interface SearchError {
  detail: string;
}

export async function searchPapers(
  query: string,
  topK: number = 5,
  useReranking: boolean = true
): Promise<SearchResult> {
  const res = await fetch(`${API_BASE}/api/v1/search/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: topK,
      use_reranking: useReranking,
    }),
  });

  if (!res.ok) {
    const err: SearchError = await res.json();
    throw new Error(err.detail || "Search failed");
  }

  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}