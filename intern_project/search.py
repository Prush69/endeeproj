import argparse
import json
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer # type: ignore
from endee_client import EndeeClient

def main() -> None:
    """Executes a semantic search against the Endee vector database.

    Parses command-line arguments to accept a user query, encodes the query
    using the `sentence-transformers/all-MiniLM-L6-v2` model, and queries the
    Endee API for the top-k nearest neighbors in the specified index. It then
    formats and prints the search results, including the title, similarity
    score, and an abstract snippet.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Semantic Search for ML Papers")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--index", type=str, default="ml_papers", help="Endee index name")
    args = parser.parse_args()

    client = EndeeClient()
    if not client.ping():
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Searching Endee for: '{args.query}'\n")
    query_vector: List[float] = model.encode(args.query).tolist()

    results_data: Dict[str, Any] = client.search(args.index, query_vector, args.k)

    if not results_data:
        print("No results returned.")
        return

    # Extract hits flexibly to handle potential Endee response formats
    hits: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]] = results_data.get("results", [])
    if not hits and "vectors" in results_data: # Backup plan if key is named vectors
        hits = results_data.get("vectors", [])

    if isinstance(hits, list) and len(hits) > 0 and isinstance(hits[0], list):
        # Type checker might still complain slightly here depending on strictness,
        # but logically we unnest the list of lists.
        hits = hits[0] # type: ignore

    if not hits:
        print("No matching papers found.")
        return

    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            continue

        raw_score = hit.get("distance", hit.get("score", 0.0))
        score: float = float(raw_score) if raw_score is not None else 0.0
        meta_str: str = hit.get("meta", "{}")
        
        meta: Dict[str, Any]
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            meta = {"title": "Unknown Title", "text": meta_str}
            
        title: str = meta.get("title", "Unknown Title")
        text: str = meta.get("text", "No abstract available.")
        
        # Format the short snippet
        snippet: str = text[:200] + "..." if len(text) > 200 else text

        print(f"[{rank}] {title}")
        print(f"Similarity Distance/Score: {score:.4f}")
        print(f"Abstract Snippet: {snippet}")
        print("-" * 60)

if __name__ == "__main__":
    main()
