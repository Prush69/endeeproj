import argparse
import json
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

def main():
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
    query_vector = model.encode(args.query).tolist()

    results_data = client.search(args.index, query_vector, args.k)

    if not results_data:
        print("No results returned.")
        return

    # Extract hits flexibly to handle potential Endee response formats
    hits = results_data.get("results", [])
    if not hits and "vectors" in results_data: # Backup plan if key is named vectors
        hits = results_data.get("vectors", [])

    if isinstance(hits, list) and len(hits) > 0 and isinstance(hits[0], list):
        hits = hits[0] # Unnest if it's a batch response

    if not hits:
        print("No matching papers found.")
        return

    for rank, hit in enumerate(hits, 1):
        score = hit.get("distance", hit.get("score", 0.0))
        meta_str = hit.get("meta", "{}")
        
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            meta = {"title": "Unknown Title", "text": meta_str}
            
        title = meta.get("title", "Unknown Title")
        text = meta.get("text", "No abstract available.")
        
        # Format the short snippet
        snippet = text[:200] + "..." if len(text) > 200 else text

        print(f"[{rank}] {title}")
        print(f"Similarity Distance/Score: {score:.4f}")
        print(f"Abstract Snippet: {snippet}")
        print("-" * 60)

if __name__ == "__main__":
    main()
