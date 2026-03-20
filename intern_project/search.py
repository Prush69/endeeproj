import json
import os
import warnings
import logging
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient
from ingest import compute_term_frequency

# --- SUPPRESS HUGGINGFACE NOISE ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def main() -> None:
    """
    Executes a semantic or hybrid search against the Endee Vector Database via CLI.
    Supports payload filtering by year or author.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Semantic Search for ML Papers")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--index", type=str, default="ml_papers", help="Endee index name")
    parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid Search (Dense + Sparse/TF)")
    parser.add_argument("--year", type=int, help="Filter results by publication year")
    parser.add_argument("--author", type=str, help="Filter results by exact author name")
    args = parser.parse_args()

    client = EndeeClient()
    try:
        if not client.ping():
            return
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    print("Initializing PyTorch and loading embedding model (Cold Start)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Searching Endee for: '{args.query}'")
    query_vector = model.encode(args.query).tolist()

    sparse_vector = None
    if args.hybrid:
        print("-> Hybrid Search Enabled")
        sparse_vector = compute_term_frequency(args.query)

    filter_dict: Dict[str, Any] = {}
    if args.year:
        filter_dict["year"] = args.year
        print(f"-> Filtering by Year: {args.year}")
    if args.author:
        filter_dict["author"] = args.author
        print(f"-> Filtering by Author: '{args.author}'")

    print()
    hits = client.search(
        args.index,
        query_vector,
        k=args.k,
        sparse_vector=sparse_vector,
        filter_dict=filter_dict if filter_dict else None
    )

    if not hits:
        print("No matching papers found.")
        client.close()
        return

    for rank, hit in enumerate(hits, 1):
        score = hit.get("score", 0.0)
        meta_str = hit.get("meta", "{}")
        
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            meta = {"title": "Unknown Title", "text": meta_str}
            
        title = meta.get("title", "Unknown Title")
        text = meta.get("text", "No abstract available.")
        year = meta.get("year", "N/A")
        author = meta.get("author", "N/A")
        
        # Format the short snippet
        snippet = text[:200] + "..." if len(text) > 200 else text

        print(f"[{rank}] {title}")
        print(f"Metadata: Year: {year} | Author: {author}")
        print(f"Similarity Distance/Score: {score:.4f}")
        print(f"Abstract Snippet: {snippet}")
        print("-" * 60)

    client.close()

if __name__ == "__main__":
    main()
