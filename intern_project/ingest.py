import argparse
import json
import time
import uuid
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

def bm25_tokenize(text: str) -> Dict[str, float]:
    """
    A simple mockup BM25 tokenization function to generate sparse vectors.
    In a real scenario, you'd use a dedicated library like rank_bm25 or splade.

    Args:
        text (str): The input text document.

    Returns:
        Dict[str, float]: A dictionary mapping tokens to their mock BM25 score.
    """
    # Just a simple term frequency counter for demonstration purposes
    # Endee's docs/sparse.md mentions passing a dictionary of word to float score
    import re
    from collections import Counter
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    total_words = len(words) if words else 1

    sparse_vec = {}
    for word, count in counts.items():
        # A mock TF-IDF/BM25 score
        sparse_vec[word] = count / total_words

    return sparse_vec

def main() -> None:
    """
    Main function to orchestrate the ingestion of enriched ML papers into Endee.
    """
    parser = argparse.ArgumentParser(description="Ingest ML papers into Endee")
    parser.add_argument("--index", type=str, default="ml_papers", help="Endee index name")
    parser.add_argument("--dataset", type=str, default="ml_papers_enriched.json", help="Path to local JSON dataset")
    args = parser.parse_args()

    # Using our custom EndeeClient with connection pooling via requests.Session
    client = EndeeClient()
    if not client.ping():
        return

    print("Loading AI Embedding model (all-MiniLM-L6-v2) from local cache...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    print(f"Creating Endee Index '{args.index}' with dimension {dim}...")
    client.create_index(args.index, dim, "cosine", 16)

    print(f"Loading super-fast offline dataset from {args.dataset}...")
    try:
        with open(args.dataset, "r") as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {args.dataset}. Have you run `python augment_dataset.py`?")
        return

    print(f"Encoding {len(docs)} documents into vector space...")
    texts = [f"{row['title']} {row['text']}" for row in docs]
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=False)

    print("Pushing intelligent vectors to Endee database...")
    vectors: List[Dict[str, Any]] = []
    for i, (row, emb) in enumerate(zip(docs, embeddings)):
        # Construct the payload metadata, including our new 'year' and 'author'
        meta = {
            "title": row.get("title", f"Doc {i}"),
            "text": row.get("text", ""),
            "year": row.get("year", 2024),
            "author": row.get("author", "Unknown")
        }

        # We also generate a simple sparse vector (mock BM25) for Hybrid Search support
        sparse_vector = bm25_tokenize(f"{row.get('title', '')} {row.get('text', '')}")

        vector_data = {
            "id": f"doc_{i}",
            "vector": emb.tolist(),
            "sparse_vector": sparse_vector,
            "meta": json.dumps(meta)
        }

        vectors.append(vector_data)

    # Insert using our connection pooled client
    client.insert_vectors(args.index, vectors)
    
    elapsed = time.time() - start_time
    print(f"Ingestion successful! Processed {len(docs)} vectors in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
