import argparse
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from endee_client import EndeeClient

def main() -> None:
    """Ingests ML papers into the Endee vector database.

    This function parses command-line arguments, connects to the Endee server,
    loads the `sentence-transformers/all-MiniLM-L6-v2` embedding model,
    and downloads the `mteb/scifact` dataset. It then computes dense embeddings
    for the text content and inserts the vectors in batches into the specified
    Endee index.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Ingest ML papers into Endee")
    parser.add_argument("--num_docs", type=int, default=100, help="Number of documents to ingest")
    parser.add_argument("--index", type=str, default="ml_papers", help="Endee index name")
    args = parser.parse_args()

    client = EndeeClient()
    if not client.ping():
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    print(f"Ensuring index '{args.index}' exists with dimension {dim}...")
    client.create_index(args.index, dim, "cosine", 16)

    print("Loading dataset (mteb/scifact)...")
    # SciFact contains scientific papers which fits the ML papers requirement well
    dataset = load_dataset("mteb/scifact", "corpus", split="corpus")
    
    docs = dataset.select(range(min(args.num_docs, len(dataset))))

    print(f"Encoding {len(docs)} documents...")
    texts: List[str] = [f"{row['title']} {row['text']}" for row in docs]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Pushing to Endee vector database...")
    vectors: List[Dict[str, Any]] = []
    for i, (row, emb) in enumerate(zip(docs, embeddings)):
        meta: Dict[str, str] = {
            "title": row.get("title", f"Document {i}"),
            "text": row.get("text", "")
        }
        vectors.append({
            "id": str(row.get("_id", i)),
            "vector": emb.tolist(),
            "meta": json.dumps(meta) # Store structured metadata as JSON string
        })
        
        # Batch insert every 50
        if len(vectors) >= 50:
            client.insert_vectors(args.index, vectors)
            vectors = []
            print(f"Inserted {i + 1} documents...")

    if vectors:
        client.insert_vectors(args.index, vectors)
        print(f"Inserted final batch. Total docs: {len(docs)}")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
