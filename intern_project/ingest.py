import argparse
import json
import time
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

def main():
    parser = argparse.ArgumentParser(description="Ingest ML papers into Endee")
    parser.add_argument("--index", type=str, default="ml_papers", help="Endee index name")
    parser.add_argument("--dataset", type=str, default="ml_papers.json", help="Path to local JSON dataset")
    args = parser.parse_args()

    client = EndeeClient()
    if not client.ping():
        return

    print("Loading AI Embedding model (all-MiniLM-L6-v2) from local cache...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    print(f"Creating Endee Index '{args.index}' with dimension {dim}...")
    client.create_index(args.index, dim, "cosine", 16)

    print(f"Loading super-fast offline dataset from {args.dataset}...")
    with open(args.dataset, "r") as f:
        docs = json.load(f)

    print(f"Encoding {len(docs)} documents into vector space...")
    texts = [f"{row['title']} {row['text']}" for row in docs]
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=False)

    print("Pushing intelligent vectors to Endee database...")
    vectors = []
    for i, (row, emb) in enumerate(zip(docs, embeddings)):
        meta = {
            "title": row.get("title", f"Doc {i}"),
            "text": row.get("text", "")
        }
        vectors.append({
            "id": f"doc_{i}",
            "vector": emb.tolist(),
            "meta": json.dumps(meta)
        })

    client.insert_vectors(args.index, vectors)
    
    elapsed = time.time() - start_time
    print(f"Ingestion successful! Processed {len(docs)} vectors in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
