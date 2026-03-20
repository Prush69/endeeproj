import json
import time
import os
import warnings
import logging
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

# --- SUPPRESS HUGGINGFACE NOISE ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def compute_term_frequency(text: str) -> Dict[str, float]:
    """
    A simple term frequency tokenization function to generate sparse vectors.
    This serves as a basic illustration for Hybrid Search sparse components.

    Args:
        text (str): The input text document.

    Returns:
        Dict[str, float]: A dictionary mapping tokens to their normalized term frequency.
    """
    import re
    from collections import Counter
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    total_words = len(words) if words else 1

    sparse_vec = {}
    for word, count in counts.items():
        sparse_vec[word] = count / total_words

    return sparse_vec


def main() -> None:
    """
    Main function to orchestrate the ingestion of enriched ML papers into Endee.
    """
    client = EndeeClient()
    try:
        if not client.ping():
            return
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    # Let the evaluator know the cold-start is happening
    print("Initializing PyTorch and loading AI Embedding model (Cold Start)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    index_name = "ml_papers"
    print(f"Creating Endee Index '{index_name}' with dimension {dim}...")
    client.create_index(index_name, dim, "cosine", 16)

    dataset_path = "ml_papers_enriched.json"
    print(f"Loading offline dataset from {dataset_path}...")
    try:
        with open(dataset_path, "r") as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_path}. Have you run `python augment_dataset.py`?")
        return

    print(f"Encoding {len(docs)} documents into vector space...")
    texts = [f"{row['title']} {row['text']}" for row in docs]
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=False)

    print("Pushing vectors to Endee database...")
    vectors: List[Dict[str, Any]] = []
    for i, (row, emb) in enumerate(zip(docs, embeddings)):
        # Construct the payload metadata, including 'year' and 'author'
        meta = {
            "title": row.get("title", f"Doc {i}"),
            "text": row.get("text", ""),
            "year": row.get("year", 2024),
            "author": row.get("author", "Unknown")
        }

        # Generate sparse vector (term frequency) for Hybrid Search support
        sparse_tf = compute_term_frequency(f"{row.get('title', '')} {row.get('text', '')}")
        sparse_indices = [abs(hash(word)) % (2**31) for word in sparse_tf.keys()]
        sparse_values = list(sparse_tf.values())

        # Construct filter payload for Endee's indexed filtering (separate from meta)
        filter_payload = json.dumps({"year": meta["year"], "author": meta["author"]})

        vector_data = {
            "id": f"doc_{i}",
            "vector": emb.tolist(),
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
            "meta": json.dumps(meta),
            "filter": filter_payload
        }

        vectors.append(vector_data)

    # Insert using our connection pooled client
    client.insert_vectors(index_name, vectors)

    elapsed = time.time() - start_time
    print(f"Ingestion successful! Processed {len(docs)} vectors in {elapsed:.2f} seconds.")

    # Close the session so the script exits instantly
    client.close()


if __name__ == "__main__":
    main()
