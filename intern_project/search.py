import argparse
import json
from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer # type: ignore
from transformers import pipeline # type: ignore
from endee_client import EndeeClient

def main() -> None:
    """Executes a semantic search against the Endee vector database and implements a full RAG pipeline.

    Parses command-line arguments to accept a user query and optional filters, encodes the query
    using the `sentence-transformers/all-MiniLM-L6-v2` model, and queries the
    Endee API for the top-k nearest neighbors in the specified index with the applied filters.
    It then formats and prints the search results, including the title, similarity
    score, year, domain, and an abstract snippet. Finally, it uses a lightweight local LLM
    (`google/flan-t5-base`) via the `transformers` pipeline to synthesize a direct answer
    from the retrieved abstracts, implementing a complete RAG pipeline.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Semantic Search and RAG for ML Papers")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--index", type=str, default="ml_papers", help="Endee index name")
    parser.add_argument("--year", type=int, help="Filter results by exact year (e.g., 2023)")
    parser.add_argument("--domain", type=str, help="Filter results by domain (e.g., 'Healthcare')")
    args = parser.parse_args()

    client = EndeeClient()
    if not client.ping():
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Searching Endee for: '{args.query}'\n")
    query_vector: List[float] = model.encode(args.query).tolist()

    # Construct the payload filter if arguments were provided
    search_filter: Optional[Dict[str, Any]] = None
    if args.year is not None or args.domain is not None:
        search_filter = {}
        if args.year is not None:
            search_filter["year"] = args.year
        if args.domain is not None:
            search_filter["domain"] = args.domain

    results_data: Dict[str, Any] = client.search(args.index, query_vector, args.k, filter=search_filter)

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

    retrieved_texts: List[str] = []

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
            meta = {"title": "Unknown Title", "text": meta_str, "year": "N/A", "domain": "N/A"}
            
        title: str = meta.get("title", "Unknown Title")
        text: str = meta.get("text", "No abstract available.")
        year: str = str(meta.get("year", "N/A"))
        domain: str = str(meta.get("domain", "N/A"))
        
        # Collect texts for the LLM synthesis
        if text != "No abstract available.":
            retrieved_texts.append(text)

        # Format the short snippet
        snippet: str = text[:200] + "..." if len(text) > 200 else text

        print(f"[{rank}] {title} ({year} - {domain})")
        print(f"Similarity Distance/Score: {score:.4f}")
        print(f"Abstract Snippet: {snippet}")
        print("-" * 60)

    # Initialize the local LLM and run the generation if context texts exist
    if retrieved_texts:
        print("\nLoading RAG LLM (google/flan-t5-base) to synthesize answer...")
        generator = pipeline("text2text-generation", model="google/flan-t5-base")

        # Combine snippets into context block
        context_block = " ".join(retrieved_texts)[:1500] # Trim to fit reasonable local context limits

        prompt = (
            f"Based on the following abstracts: {context_block}\n\n"
            f"Answer the question: {args.query}"
        )

        response = generator(prompt, max_new_tokens=150)

        if isinstance(response, list) and len(response) > 0:
            synthesized_answer = response[0].get("generated_text", "Could not generate answer.")
            print("\n" + "=" * 60)
            print("💡 Synthesized Answer:")
            print("=" * 60)
            print(synthesized_answer)
            print("=" * 60)

if __name__ == "__main__":
    main()
