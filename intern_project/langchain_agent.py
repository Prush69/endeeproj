import json
import os
import warnings
import logging
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import Field
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

# --- SUPPRESS HUGGINGFACE NOISE ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class EndeeRetriever(BaseRetriever):
    """
    A custom LangChain Retriever that hooks into the Endee Vector Database.
    This demonstrates Endee's utility as a high-performance memory layer
    for agentic workflows and LLM pipelines.
    """
    client: EndeeClient = Field(default_factory=EndeeClient)
    index_name: str = "ml_papers"
    k: int = 3
    # Use a generic type for the embedder to avoid complex Pydantic validation issues
    # with the SentenceTransformer object.
    embedder: Any = Field(default_factory=lambda: SentenceTransformer("all-MiniLM-L6-v2"))

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        """
        Synchronously retrieves documents from Endee based on the query.
        """
        # Encode the query into a dense vector
        query_vector = self.embedder.encode(query).tolist()

        # Perform the search against Endee
        hits = self.client.search(self.index_name, query_vector, k=self.k)

        documents = []
        for hit in hits:
            meta_str = hit.get("meta", "{}")
            try:
                meta = json.loads(meta_str)
            except json.JSONDecodeError:
                meta = {"title": "Unknown Title", "text": meta_str}

            # Create a LangChain Document
            doc = Document(
                page_content=meta.get("text", ""),
                metadata={
                    "title": meta.get("title", "Unknown Title"),
                    "year": meta.get("year", "N/A"),
                    "author": meta.get("author", "N/A"),
                    "score": hit.get("score", 0.0)
                }
            )
            documents.append(doc)

        return documents


def main() -> None:
    """
    Demonstrates the custom EndeeRetriever in a RAG pipeline.
    Uses a simple template-based generation approach with retrieved context
    from Endee, showing its utility as an agentic memory layer.
    """
    print("Initializing PyTorch and loading AI Embedding model (Cold Start)...")
    retriever = EndeeRetriever(k=3)

    try:
        if not retriever.client.ping():
            print("Endee server is not reachable. Exiting.")
            return
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    query = "How do attention mechanisms improve neural networks?"
    print(f"\nQuerying Endee for: '{query}'")

    # Step 1: Retrieve relevant documents from Endee
    docs = retriever.invoke(query)

    if not docs:
        print("No documents retrieved. Have you ingested data into Endee?")
        return

    print(f"\nRetrieved {len(docs)} documents from Endee:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Author: {doc.metadata.get('author')} | Year: {doc.metadata.get('year')}")
        print(f"Content snippet: {doc.page_content[:150]}...")

    # Step 2: Demonstrate RAG-style generation using retrieved context
    print("\n" + "="*60)
    print("Demonstrating RAG Pipeline with Endee as Retriever")
    print("="*60)

    # Build context from retrieved documents
    context_parts = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        author = doc.metadata.get("author", "Unknown")
        year = doc.metadata.get("year", "N/A")
        context_parts.append(
            f"[{title} by {author}, {year}]: {doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # RAG prompt template
    rag_template = PromptTemplate(
        template=(
            "Based on the following research papers retrieved from the Endee vector database, "
            "answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer (synthesized from retrieved papers):"
        ),
        input_variables=["context", "question"]
    )

    # Format the prompt (in production, this would go to an LLM like GPT-4 or Claude)
    formatted_prompt = rag_template.format(context=context, question=query)

    print(f"\nQuestion: {query}")
    print(f"\nGenerated RAG Prompt (ready for LLM):")
    print("-" * 40)
    print(formatted_prompt[:500])
    if len(formatted_prompt) > 500:
        print(f"... [{len(formatted_prompt) - 500} more characters]")
    print("-" * 40)

    # Simulated LLM answer (in production, replace with actual LLM call)
    simulated_answer = (
        "Based on the retrieved papers, attention mechanisms allow neural networks to dynamically "
        "focus on different parts of the input sequence, overcoming the bottleneck of fixed-length "
        "context vectors. The Transformer architecture (Vaswani et al., 2017) demonstrated that "
        "attention alone can achieve state-of-the-art results, eliminating the need for recurrence. "
        "This has been foundational for models like BERT and GPT."
    )

    print(f"\nSimulated LLM Answer: {simulated_answer}")
    print(f"\nSource Documents Used: {len(docs)}")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.metadata.get('title')} ({doc.metadata.get('year')})")

    retriever.client.close()

if __name__ == "__main__":
    main()
