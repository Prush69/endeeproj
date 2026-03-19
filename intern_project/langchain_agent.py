import json
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain_community.llms import FakeListLLM  # Using a fake LLM to avoid needing API keys for the demo
from endee_client import EndeeClient

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
    Demonstrates the custom EndeeRetriever in a simple LangChain pipeline,
    setting up a conversational RetrievalQA chain.
    """
    print("Initializing Endee Retriever...")
    retriever = EndeeRetriever(k=2)

    if not retriever.client.ping():
        print("Endee server is not reachable. Exiting.")
        return

    query = "How do attention mechanisms improve neural networks?"
    print(f"\nQuerying Endee for: '{query}'")

    docs = retriever.invoke(query)

    if not docs:
        print("No documents retrieved. Have you ingested data into Endee?")
        return

    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Author: {doc.metadata.get('author')} | Year: {doc.metadata.get('year')}")
        print(f"Content snippet: {doc.page_content[:150]}...")

    print("\n" + "="*50)
    print("Demonstrating RetrievalQA Chain with Endee")
    print("="*50)

    # We use a Fake LLM here so the demo runs perfectly locally without OpenAI/Anthropic API keys,
    # but in a real scenario you would drop in ChatOpenAI() or similar.
    responses = [
        "Based on the retrieved papers, attention mechanisms allow neural networks to dynamically "
        "focus on different parts of the input sequence, overcoming the bottleneck of fixed-length context vectors. "
        "This significantly improves performance on tasks like machine translation."
    ]
    llm = FakeListLLM(responses=responses)

    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print(f"\nQuestion: {query}")
    print("\nGenerating answer via QA Chain...")

    result = qa_chain.invoke({"query": query})

    print(f"\nAnswer: {result['result']}")
    print(f"\nSource Documents Used: {len(result['source_documents'])}")

if __name__ == "__main__":
    main()
