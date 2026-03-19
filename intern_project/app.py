import streamlit as st
import json
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient
from ingest import bm25_tokenize  # Reuse our mock tokenizer

st.set_page_config(page_title="Endee RAG Assistant", page_icon="🔍", layout="wide")

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def main() -> None:
    """
    Main Streamlit application function. Implements the UI for semantic search,
    including advanced Endee features like Hybrid Search and Payload Filtering.
    """
    st.title("🔍 Endee Semantic Search Assistant")
    st.markdown("A lightning-fast RAG pipeline showcasing the advanced capabilities of the **Endee Vector Database**.")
    
    client = EndeeClient()
    if not client.ping():
        st.error("Cannot connect to Endee server at localhost:8080! Please ensure Docker is running.")
        return
        
    model = load_embedder()
    
    query = st.text_input("Ask a research question:", placeholder="e.g., How does attention work?")

    # Advanced Search Options Sidebar
    st.sidebar.header("⚙️ Advanced Search Features")
    k = st.sidebar.slider("Number of results", 1, 10, 3)

    st.sidebar.subheader("Hybrid Search")
    enable_hybrid = st.sidebar.toggle("Enable Hybrid Search (Dense + Sparse/BM25)", value=False)

    st.sidebar.subheader("Payload Metadata Filtering")
    filter_by_year = st.sidebar.checkbox("Filter by Year")
    year_filter = None
    if filter_by_year:
        year_filter = st.sidebar.slider("Select Year", 2017, 2024, 2024)

    filter_by_author = st.sidebar.checkbox("Filter by Author")
    author_filter = None
    if filter_by_author:
        authors = [
            "A. Vaswani", "Y. LeCun", "G. Hinton", "Y. Bengio",
            "I. Goodfellow", "K. He", "J. Redmon", "T. Chen",
            "A. Karpathy", "F. Chollet"
        ]
        author_filter = st.sidebar.selectbox("Select Author", authors)
    
    if st.button("Search 🚀"):
        if query:
            with st.spinner("Searching Endee Database..."):
                query_vector = model.encode(query).tolist()

                sparse_vector: Optional[Dict[str, float]] = None
                if enable_hybrid:
                    sparse_vector = bm25_tokenize(query)

                filter_dict: Dict[str, Any] = {}
                if filter_by_year and year_filter is not None:
                    filter_dict["year"] = year_filter
                if filter_by_author and author_filter is not None:
                    filter_dict["author"] = author_filter

                hits = client.search(
                    "ml_papers",
                    query_vector,
                    k=k,
                    sparse_vector=sparse_vector,
                    filter_dict=filter_dict if filter_dict else None
                )
                
                if not hits:
                    st.warning("No matching papers found. Adjust your filters or query.")
                    return
                
                st.success(f"Found {len(hits)} relevant results!")
                
                for i, hit in enumerate(hits, 1):
                    score = hit.get("score", 0.0)
                    meta_str = hit.get("meta", "{}")
                    try:
                        meta = json.loads(meta_str)
                    except json.JSONDecodeError:
                        meta = {"title": "Unknown", "text": meta_str}
                        
                    title = meta.get('title', 'Unknown Title')
                    year = meta.get('year', 'N/A')
                    author = meta.get('author', 'N/A')

                    with st.expander(f"{i}. {title} (Score: {score:.4f})", expanded=(i==1)):
                        st.markdown(f"**Metadata:** Year: `{year}` | Author: `{author}`")
                        st.markdown(f"**Abstract:**\n\n{meta.get('text', 'No text available.')}")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
