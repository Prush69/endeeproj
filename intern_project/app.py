import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

st.set_page_config(page_title="Endee RAG Assistant", page_icon="🔍", layout="wide")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def main():
    st.title("🔍 Endee Semantic Search Assistant")
    st.markdown("A lightning-fast RAG pipeline built on the **Endee Vector Database**.")
    
    client = EndeeClient()
    if not client.ping():
        st.error("Cannot connect to Endee server at localhost:8080! Please ensure Docker is running.")
        return
        
    model = load_embedder()
    
    query = st.text_input("Ask a research question:", placeholder="e.g., How does attention work?")
    k = st.slider("Number of results", 1, 10, 3)
    
    if st.button("Search 🚀"):
        if query:
            with st.spinner("Searching Endee Database..."):
                query_vector = model.encode(query).tolist()
                hits = client.search("ml_papers", query_vector, k=k)
                
                if not hits:
                    st.warning("No matching papers found. Have you run `python ingest.py` yet?")
                    return
                
                st.success(f"Found {len(hits)} relevant results in milliseconds!")
                
                for i, hit in enumerate(hits, 1):
                    score = hit.get("score", 0.0)
                    meta_str = hit.get("meta", "{}")
                    try:
                        meta = json.loads(meta_str)
                    except json.JSONDecodeError:
                        meta = {"title": "Unknown", "text": meta_str}
                        
                    with st.expander(f"{i}. {meta.get('title', 'Unknown Title')} (Score: {score:.4f})", expanded=(i==1)):
                        st.markdown(f"**Abstract:**\n\n{meta.get('text', 'No text available.')}")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
