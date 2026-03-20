**Note: This project is built on a forked version of the Endee repository as per the evaluation guidelines, and the original repo has been starred.**

# Endee RAG System: Semantic Research Assistant

## Project Overview and Problem Statement
Navigating the rapidly growing volume of machine learning literature is a significant challenge for researchers and engineers. This project addresses that problem by implementing a Semantic Research Assistant — a production-grade retrieval-augmented generation (RAG) and semantic search pipeline. Built upon the extremely fast **Endee Vector Database**, this assistant enables users to instantly query, filter, and synthesize insights from dense AI research papers, transforming how knowledge is discovered.

## System Design / Technical Approach
The architecture follows a robust, end-to-end RAG pipeline:
1. **Data Ingestion:** An offline dataset of 14 landmark ML papers is enriched with historically accurate metadata (`year`, `author`) via `augment_dataset.py`.
2. **Embedding Generation:** Paper texts are embedded into 384-dimensional vector space using `sentence-transformers` (`all-MiniLM-L6-v2`).
3. **Vector Storage:** Dense vectors, sparse term-frequency features, indexed filter metadata, and unstructured payload metadata are batch-inserted into Endee via a connection-pooled Python client (`endee_client.py`).
4. **Retrieval & UI:** Users query the system via an interactive Streamlit frontend (`app.py`) or CLI (`search.py`), leveraging Endee's **Hybrid Search** (dense + sparse), **Payload Filtering** (year, author), and a **LangChain integration** (`langchain_agent.py`) as an agentic memory layer.

```text
[Offline JSON Dataset] --> [augment_dataset.py] --> [Enriched JSON]
                                                          |
                                                          v
                                                    [ingest.py] (Sentence Transformers + Term Frequency)
                                                          | (Dense Vectors + Sparse Indices + Filter Metadata)
                                                          v
                                                [Endee Vector Database]
                                                          ^
                                                          | (Dense / Hybrid / Filtered Queries)
                                                          v
                                     [Streamlit UI (app.py) / CLI (search.py) / LangChain (langchain_agent.py)]
```

## Explanation of How Endee is Used
Endee was chosen as the core infrastructure due to its ultra-fast C++ backend and seamless local deployment via Docker. This project leverages the following Endee capabilities:
- **Cosine-space HNSW indices** for high-accuracy dense semantic retrieval.
- **Indexed Payload Filtering** via the `filter` field — enabling server-side metadata-aware searches (e.g., filter by publication year or author) using Endee's Roaring Bitmap infrastructure.
- **Hybrid Search** combining dense embedding similarity with sparse term-frequency vectors (`sparse_indices` + `sparse_values`) for improved recall on keyword-heavy queries.
- **MessagePack (msgpack) responses** for high-performance binary serialization of search results.
- **Connection pooling** via `requests.Session` for efficient HTTP communication.

## Premium Features Included
- **Super-Fast Offline Dataset**: Ingestion completes in under 1 second using a pre-packaged JSON dataset instead of downloading gigabytes from HuggingFace.
- **Historically Accurate Metadata**: All 14 papers are enriched with their real publication year and lead author (e.g., "Attention Is All You Need" → Ashish Vaswani, 2017).
- **Performance Benchmarking**: A standalone script (`benchmark.py`) generates 5,000 synthetic vectors, measures insertion throughput (~825 vectors/second) and average KNN search latency (~3.4 ms per query).
- **Web UI Interface**: A fully interactive **Streamlit frontend** (`app.py`) with sidebar controls for hybrid search and metadata filtering.
- **LangChain Integration**: `langchain_agent.py` demonstrates Endee as a custom `BaseRetriever` in a RAG pipeline with a prompt template, showcasing its use as an agentic memory layer.
- **Unit Tests**: `tests/test_endee_client.py` provides pytest-based test coverage for the client's core functionality (ping, insert, search, error handling).

## Project Structure
```
intern_project/
├── endee_client.py          # Python SDK for Endee with connection pooling & msgpack support
├── ingest.py                # Data ingestion: embedding, sparse vectors, filter metadata
├── app.py                   # Streamlit web UI for interactive RAG search
├── search.py                # CLI search tool with hybrid/filter flags
├── langchain_agent.py       # LangChain retriever integration demo
├── benchmark.py             # Performance benchmarking (throughput + latency)
├── augment_dataset.py       # Enriches dataset with accurate metadata
├── ml_papers.json           # Raw dataset (14 ML papers)
├── ml_papers_enriched.json  # Enriched dataset with year/author metadata
├── requirements.txt         # Python dependencies
├── tests/
│   ├── conftest.py          # Test import path configuration
│   └── test_endee_client.py # Unit tests for EndeeClient
└── .gitignore
```

## Setup Instructions

### 1. Start the Endee Vector Database
Start the Endee server using the official pre-built Docker image:

```bash
docker run -d \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```
*(Alternatively, if you've compiled Endee from source, run `docker compose up -d` in the root repository.)*

Verify Endee is alive:
```bash
curl http://127.0.0.1:8080/api/v1/health
# Expected: {"timestamp":...,"status":"ok"}
```

### 2. Set Up the Python Environment
```bash
cd intern_project
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Knowledge Ingestion
Create the `ml_papers` index, embed all 14 papers, generate sparse term-frequency vectors, and batch-insert everything into Endee:
```bash
python ingest.py
```
Expected output:
```
Endee server is reachable.
Creating Endee Index 'ml_papers' with dimension 384...
Encoding 14 documents into vector space...
Ingestion successful! Processed 14 vectors in ~0.3 seconds.
```

### 4. Semantic Search Interfaces

**Method A: Terminal CLI**
```bash
# Basic dense semantic search
python search.py "deep learning for computer vision" --k 3

# Hybrid search (dense + sparse term frequency)
python search.py "language model training" --k 3 --hybrid

# Filter by publication year
python search.py "neural network optimization" --k 3 --year 2014

# Filter by author
python search.py "computer vision" --k 3 --author "Kaiming He"
```

**Method B: Streamlit Web UI (Recommended)**
```bash
streamlit run app.py
```
This opens an interactive search interface with sidebar controls for hybrid search, year filtering, and author filtering.

### 5. Run the LangChain Agent Demo
Demonstrates Endee as a retriever in a LangChain `RetrievalQA` chain:
```bash
python langchain_agent.py
```

### 6. Evaluate Vector Database Scalability
Benchmark Endee's insertion throughput and KNN search latency:
```bash
python benchmark.py
```
This generates 5,000 synthetic 384-dimensional vectors, measures insertion throughput (~825 vectors/second), and runs 100 queries to calculate average search latency (~3.4 ms per query).

### 7. Run Unit Tests
```bash
pytest tests/ -v
```
