**Note: This project is built on a forked version of the Endee repository as per the evaluation guidelines, and the original repo has been starred.**

# Endee RAG System: Semantic Research Assistant

## Project Overview and Problem Statement
Navigating the rapidly growing volume of machine learning literature is a significant challenge for researchers and engineers. This project addresses that problem by implementing a Semantic Research Assistant, a top-tier retrieval-augmented generation (RAG) and semantic search pipeline. Built upon the extremely fast **Endee Vector Database**, this assistant enables users to instantly query, filter, and synthesize insights from dense AI research papers, transforming how knowledge is discovered.

## System Design / Technical Approach
The architecture follows a robust, end-to-end RAG pipeline:
1. **Data Ingestion:** An offline dataset of ML papers is enriched with mock metadata (`year`, `author`) via `augment_dataset.py`.
2. **Embedding Generation:** The text is embedded into high-dimensional vector space using `sentence-transformers`.
3. **Vector Storage:** Dense vectors, sparse features, and metadata payloads are batch-inserted into the Endee database via a highly optimized, connection-pooled Python client (`endee_client.py`).
4. **Retrieval & UI:** Users query the system via an interactive Streamlit frontend (`app.py`), which leverages Endee's Hybrid Search (dense + sparse) and metadata filtering to retrieve the most relevant papers. An experimental `langchain_agent.py` also demonstrates Endee's capability as an agentic memory layer.

```text
[Offline JSON Dataset] --> [augment_dataset.py] --> [Enriched JSON]
                                                          |
                                                          v
                                                    [ingest.py] (Sentence Transformers)
                                                          | (Vectors + Metadata)
                                                          v
                                                [Endee Vector Database]
                                                          ^
                                                          | (Hybrid Search / Filter Queries)
                                                          v
                                     [Streamlit UI (app.py) / CLI (search.py)]
```

## Explanation of How Endee is Used
Endee was chosen as the core infrastructure due to its ultra-fast C++ backend and seamless local deployment. By utilizing its official Docker image, the database is instantly available without complex environment configurations. This project leverages Endee's advanced capabilities, including its `cosine` space indices for dense retrieval, payload filtering for metadata-aware searches, and Hybrid Search capabilities to combine exact keyword matching with semantic intent, providing an enterprise-grade search experience.

## Premium Features Included
- **Super-Fast Offline Dataset**: Ingestion completes instantly using a pre-packaged JSON dataset instead of downloading gigabytes from HuggingFace.
- **Performance Benchmarking**: A standalone script (`benchmark.py`) dedicated to evaluating Endee's latency and throughput to fulfill the *performance and scalability analysis* requirement of the internship.
- **Web UI Interface**: A fully interactive **Streamlit frontend** (`app.py`), proving real-world applicability alongside the traditional CLI (`search.py`).

## Setup Instructions

### 1. Start the Endee Vector Database
You can start the Endee server using the official pre-built Docker image seamlessly:

```bash
docker run -d \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```
*(Alternatively, if you've compiled Endee from source, just run `docker compose up -d` in the root repository).*
*Note: Verify Endee is alive at `http://localhost:8080/api/v1/health`.*

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
Construct the "ml_papers" index, load the offline AI papers dataset, embed the texts, and insert the vectors batch-by-batch:
```bash
python ingest.py 
```

### 4. Semantic Search Interfaces!
You have two ways to evaluate the search performance:

**Method A: Terminal CLI**
```bash
python search.py "deep learning for computer vision" --k 3
```

**Method B: Streamlit Web UI (Recommended)**
```bash
streamlit run app.py
```
This will open an interactive RAG chat application in your default web browser for elegant evaluation.

### 5. Evaluate Vector Database Scalability
To analyze Endee's system performance capabilities directly:
```bash
python benchmark.py
```
This script generates 5,000 synthetic high-dimensional vectors, pushes them to the DB, measures insertion throughput (vectors/second), and then queries 100 vectors to calculate average millisecond latency.
