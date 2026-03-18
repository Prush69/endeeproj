# Endee RAG System: Semantic Research Assistant

## Project Overview

Welcome to the **Endee RAG System: Semantic Research Assistant**. This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) and semantic search pipeline utilizing the robust, high-performance **Endee Vector Database**.

The goal of this system is to enable fast, highly accurate semantic queries against a corpus of dense scientific texts. We utilize the `mteb/scifact` dataset to represent a collection of complex Machine Learning and Scientific papers. This tool enables users to run natural language queries against this literature and immediately retrieve highly relevant contextual snippets.

## System Design

This pipeline bridges state-of-the-art NLP models with efficient vector retrieval architecture. The core architecture is broken down as follows:

1. **Embedding Layer**: Natural language (both scientific abstracts and user queries) is mapped to a high-dimensional dense vector space using `sentence-transformers/all-MiniLM-L6-v2`. This model offers a strong balance of semantic representation quality and rapid inference time.
2. **Data Ingestion (`ingest.py`)**:
   - Downloads and preprocesses the document corpus.
   - Computes embeddings iteratively.
   - Serializes document metadata alongside vector arrays.
   - Pushes batches of vectors into the vector index to ensure efficient network payload sizes.
3. **Vector Storage & Retrieval (Endee Database)**:
   - Receives vectorized documents and metadata.
   - Constructs an optimized **HNSW (Hierarchical Navigable Small World)** graph using cosine distance.
   - Performs ultra-fast approximate nearest neighbor (ANN) searches upon request.
4. **API Client (`endee_client.py`)**:
   - A robust Python API wrapper that communicates with the Endee C++ backend via RESTful HTTP requests.
5. **Search Interface (`search.py`)**:
   - Serves as the user-facing CLI. Embeds incoming queries, retrieves $k$-nearest neighbors from Endee, decodes the associated JSON metadata, and presents formatted contextual abstracts.

## Utilizing the Endee Vector Database

**Endee** was chosen for this project due to its low-latency search characteristics and its native ability to tie rich, unstructured JSON metadata directly to the vector nodes.

Within this project, Endee handles:
* **Index Creation**: We instantiate an index specifically tuned for `cosine` space with the precise dimensionality (`dim=384`) required by the MiniLM model.
* **Vector + Meta Storage**: We take advantage of Endee's `meta` payload to inject structured JSON (Title, Abstract Text) directly into the index. This removes the need for a secondary relational database to map vector IDs back to human-readable text.
* **Similarity Search**: Queries are sent as raw float arrays to the `/search` endpoint, which traverses the internal HNSW graph and rapidly returns the top $K$ most similar documents and their scores.

## Setup & Execution Instructions

Follow these steps to deploy and test the Semantic Research Assistant.

### 1. Initialize the Endee Vector Database
You must have an active instance of the Endee Vector Database running. The simplest approach is via Docker:

```bash
docker run -d \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```
*(Note: If compiling Endee from source, utilize `docker compose up -d` from the root repository directory).*

Ensure the database is actively receiving traffic by checking its health endpoint: `http://localhost:8080/api/v1/health`.

### 2. Configure Python Environment
Navigate to the project root and establish an isolated virtual environment:

```bash
# Enter the project directory
cd intern_project

# Create a virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux/macOS:
source venv/bin/activate

# Install strictly defined dependencies
pip install -r requirements.txt
```

### 3. Execute Data Ingestion Pipeline
Construct the vector index and populate it with dense scientific literature.

```bash
python ingest.py --num_docs 100 --index ml_papers
```
*(This will download the corpus, load the MiniLM weights, compute embeddings, and execute batch inserts to the Endee server).*

### 4. Perform Semantic Queries
Execute natural language searches against the ingested literature to retrieve relevant abstracts.

```bash
python search.py "What is deep learning used for in healthcare?" --k 3
```
