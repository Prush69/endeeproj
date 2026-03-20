# Endee RAG System: Semantic Research Assistant

This intern project implements a top-tier retrieval-augmented generation (RAG) and semantic search pipeline using the extremely fast **Endee Vector Database**. It embeds actual machine learning papers using `sentence-transformers` and demonstrates production-ready API integration, benchmarking, and a graphical user interface.

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
