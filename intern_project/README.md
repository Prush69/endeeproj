# Endee RAG System: Semantic Research Assistant

This intern project implements a retrieval-augmented generation (RAG) and semantic search pipeline using the open-source Endee Vector Database. It embeds machine learning papers from the `mteb/scifact` dataset using `sentence-transformers` and performs fast semantic retrieval via the Endee API.

## Project Structure
- `endee_client.py`: Python wrapper to interact with the local Endee HTTP API.
- `ingest.py`: Pipeline to download the dataset, compute dense embeddings, and push them to Endee in batches.
- `search.py`: Command-line interface to execute user queries, embed them, and retrieve the top-K visual formatted results.
- `requirements.txt`: Python package dependencies.

## Setup Instructions

### 1. Start the Endee Vector Database
You can start the Endee server using the official pre-built Docker image:

```bash
docker run -d \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```
*(Alternatively, if you have built Endee from source, you can run `docker compose up -d` from the root repository directory).*

*Note: Make sure Endee is accessible at `http://localhost:8080/api/v1/health`.*

### 2. Set Up the Python Environment
```bash
# Enter the project directory
cd intern_project

# Create a virtual environment and install dependencies
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Data Ingestion
This script will construct the "ml_papers" index, download the dataset, embed the text, and insert the vectors batch-by-batch into the Endee database.
```bash
python ingest.py --num_docs 100 --index ml_papers
```

### 4. Execute a Semantic Search
Query the database using natural language:
```bash
python search.py "What is deep learning used for in healthcare?" --k 3
```
