import requests
from typing import List, Dict, Any, Optional
import sys

class EndeeClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        
    def ping(self) -> bool:
        """Check if the Endee server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("Endee server is reachable.")
                return True
            else:
                print(f"Endee server returned status: {response.status_code}")
                return False
        except requests.exceptions.RequestException:
            print(f"Connection error: Could not connect to Endee server at {self.base_url}.")
            print("Please ensure the Endee server is running before continuing.")
            sys.exit(1)

    def create_index(self, index_name: str, dim: int, space_type: str = "cosine", m: int = 16) -> bool:
        """Create a new index in Endee."""
        url = f"{self.base_url}/api/v1/index/create"
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "M": m
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            if response.status_code in (200, 201):
                print(f"Index '{index_name}' successfully created.")
                return True
            else:
                print(f"Note: Index creation returned status {response.status_code}: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error creating index: {e}")
            return False

    def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert batches of vectors.
        vectors list should contain dict items with keys: 'id', 'vector', 'meta' (optional), 'filter' (optional)
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/vector/insert"
        try:
            response = requests.post(url, json=vectors, headers=self.headers)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return {"status": "success"}  # Endee might return raw string
        except requests.exceptions.RequestException as e:
            print(f"Error inserting vectors: {e}")
            print(f"Response: {e.response.text if hasattr(e, 'response') and e.response else 'N/A'}")
            return {}

    def search(self, index_name: str, query_vector: List[float], k: int = 5) -> Dict[str, Any]:
        """Search for top-k nearest neighbors."""
        url = f"{self.base_url}/api/v1/index/{index_name}/search"
        payload = {
            "k": k,
            "vector": query_vector
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error searching vectors: {e}")
            print(f"Response: {e.response.text if hasattr(e, 'response') and e.response else 'N/A'}")
            return {}
