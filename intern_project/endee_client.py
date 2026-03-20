import requests
import msgpack
import json
from typing import List, Dict, Any, Optional

class EndeeClient:
    """
    A robust Python client for interacting with the Endee Vector Database.
    Implements connection pooling for high performance and supports advanced
    features like Hybrid Search and Payload Filtering.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """
        Initializes the EndeeClient.

        Args:
            base_url (str): The base URL of the Endee server.
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        # Implement requests.Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def ping(self) -> bool:
        """
        Check if the Endee server is running and accessible.

        Returns:
            bool: True if reachable, False otherwise.

        Raises:
            ConnectionError: If the server cannot be reached.
        """
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("Endee server is reachable.")
                return True
            else:
                print(f"Endee server returned status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Endee server unreachable at {self.base_url}. Please ensure it is running.") from e

    def create_index(self, index_name: str, dim: int, space_type: str = "cosine", m: int = 16) -> bool:
        """
        Create a new index in Endee.

        Args:
            index_name (str): Name of the index.
            dim (int): Vector dimensionality.
            space_type (str): Distance metric (e.g., 'cosine', 'l2').
            m (int): M parameter for HNSW.

        Returns:
            bool: True if created, False otherwise.
        """
        url = f"{self.base_url}/api/v1/index/create"
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "M": m
        }
        try:
            response = self.session.post(url, json=payload)
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
        vectors list should contain dict items with keys: 'id', 'vector', 'meta' (optional),
        'sparse_vector' (optional)

        Args:
            index_name (str): Name of the index.
            vectors (List[Dict[str, Any]]): List of vector payloads.

        Returns:
            Dict[str, Any]: Server response.
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/vector/insert"
        try:
            response = self.session.post(url, json=vectors)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return {"status": "success"}  
        except requests.exceptions.RequestException as e:
            print(f"Error inserting vectors: {e}")
            print(f"Response: {e.response.text if hasattr(e, 'response') and e.response else 'N/A'}")
            return {}

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        sparse_vector: Optional[Dict[str, float]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k nearest neighbors, optionally using Hybrid Search and Payload Filtering.

        Args:
            index_name (str): Name of the index.
            query_vector (List[float]): Dense query vector.
            k (int): Number of results to return.
            sparse_vector (Optional[Dict[str, float]]): Sparse BM25 vector for hybrid search.
            filter_dict (Optional[Dict[str, Any]]): Metadata filter criteria.

        Returns:
            List[Dict[str, Any]]: List of retrieved vector hits.
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/search"
        payload: Dict[str, Any] = {
            "k": k,
            "vector": query_vector
        }

        if sparse_vector is not None:
            # Endee API expects sparse_indices (uint32[]) and sparse_values (float[])
            # Convert from {word: score} dict using hash-based indices
            indices = [abs(hash(word)) % (2**31) for word in sparse_vector.keys()]
            values = list(sparse_vector.values())
            payload["sparse_indices"] = indices
            payload["sparse_values"] = values

        if filter_dict is not None:
            # Endee API expects filter as a JSON string containing an array:
            # [{"field": {"$eq": value}}, ...]
            filter_array = []
            for field, value in filter_dict.items():
                filter_array.append({field: {"$eq": value}})
            payload["filter"] = json.dumps(filter_array)

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            
            parsed_hits = []
            if "application/msgpack" in content_type:
                raw_data = msgpack.unpackb(response.content, raw=False)
                if isinstance(raw_data, list) and len(raw_data) > 0:
                    # Determine the structure: either a flat list of [score, id, meta]
                    # or a nested list [[score, id, meta], ...] wrapped in an outer list
                    items = raw_data
                    if len(raw_data) == 1 and isinstance(raw_data[0], list) and len(raw_data[0]) > 0 and isinstance(raw_data[0][0], list):
                        items = raw_data[0]
                    for hit in items:
                        if isinstance(hit, list) and len(hit) >= 3:
                            meta_raw = hit[2]
                            meta_str = meta_raw.decode('utf-8') if isinstance(meta_raw, bytes) else str(meta_raw)
                            parsed_hits.append({
                                "score": hit[0],
                                "id": hit[1],
                                "meta": meta_str
                            })
            else:
                # Fallback for JSON
                json_data = response.json()
                if isinstance(json_data, dict):
                    parsed_hits = json_data.get("results", json_data.get("vectors", []))
            
            # Fallback client-side post-filtering if native filter failed
            # (Endee handles it natively, but this is a robust fallback as requested)
            if filter_dict and parsed_hits:
                filtered_hits = []
                for hit in parsed_hits:
                    try:
                        meta = json.loads(hit.get("meta", "{}"))
                        match = True
                        for k_filter, v_filter in filter_dict.items():
                            meta_val = meta.get(k_filter)

                            # Type-aware comparison
                            if isinstance(v_filter, int) or isinstance(v_filter, float):
                                try:
                                    # Attempt to cast meta_val to float for numeric comparison
                                    if float(meta_val) != float(v_filter):
                                        match = False
                                        break
                                except (ValueError, TypeError):
                                    match = False
                                    break
                            else:
                                if str(meta_val) != str(v_filter):
                                    match = False
                                    break

                        if match:
                            filtered_hits.append(hit)
                    except json.JSONDecodeError:
                        continue
                return filtered_hits

            return parsed_hits
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching vectors: {e}")
            print(f"Response: {e.response.text if hasattr(e, 'response') and e.response else 'N/A'}")
            return []

    def close(self) -> None:
        """Explicitly close the HTTP session to prevent script hang on exit."""
        self.session.close()

if __name__ == "__main__":
    # Standard guard just in case someone executes this directly
    print("EndeeClient is a library. Please import it to use.")
