import requests # type: ignore
from typing import List, Dict, Any, Optional
import sys

class EndeeClient:
    """Client to interact with the Endee vector database API.

    Attributes:
        base_url (str): The base URL of the Endee server.
        headers (Dict[str, str]): HTTP headers to use for requests.
    """

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        """Initializes the EndeeClient.

        Args:
            base_url (str, optional): The base URL of the Endee server. Defaults to "http://localhost:8080".
        """
        self.base_url: str = base_url.rstrip("/")
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        
    def ping(self) -> bool:
        """Checks if the Endee server is running and accessible.

        Returns:
            bool: True if the server is reachable and returns a 200 status code, False otherwise.

        Raises:
            SystemExit: If a connection error occurs.
        """
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
        """Creates a new index in the Endee database.

        Args:
            index_name (str): The name of the index to create.
            dim (int): The dimensionality of the vectors to be stored in the index.
            space_type (str, optional): The distance metric to use (e.g., 'cosine', 'l2', 'ip'). Defaults to "cosine".
            m (int, optional): The number of bi-directional links created for every new element during insertion (M parameter in HNSW). Defaults to 16.

        Returns:
            bool: True if the index was successfully created or already exists, False otherwise.
        """
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
        """Inserts a batch of vectors into a specified index.

        Args:
            index_name (str): The name of the index to insert vectors into.
            vectors (List[Dict[str, Any]]): A list of dictionaries representing the vectors to insert.
                Each dictionary should contain the keys: 'id' (str), 'vector' (List[float]),
                and optionally 'meta' (str) and 'filter' (Dict[str, Any]).

        Returns:
            Dict[str, Any]: A dictionary containing the response from the server.
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

    def search(self, index_name: str, query_vector: List[float], k: int = 5, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Searches for the top-k nearest neighbors to a query vector in a specified index.

        Args:
            index_name (str): The name of the index to search.
            query_vector (List[float]): The vector to search for.
            k (int, optional): The number of nearest neighbors to retrieve. Defaults to 5.
            filter (Optional[Dict[str, Any]], optional): Optional payload filter to restrict the search. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the search results from the server.
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/search"
        payload: Dict[str, Any] = {
            "k": k,
            "vector": query_vector
        }
        if filter is not None:
            payload["filter"] = filter
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error searching vectors: {e}")
            print(f"Response: {e.response.text if hasattr(e, 'response') and e.response else 'N/A'}")
            return {}
