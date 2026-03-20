import pytest
import json
from unittest.mock import patch, MagicMock
from endee_client import EndeeClient


class TestPing:
    """Tests for the EndeeClient.ping() method."""

    def test_ping_success(self):
        """ping() should return True when the server responds with 200."""
        client = EndeeClient()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client.session, "get", return_value=mock_response):
            assert client.ping() is True

    def test_ping_raises_connection_error_on_failure(self):
        """ping() should raise ConnectionError when the server is unreachable."""
        import requests
        client = EndeeClient()

        with patch.object(
            client.session, "get", side_effect=requests.exceptions.ConnectionError()
        ):
            with pytest.raises(ConnectionError):
                client.ping()

    def test_ping_returns_false_on_non_200(self):
        """ping() should return False when the server responds with a non-200 status."""
        client = EndeeClient()
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(client.session, "get", return_value=mock_response):
            assert client.ping() is False


class TestInsertVectors:
    """Tests for the EndeeClient.insert_vectors() method."""

    def test_insert_vectors_success(self):
        """insert_vectors() should return the JSON response on success."""
        client = EndeeClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "success", "count": 2}

        vectors = [
            {"id": "v1", "vector": [0.1, 0.2, 0.3]},
            {"id": "v2", "vector": [0.4, 0.5, 0.6]},
        ]

        with patch.object(client.session, "post", return_value=mock_response):
            result = client.insert_vectors("test_index", vectors)

        assert result == {"status": "success", "count": 2}


class TestSearchMsgpack:
    """Tests for the EndeeClient.search() msgpack response parsing."""

    def test_search_parses_msgpack_response(self):
        """search() should correctly parse a msgpack-encoded response."""
        import msgpack

        client = EndeeClient()

        # Simulate the nested msgpack structure Endee returns:
        # [[score, id, meta_bytes], ...]
        meta = json.dumps({"title": "Test Paper", "year": 2020}).encode("utf-8")
        raw_data = [[[0.95, "doc_0", meta]]]
        packed = msgpack.packb(raw_data, use_bin_type=True)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/msgpack"}
        mock_response.content = packed

        with patch.object(client.session, "post", return_value=mock_response):
            hits = client.search("test_index", [0.1, 0.2, 0.3], k=1)

        assert len(hits) == 1
        assert hits[0]["score"] == 0.95
        assert hits[0]["id"] == "doc_0"
        assert "Test Paper" in hits[0]["meta"]

    def test_search_returns_empty_on_error(self):
        """search() should return an empty list on request failure."""
        import requests

        client = EndeeClient()

        with patch.object(
            client.session,
            "post",
            side_effect=requests.exceptions.ConnectionError(),
        ):
            result = client.search("test_index", [0.1, 0.2])

        assert result == []
