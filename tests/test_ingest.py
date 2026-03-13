from unittest.mock import MagicMock, patch
from ingest import collection_exists, create_collection, ingest


def test_collection_exists_true():
    client = MagicMock()
    client.has_collection.return_value = True
    assert collection_exists(client) is True


def test_collection_exists_false():
    client = MagicMock()
    client.has_collection.return_value = False
    assert collection_exists(client) is False


def test_create_collection_calls_create():
    client = MagicMock()
    create_collection(client)
    client.create_collection.assert_called_once()


def test_ingest_skips_if_collection_exists():
    client = MagicMock()
    client.has_collection.return_value = True
    with patch("ingest.get_client", return_value=client):
        ingest()
    client.insert.assert_not_called()


def test_ingest_inserts_documents():
    client = MagicMock()
    client.has_collection.return_value = False
    client.insert.return_value = {"insert_count": 6}
    fake_embeddings = [[0.1] * 384] * 6
    with patch("ingest.get_client", return_value=client), \
         patch("ingest.SentenceTransformer") as MockST:
        MockST.return_value.encode.return_value.tolist.return_value = fake_embeddings
        ingest()
    client.insert.assert_called_once()
