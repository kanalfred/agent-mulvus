from unittest.mock import MagicMock, patch
import rag


def test_retrieve_returns_texts():
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.1] * 384]

    mock_hit = {"entity": {"text": "A vector database stores embeddings."}}
    mock_client = MagicMock()
    mock_client.search.return_value = [[mock_hit, mock_hit]]

    with patch("rag._get_model", return_value=mock_model), \
         patch("rag._get_client", return_value=mock_client):
        results = rag.retrieve("What is a vector database?")

    assert isinstance(results, list)
    assert len(results) == 2
    assert "vector database" in results[0]


def test_retrieve_calls_search_with_correct_collection():
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.0] * 384]
    mock_client = MagicMock()
    mock_client.search.return_value = [[]]

    with patch("rag._get_model", return_value=mock_model), \
         patch("rag._get_client", return_value=mock_client):
        rag.retrieve("test query")

    call_kwargs = mock_client.search.call_args
    assert call_kwargs.kwargs["collection_name"] == "rag_docs"
