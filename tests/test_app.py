from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_get_index_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_chat_returns_answer():
    mock_result = MagicMock()
    mock_result.output = "A vector database stores embeddings."
    mock_result.all_messages_json.return_value = b"[]"

    with patch("app.agent.run_sync", return_value=mock_result):
        response = client.post("/chat", json={"question": "What is a vector database?", "history": []})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "A vector database stores embeddings."
    assert "history" in data


def test_chat_returns_502_on_model_error():
    from pydantic_ai.exceptions import ModelHTTPError
    with patch("app.agent.run_sync", side_effect=ModelHTTPError(400, "test-model", "low credits")):
        response = client.post("/chat", json={"question": "hello", "history": []})
    assert response.status_code == 502


def test_chat_returns_500_on_generic_error():
    with patch("app.agent.run_sync", side_effect=RuntimeError("something broke")):
        response = client.post("/chat", json={"question": "hello", "history": []})
    assert response.status_code == 500
