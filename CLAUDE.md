# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands must use the local venv:
```bash
.venv/bin/python
.venv/bin/pip install <package>
```

External services required:
- **Milvus** at `http://localhost:19530` (managed via Attu on port 8000)
- **OpenRouter** API key set as `OPENROUTER_API_KEY` env var

## Common Commands

```bash
# Ingest documents into Milvus (idempotent — skips if collection exists)
.venv/bin/python ingest.py

# Run the full end-to-end demo (ingest + demo queries)
.venv/bin/python main.py

# Start the web chat UI
.venv/bin/uvicorn app:app --port 8085 --reload

# Ask a single question from the CLI
.venv/bin/python -c "from rag import ask; print(ask('your question'))"

# Run all tests (no external services needed — all mocked)
.venv/bin/pytest tests/ -v

# Run a single test file or test function
.venv/bin/pytest tests/test_rag.py -v
.venv/bin/pytest tests/test_app.py::test_chat_returns_answer -v
```

## Docker Services

Milvus runs as 4 Docker containers. To manage them:

```bash
# Start
docker start milvus-etcd milvus-minio milvus-standalone milvus-attu

# Stop
docker stop milvus-attu milvus-standalone milvus-minio milvus-etcd
```

## Architecture

The pipeline has two phases:

**Ingestion** (`ingest.py` → Milvus):
- `documents.py` — static knowledge base (6 synthetic docs about vector DBs / RAG)
- `ingest.py` — encodes docs with `all-MiniLM-L6-v2` (384-dim, COSINE), creates `rag_docs` collection with `IVF_FLAT` index, upserts rows. Idempotent.

**Retrieval** (`rag.py` → `app.py`):
- `rag.py` — defines a PydanticAI `Agent` using OpenRouter (`meta-llama/llama-3.1-8b-instruct:free` via OpenAI-compatible API at `https://openrouter.ai/api/v1`). The agent has one tool `retrieve` (`@agent.tool_plain`) that embeds the query and fetches `TOP_K=3` chunks from Milvus. Exports both `agent` and `ask()`.
- `app.py` — FastAPI server. `GET /` serves `chat.html`. `POST /chat` accepts `{question, history}`, deserializes history with `ModelMessagesTypeAdapter`, calls `agent.run_sync()`, returns `{answer, history}` (full serialized message list for multi-turn context).
- `chat.html` — self-contained single-file chat UI. Stores `history` array in JS and round-trips it with every request.

**Key design detail:** The client owns conversation history — it stores the full serialized `ModelMessage` list returned by the server and sends it back on each request. The server is stateless.

## LLM Configuration

The LLM is configured in `rag.py` using PydanticAI's `OpenAIModel` pointed at OpenRouter:
```python
OpenAIModel("meta-llama/llama-3.1-8b-instruct:free", provider=OpenAIProvider(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]))
```
Set `OPENROUTER_API_KEY` before running. To swap models, change `OPENROUTER_MODEL`. Alternatives if rate-limited: `google/gemini-2.0-flash-exp:free`, `qwen/qwen-2.5-7b-instruct:free`. Coding-focused models do not reliably handle tool calling — use general-purpose models.
