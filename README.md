# Milvus RAG Chat

A Retrieval-Augmented Generation (RAG) chatbot built with [Milvus](https://milvus.io/), [PydanticAI](https://ai.pydantic.dev/), and [FastAPI](https://fastapi.tiangolo.com/). Ask questions about AI and vector databases — the agent retrieves relevant context from the knowledge base before answering.

## How it works

```
User question
     │
     ▼
Embed query (all-MiniLM-L6-v2, 384-dim)
     │
     ▼
Vector search in Milvus (COSINE, top-3 chunks)
     │
     ▼
LLM generates answer with retrieved context (via OpenRouter)
     │
     ▼
Streamed response in the chat UI
```

**Two phases:**

- **Ingestion** (`ingest.py`) — encodes 6 synthetic documents about vector DBs and RAG, creates a `rag_docs` collection in Milvus with an `IVF_FLAT` index, and upserts the embeddings. Idempotent (skips if collection already exists).
- **Retrieval** (`rag.py` + `app.py`) — a PydanticAI agent with a `retrieve` tool that embeds the user query, fetches the top-3 matching chunks from Milvus, and streams the LLM response back to the browser via SSE.

The server is stateless — the client owns conversation history and sends it back on each request for multi-turn context.

## Project structure

```
.
├── app.py          # FastAPI server (GET /, POST /chat, POST /chat/stream)
├── chat.html       # Single-file chat UI (SSE streaming, multi-turn history)
├── rag.py          # PydanticAI agent + Milvus retrieve tool
├── ingest.py       # One-time document ingestion into Milvus
├── documents.py    # Static knowledge base (6 synthetic documents)
├── main.py         # End-to-end demo script (ingest + sample queries)
├── docker-compose.yml  # Milvus stack + app container
├── Dockerfile      # App container image
├── requirements.txt
└── tests/
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- An [OpenRouter](https://openrouter.ai/) account (free tier is sufficient)

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | API key from [openrouter.ai/keys](https://openrouter.ai/keys). Used to call the LLM via OpenRouter's OpenAI-compatible API. |

The LLM model is set in `rag.py`:

```python
OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"
```

Free alternative models if rate-limited:
- `google/gemini-2.0-flash-exp:free`
- `qwen/qwen-2.5-7b-instruct:free`
- `meta-llama/llama-3.1-8b-instruct:free`

> **Note:** Use general-purpose models — coding-focused models do not reliably handle tool calling.

## Running with Docker (recommended)

This starts Milvus (etcd + MinIO + standalone) and the chat app in one command:

```bash
# 1. Create your .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 2. Start all services
docker compose up -d

# 3. Ingest documents into Milvus (run once)
docker exec milvus-app python ingest.py

# 4. Open the chat UI
# http://localhost:8085
```

To stop all services:

```bash
docker compose down
```

## Running locally (without Docker)

Requires Python 3.11+ and a running Milvus instance.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 4. Ingest documents (run once)
python ingest.py

# 5. Start the web server
uvicorn app:app --port 8085 --reload

# 6. Open the chat UI
# http://localhost:8085
```

## Other commands

```bash
# Run the end-to-end demo (ingest + sample queries printed to stdout)
python main.py

# Ask a single question from the CLI
python -c "from rag import ask; print(ask('What is a vector database?'))"

# Run tests (no external services needed — all mocked)
pytest tests/ -v
```

## Services

| Service | URL | Description |
|---|---|---|
| Chat UI | http://localhost:8085 | Web chat interface |
| Milvus gRPC | localhost:19530 | Vector database API |
| Attu (Milvus UI) | http://localhost:8000 | Visual Milvus management console |
| MinIO console | http://localhost:9001 | Object storage UI (user: `minioadmin`, pass: `minioadmin`) |
