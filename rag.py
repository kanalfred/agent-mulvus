"""PydanticAI agent with Milvus retrieval tool."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"

# To switch back to Ollama, comment out the Agent block above and use:
# OLLAMA_BASE_URL = "http://localhost:11434/v1"
# OLLAMA_MODEL = "qwen2.5:7b"
# provider=OpenAIProvider(base_url=OLLAMA_BASE_URL, api_key="ollama")

COLLECTION_NAME = "rag_docs"
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


agent = Agent(
    OpenAIModel(
        OPENROUTER_MODEL,
        provider=OpenAIProvider(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY", "no-key"),
        ),
    ),
    system_prompt=(
        "You are a helpful assistant specializing in AI and vector databases. "
        "Always use the retrieve tool to look up relevant context from the knowledge base "
        "before answering. Base your answer on the retrieved context."
    ),
)


@agent.tool_plain
def retrieve(query: str) -> list[str]:
    """Search the knowledge base for documents relevant to the query.

    Args:
        query: The search query to find relevant documents for.

    Returns:
        A list of relevant document texts from the knowledge base.
    """
    model = _get_model()
    client = _get_client()

    embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=embedding,
        anns_field="embedding",
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=TOP_K,
        output_fields=["text"],
    )

    texts = [hit["entity"]["text"] for hit in results[0]]
    return texts


def ask(question: str) -> str:
    """Run the RAG agent on a question and return the answer."""
    result = agent.run_sync(question)
    return result.output


if __name__ == "__main__":
    question = "What is a vector database?"
    print(f"Q: {question}")
    print(f"A: {ask(question)}")
