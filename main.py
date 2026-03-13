"""End-to-end RAG demo: ingest documents then run demo queries."""

from ingest import ingest
from rag import ask

DEMO_QUESTIONS = [
    "What is a vector database?",
    "How does RAG work?",
    "What makes Milvus different from traditional databases?",
]


def main() -> None:
    print("=== RAG Demo with Milvus + PydanticAI ===\n")

    print("--- Ingestion ---")
    ingest()
    print()

    print("--- Demo Queries ---")
    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"[{i}] Q: {question}")
        answer = ask(question)
        print(f"    A: {answer}")
        print()


if __name__ == "__main__":
    main()
