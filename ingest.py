"""Embed documents and upsert into Milvus collection."""

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

from documents import DOCUMENTS

COLLECTION_NAME = "rag_docs"
EMBEDDING_DIM = 384
MILVUS_URI = "http://localhost:19530"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


def collection_exists(client: MilvusClient) -> bool:
    return client.has_collection(COLLECTION_NAME)


def create_collection(client: MilvusClient) -> None:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 64},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}'")


def ingest() -> None:
    client = get_client()

    if collection_exists(client):
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping ingestion.")
        return

    create_collection(client)

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [doc["text"] for doc in DOCUMENTS]
    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    rows = [
        {"id": doc["id"], "text": doc["text"], "embedding": emb}
        for doc, emb in zip(DOCUMENTS, embeddings)
    ]

    result = client.insert(collection_name=COLLECTION_NAME, data=rows)
    print(f"Inserted {result['insert_count']} documents into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    ingest()
