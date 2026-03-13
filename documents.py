"""Synthetic knowledge base documents about AI and Vector Databases."""

DOCUMENTS = [
    {
        "id": 1,
        "title": "What is a Vector Database?",
        "text": (
            "A vector database is a specialized database designed to store, index, and query "
            "high-dimensional vectors efficiently. Unlike traditional relational databases that "
            "store structured data in rows and columns, vector databases store mathematical "
            "representations of data called embeddings. These embeddings capture the semantic "
            "meaning of text, images, audio, or other data types as arrays of floating-point "
            "numbers. Vector databases support similarity search, allowing you to find the most "
            "semantically similar items to a given query. This makes them ideal for applications "
            "like recommendation systems, semantic search, and retrieval-augmented generation. "
            "Popular vector databases include Milvus, Pinecone, Weaviate, and Qdrant."
        ),
    },
    {
        "id": 2,
        "title": "How Embeddings Work",
        "text": (
            "Embeddings are dense numerical representations of data in a continuous vector space. "
            "They are produced by neural network models trained on large corpora of text, images, "
            "or other data. The key property of embeddings is that semantically similar items are "
            "mapped to nearby points in the vector space. For example, the words 'king' and 'queen' "
            "have similar embeddings because they share many contextual properties. Sentence "
            "transformer models like all-MiniLM-L6-v2 produce 384-dimensional embeddings for text. "
            "Larger models like OpenAI's text-embedding-ada-002 produce 1536-dimensional embeddings. "
            "The quality of embeddings depends on the model architecture, training data, and the "
            "specific task. Good embeddings capture both syntactic and semantic relationships."
        ),
    },
    {
        "id": 3,
        "title": "RAG Architecture Overview",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models "
            "by grounding their responses in external knowledge. A RAG pipeline has two main phases: "
            "indexing and retrieval. During indexing, documents are split into chunks, encoded as "
            "embeddings, and stored in a vector database. During retrieval, a user query is encoded "
            "with the same embedding model, and the most similar document chunks are fetched from "
            "the database. These retrieved chunks are then injected into the LLM's context as "
            "supporting evidence before the model generates its answer. RAG reduces hallucinations, "
            "enables the use of up-to-date information, and allows LLMs to cite specific sources. "
            "It is widely used in enterprise search, chatbots, and question-answering systems."
        ),
    },
    {
        "id": 4,
        "title": "Milvus Features and Architecture",
        "text": (
            "Milvus is an open-source vector database built for scalable similarity search. It was "
            "designed from the ground up to handle billion-scale vector datasets with high throughput "
            "and low latency. Milvus supports multiple index types including IVF_FLAT, HNSW, and "
            "DiskANN, each offering different trade-offs between accuracy, speed, and memory usage. "
            "It supports multiple metric types for similarity: L2 (Euclidean distance), IP (inner "
            "product), and COSINE similarity. Milvus has a distributed architecture with separate "
            "storage and compute layers, enabling independent scaling. It provides SDKs for Python, "
            "Java, Go, and Node.js. Milvus also supports hybrid search combining dense vectors with "
            "sparse vectors or scalar filtering, making it suitable for complex retrieval scenarios."
        ),
    },
    {
        "id": 5,
        "title": "Similarity Search: Cosine vs Dot Product vs Euclidean",
        "text": (
            "Vector similarity search relies on distance or similarity metrics to find nearest "
            "neighbors. The three most common metrics are cosine similarity, dot product (inner "
            "product), and Euclidean (L2) distance. Cosine similarity measures the angle between "
            "two vectors, ignoring magnitude — it is best when you care about direction, not scale, "
            "which is typical in text retrieval where document length varies. Dot product (inner "
            "product) measures both angle and magnitude — it is preferred when vectors are normalized "
            "or when magnitude encodes relevance. Euclidean distance measures the straight-line "
            "distance between points — it works well for image embeddings and spatial data. For most "
            "text-based RAG applications, cosine similarity is the default choice because sentence "
            "transformer models produce embeddings where semantic similarity correlates with the "
            "cosine of the angle between vectors."
        ),
    },
    {
        "id": 6,
        "title": "Use Cases for Vector Search",
        "text": (
            "Vector search enables a broad range of AI-powered applications beyond traditional "
            "keyword search. Semantic search allows users to find documents by meaning rather than "
            "exact keyword matches — a query for 'car repair' can return results about 'automobile "
            "maintenance' even without keyword overlap. Recommendation systems use vector similarity "
            "to find products, movies, or articles similar to a user's past behavior. Anomaly "
            "detection identifies data points that are far from their nearest neighbors in embedding "
            "space. Image and video search enables finding visually similar content. In biology, "
            "protein structure search uses embeddings to find similar protein folds. Multimodal "
            "search combines text and image embeddings to find relevant content across modalities. "
            "RAG systems for LLMs are currently the most popular use case, enabling chatbots and "
            "assistants to retrieve domain-specific knowledge dynamically."
        ),
    },
]
