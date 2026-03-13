"""Print the embedding vector for a query — paste the output into Attu's Vector Search box."""

import sys
from sentence_transformers import SentenceTransformer

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your query: ")
emb = SentenceTransformer("all-MiniLM-L6-v2").encode([query], normalize_embeddings=True)[0].tolist()
print(emb)
