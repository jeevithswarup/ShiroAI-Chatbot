from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

sentence=[
      "I love AI",
    "I enjoy artificial intelligence",
    "I like pizza",
    "Machine learning is amazing"
]

embeddings=model.encode(sentence)

for sentence, embedding in zip(sentences, embeddings):
    print(sentence)
    print("Dimension:", len(embedding))
    print("First 10 values:", embedding[:10])
    print("-" * 50)