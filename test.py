from app.rag import read_pdf, create_chunks
from app.embeddings import create_embedding
text = read_pdf("documents/JeevithSwarup.pdf")





text = read_pdf("documents/JeevithSwarup.pdf")

chunks = create_chunks(text)

print(f"Total Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n===== Chunk {i+1} =====")
    print(chunk)



text = "Python is a programming language."

embedding = create_embedding(text)

print(embedding)    