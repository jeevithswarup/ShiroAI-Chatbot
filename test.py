from app.rag import read_pdf, create_chunks
from app.embeddings import create_embedding
from app.vector_db import store_embeddings, count, get_all

text = read_pdf("documents/JeevithSwarup.pdf")

chunks = create_chunks(text)

embeddings = create_embedding(chunks)

store_embeddings(chunks, embeddings)

print("Stored:", count())

print(get_all())
