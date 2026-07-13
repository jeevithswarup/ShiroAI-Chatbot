from app.vector_db import semantic_search
from app.services import stream_llm

question = input("Ask: ")

chunks = semantic_search(question)

context = "\n\n".join(chunks)

for token in stream_llm(question, context):
    print(token, end="", flush=True)