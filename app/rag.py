
# rag.py

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.embeddings import create_embedding
from app.vector_db import collection


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def create_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


def index_pdf(file_path: str):
    text = read_pdf(file_path)
    chunks = create_chunks(text)

    embeddings = [create_embedding(chunk).tolist() for chunk in chunks]

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print(f"Stored {len(chunks)} chunks.")