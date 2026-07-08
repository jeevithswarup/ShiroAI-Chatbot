import chromadb # type: ignore

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)

def store_embeddings(chunks, embeddings):
    """
    Store document chunks and their embeddings.
    """

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )


def get_all():
    """
    Return everything stored in the collection.
    """

    return collection.get()


def count():
    """
    Return the number of stored chunks.
    """

    return collection.count()


def search(query_embedding, n_results=3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )