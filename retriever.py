import chromadb

from embeddings import generate_embeddings


# ============================================
# ChromaDB Client
# ============================================

client = chromadb.PersistentClient(
    path="vector_store"
)

collection = client.get_or_create_collection(
    name="rag_documents"
)


# ============================================
# Clear Existing Documents
# ============================================

def clear_collection():

    global collection

    try:
        client.delete_collection(
            "rag_documents"
        )

    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="rag_documents"
    )

    print("Old ChromaDB data cleared.")


# ============================================
# Store Chunks
# ============================================

def store_chunks(
    chunks,
    sources,
    pages,
    chunk_numbers
):

    if not chunks:

        print("No chunks to store.")

        return

    if not (
        len(chunks)
        == len(sources)
        == len(pages)
        == len(chunk_numbers)
    ):

        raise ValueError(
            "Chunks, sources, pages and "
            "chunk numbers must have the same length."
        )

    # ----------------------------------------
    # Generate embeddings
    # ----------------------------------------

    embeddings = generate_embeddings(
        chunks
    )

    # ----------------------------------------
    # Create unique IDs
    # ----------------------------------------

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    # ----------------------------------------
    # Create metadata
    # ----------------------------------------

    metadatas = [

        {
            "source": sources[i],
            "page": pages[i],
            "chunk": chunk_numbers[i]
        }

        for i in range(len(chunks))
    ]

    # ----------------------------------------
    # Store in ChromaDB
    # ----------------------------------------

    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings,

        metadatas=metadatas
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB"
    )


# ============================================
# Retrieve Chunks
# ============================================

def retrieve_chunks(
    question,
    top_k=5
):

    # ----------------------------------------
    # Generate question embedding
    # ----------------------------------------

    question_embedding = generate_embeddings(
        [question]
    )[0]

    # ----------------------------------------
    # Check document count
    # ----------------------------------------

    total_documents = collection.count()

    if total_documents == 0:

        return [], [], []

    # ----------------------------------------
    # Prevent requesting too many documents
    # ----------------------------------------

    top_k = min(
        top_k,
        total_documents
    )

    # ----------------------------------------
    # Query ChromaDB
    # ----------------------------------------

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=top_k
    )

    # ----------------------------------------
    # Extract results
    # ----------------------------------------

    documents = results["documents"][0]

    distances = results["distances"][0]

    metadatas = results["metadatas"][0]

    # ----------------------------------------
    # Return results
    # ----------------------------------------

    return (
        documents,
        distances,
        metadatas
    )