import time


def open_collection(db_dir: str, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=db_dir)
    return client.get_collection(collection_name)


def create_collection(db_dir: str, collection_name: str, embedding_model: str, reset: bool = True):
    import chromadb

    print(f"Opening Chroma database at {db_dir} for collection {collection_name}", flush=True)
    client = chromadb.PersistentClient(path=db_dir)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}", flush=True)
        except Exception:
            pass

    print(f"Creating/opening collection: {collection_name}", flush=True)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
    )


def add_in_batches(
    collection,
    embed_model,
    ids,
    docs,
    metadatas,
    batch_size: int = 64,
    embedding_docs=None,
):
    total = len(docs)
    if total == 0:
        print("No documents to index.", flush=True)
        return
    if embedding_docs is not None and len(embedding_docs) != total:
        raise ValueError("embedding_docs must have the same length as docs.")

    started_at = time.monotonic()
    print(f"Starting indexing: {total} documents, batch size {batch_size}", flush=True)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_docs = docs[start:end]
        batch_embedding_docs = embedding_docs[start:end] if embedding_docs is not None else batch_docs
        batch_number = start // batch_size + 1
        batch_total = (total + batch_size - 1) // batch_size
        batch_start = time.monotonic()
        print(
            f"  Batch {batch_number}/{batch_total}: embedding documents {start + 1}-{end}",
            flush=True,
        )
        embeddings = embed_model.encode(
            batch_embedding_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        encoded_at = time.monotonic()
        print(
            f"  Batch {batch_number}/{batch_total}: embeddings ready in {encoded_at - batch_start:.1f}s; writing to Chroma",
            flush=True,
        )
        collection.add(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        elapsed = time.monotonic() - started_at
        print(
            f"Indexed {end}/{total} documents "
            f"(batch {time.monotonic() - batch_start:.1f}s, total {elapsed:.1f}s)",
            flush=True,
        )
