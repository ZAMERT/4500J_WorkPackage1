def open_collection(db_dir: str, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=db_dir)
    return client.get_collection(collection_name)


def create_collection(db_dir: str, collection_name: str, embedding_model: str, reset: bool = True):
    import chromadb

    client = chromadb.PersistentClient(path=db_dir)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
    )


def add_in_batches(collection, embed_model, ids, docs, metadatas, batch_size: int = 64):
    for start in range(0, len(docs), batch_size):
        end = min(len(docs), start + batch_size)
        batch_docs = docs[start:end]
        embeddings = embed_model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        print(f"Indexed {end}/{len(docs)} chunks")
