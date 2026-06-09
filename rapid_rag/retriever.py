from typing import List, Optional

from .embeddings import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from .vectorstore import open_collection


class RapidRetriever:
    def __init__(self, db_dir: str, collection_name: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.embed_model = load_embedding_model(embedding_model)
        self.collection = open_collection(db_dir, collection_name)

    def _query(self, query: str, top_k: int, language: Optional[str] = None) -> List[dict]:
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True).tolist()[0]
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if language:
            kwargs["where"] = {"language": language}

        result = self.collection.query(**kwargs)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "score": 1.0 - distance if distance is not None else None,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 12,
        language: Optional[str] = "en",
        fallback_language: Optional[str] = "en",
    ) -> List[dict]:
        candidates = self._query(query, candidate_k, language=language) if language else self._query(query, candidate_k)

        if fallback_language and fallback_language != language and len(candidates) < top_k:
            candidates.extend(self._query(query, candidate_k, language=fallback_language))

        if not candidates and language:
            candidates = self._query(query, candidate_k)

        deduped = {}
        for item in candidates:
            metadata = item["metadata"] or {}
            key = (
                metadata.get("language"),
                metadata.get("manual"),
                metadata.get("file"),
                metadata.get("section"),
                metadata.get("chunk_id"),
            )
            current = deduped.get(key)
            if current is None or item["distance"] < current["distance"]:
                deduped[key] = item

        return sorted(deduped.values(), key=lambda item: item["distance"])[:top_k]
