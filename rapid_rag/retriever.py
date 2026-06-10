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


class SegmentedRetriever:
    """Queries rapid_definitions / rapid_syntax / rapid_examples independently."""

    COLLECTIONS = {
        "s1": "rapid_definitions",
        "s2": "rapid_syntax",
        "s3": "rapid_examples",
    }

    def __init__(self, db_dir: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.embed_model = load_embedding_model(embedding_model)
        self.collections = {
            seg: open_collection(db_dir, name)
            for seg, name in self.COLLECTIONS.items()
        }

    def _query_collection(self, collection, query_embedding: list, top_k: int, language: Optional[str]) -> List[dict]:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if language:
            kwargs["where"] = {"language": language}
        result = collection.query(**kwargs)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist if dist is not None else None,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 12,
        language: Optional[str] = "en",
        fallback_language: Optional[str] = "en",
        weights: Optional[dict] = None,
    ) -> List[dict]:
        """weights: per-segment candidate budget, e.g. {"s1": 0.5, "s2": 0.2, "s3": 0.3}.
        Values are treated as relative proportions of candidate_k."""
        weights = weights or {"s1": 1.0, "s2": 1.0, "s3": 1.0}
        total_w = sum(weights.values()) or 1.0
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True).tolist()[0]

        all_candidates = []
        for seg, collection in self.collections.items():
            w = weights.get(seg, 1.0)
            seg_k = max(1, round(candidate_k * w / total_w))
            results = self._query_collection(collection, query_embedding, seg_k, language)
            if fallback_language and fallback_language != language and len(results) < 2:
                results += self._query_collection(collection, query_embedding, seg_k, fallback_language)
            for r in results:
                r["segment"] = seg
            all_candidates.extend(results)

        deduped = {}
        for item in all_candidates:
            meta = item["metadata"] or {}
            key = (meta.get("language"), meta.get("file"), meta.get("section"), meta.get("chunk_id"))
            if key not in deduped or item["distance"] < deduped[key]["distance"]:
                deduped[key] = item

        return sorted(deduped.values(), key=lambda x: x["distance"])[:top_k]
