from typing import List, Optional

from .bm25_retriever import BM25Retriever
from .cards import CARD_COLLECTION
from .embeddings import DEFAULT_EMBEDDING_MODEL
from .retriever import SegmentedRetriever
from .vectorstore import open_collection

RRF_K = 60  # standard RRF constant, higher = smoother rank blending


def _rrf_score(rank: int, weight: float = 1.0) -> float:
    return weight / (RRF_K + rank + 1)


def reciprocal_rank_fusion(
    result_lists: list[tuple[list[dict], float]],
    top_k: int = 6,
) -> List[dict]:
    """Merge multiple ranked result lists using RRF.

    result_lists: [(results, weight), ...] where results are already ranked by relevance.
    Each result must have a metadata dict with file, section, chunk_id or document.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for results, weight in result_lists:
        for rank, item in enumerate(results):
            meta = item.get("metadata") or {}
            key = (
                meta.get("language", ""),
                meta.get("file", ""),
                meta.get("section", ""),
                str(meta.get("section_instance", 0)),
                str(meta.get("chunk_id", item.get("document", "")[:80])),
            )
            key_str = "::".join(key)
            scores[key_str] = scores.get(key_str, 0.0) + _rrf_score(rank, weight)
            if key_str not in items:
                items[key_str] = item

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    results = []
    for k in ranked_keys[:top_k]:
        item = items[k].copy()
        item["rrf_score"] = scores[k]
        results.append(item)
    return results


class HybridRetriever:
    """Combines segmented chunks, BM25, and optional canonical RAPID cards."""

    def __init__(
        self,
        db_dir: str,
        manuals: list[dict],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        use_cards: bool = True,
    ):
        self.vector = SegmentedRetriever(db_dir, embedding_model)
        self.bm25 = BM25Retriever(manuals)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.card_collection = None
        if use_cards:
            try:
                self.card_collection = open_collection(db_dir, CARD_COLLECTION)
            except Exception:
                self.card_collection = None

    def _query_cards(self, query: str, top_k: int, language: Optional[str]) -> List[dict]:
        if self.card_collection is None:
            return []
        query_embedding = self.vector.embed_model.encode([query], normalize_embeddings=True).tolist()[0]
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if language:
            kwargs["where"] = {"language": language}
        result = self.card_collection.query(**kwargs)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist if dist is not None else None,
                "card_source": "card_query",
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def _cards_for_files(self, files: list[str], language: Optional[str]) -> List[dict]:
        if self.card_collection is None:
            return []
        cards = []
        seen = set()
        for file_name in files:
            if not file_name or file_name in seen:
                continue
            seen.add(file_name)
            where = {"file": file_name}
            if language:
                where = {"$and": [{"file": file_name}, {"language": language}]}
            try:
                result = self.card_collection.get(where=where, include=["documents", "metadatas"])
            except Exception:
                continue
            for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
                cards.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": None,
                    "score": None,
                    "card_source": "card_expand",
                })
        return cards

    def _promote_cards(self, query: str, fused: list[dict], top_k: int, language: Optional[str]) -> list[dict]:
        if self.card_collection is None:
            return fused[:top_k]

        direct_cards = self._query_cards(query, max(1, top_k), language)
        files = [(item.get("metadata") or {}).get("file") for item in fused]
        expanded_cards = self._cards_for_files(files, language)

        promoted = []
        seen = set()
        for card in expanded_cards + direct_cards:
            meta = card.get("metadata") or {}
            key = (meta.get("language"), meta.get("file"), meta.get("instruction"))
            if key in seen:
                continue
            seen.add(key)
            promoted.append(card)

        if len(promoted) >= top_k:
            return promoted[:top_k]

        for item in fused:
            meta = item.get("metadata") or {}
            key = (meta.get("language"), meta.get("file"), meta.get("section"), meta.get("chunk_id"))
            if key in seen:
                continue
            promoted.append(item)
            if len(promoted) >= top_k:
                break
        return promoted

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 12,
        language: Optional[str] = "en",
        fallback_language: Optional[str] = "en",
    ) -> List[dict]:
        vector_results = self.vector.retrieve(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            language=language,
            fallback_language=fallback_language,
        )
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k, language=language)

        fused = reciprocal_rank_fusion(
            [
                (vector_results, self.vector_weight),
                (bm25_results, self.bm25_weight),
            ],
            top_k=max(top_k, candidate_k),
        )
        return self._promote_cards(query, fused, top_k=top_k, language=language)
