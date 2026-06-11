from typing import List, Optional

from .bm25_retriever import BM25Retriever
from .embeddings import DEFAULT_EMBEDDING_MODEL
from .retriever import SegmentedRetriever

RRF_K = 60  # standard RRF constant, higher = smoother rank blending


def _rrf_score(rank: int, weight: float = 1.0) -> float:
    return weight / (RRF_K + rank + 1)


def reciprocal_rank_fusion(
    result_lists: list[tuple[list[dict], float]],
    top_k: int = 6,
) -> List[dict]:
    """Merge multiple ranked result lists using RRF.

    result_lists: [(results, weight), ...] where results are already ranked by relevance.
    Each result must have a 'metadata' dict with 'file', 'section', 'chunk_id' or 'document'.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for results, weight in result_lists:
        for rank, item in enumerate(results):
            meta = item.get("metadata") or {}
            # use document text as fallback key for BM25 results without chunk_id
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
    """Combines SegmentedRetriever (vector) + BM25Retriever (keyword) via RRF."""

    def __init__(
        self,
        db_dir: str,
        manuals: list[dict],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.vector = SegmentedRetriever(db_dir, embedding_model)
        self.bm25 = BM25Retriever(manuals)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

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

        return reciprocal_rank_fusion(
            [
                (vector_results, self.vector_weight),
                (bm25_results, self.bm25_weight),
            ],
            top_k=top_k,
        )
