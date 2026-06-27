from typing import List, Optional

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(
    query: str,
    candidates: List[dict],
    top_k: int,
    model_name: str = DEFAULT_RERANK_MODEL,
    enabled: bool = True,
) -> List[dict]:
    """
    对候选文档做 CrossEncoder 精排。

    Args:
        query:       用户查询
        candidates:  来自 HybridRetriever 的候选列表，每项含 "document" 字段
        top_k:       最终返回条数
        model_name:  CrossEncoder 模型名
        enabled:     False 时直接按原顺序截断返回，不加载模型

    Returns:
        精排后的 top_k 条，每项新增 "rerank_score" 字段
    """
    if not enabled or not candidates:
        return candidates[:top_k]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError("sentence-transformers 未安装，请运行: pip install sentence-transformers")

    model = CrossEncoder(model_name)
    pairs = [(query, item["document"]) for item in candidates]
    scores = model.predict(pairs)

    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]