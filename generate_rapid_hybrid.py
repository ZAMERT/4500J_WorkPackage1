import argparse
import json
from typing import List, Tuple

from dotenv import load_dotenv

from rapid_rag.agent import generate_with_closed_loop
from rapid_rag.embeddings import DEFAULT_EMBEDDING_MODEL
from rapid_rag.hybrid_retriever import HybridRetriever
from rapid_rag.llm import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_THINKING
from rapid_rag.loaders import DEFAULT_DOC_DIRS, DEFAULT_MANUAL_ROOT, discover_manual_dirs
from rapid_rag.prompts import rapid_generation_prompt
from rapid_rag.reranker import DEFAULT_RERANK_MODEL, rerank

load_dotenv()

DEFAULT_DB_DIR = "rapid_chroma_db_segmented"
DEFAULT_TOP_K = 8
DEFAULT_CANDIDATE_K = 18


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ABB RAPID code with hybrid RAG (vector + BM25)")
    parser.add_argument("task", nargs="*", help="User requirement. If omitted, a built-in example is used.")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    parser.add_argument("--manual-root", default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--manual-dir", default=None)
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--doc-dirs", nargs="+", default=list(DEFAULT_DOC_DIRS))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--language", default="en")
    parser.add_argument("--fallback-language", default="en")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--rerank", action="store_true", help="Enable CrossEncoder reranking after hybrid retrieval")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL, help="CrossEncoder model used when --rerank is set")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=DEFAULT_THINKING)
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-loop", type=int, default=2)
    parser.add_argument("--show-trace", action="store_true")
    return parser.parse_args()


def generate_rapid(
    user_task: str,
    db_dir: str = DEFAULT_DB_DIR,
    manuals: list = None,
    language: str = "en",
    fallback_language: str = "en",
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    rerank_enabled: bool = False,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    thinking: str = DEFAULT_THINKING,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_loop: int = 2,
) -> Tuple[str, List[dict], dict]:
    retriever = HybridRetriever(
        db_dir=db_dir,
        manuals=manuals or [],
        embedding_model=embedding_model,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
    
    def retrieve_for_generation(query: str):
        retrieved = retriever.retrieve(
            query, 
            top_k=candidate_k if rerank_enabled else top_k,
            candidate_k=candidate_k,
            language=language,
            fallback_language=fallback_language,
        )
        
        if rerank_enabled:
            retrieved = rerank(
                user_task,
                retrieved,
                top_k=top_k,
                model_name=rerank_model,
                enabled=True,
            )

        return retrieved 
    
    return generate_with_closed_loop(
        user_task=user_task,
        retrieve_fn=retrieve_for_generation,
        llm_model=llm_model,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        max_loop=max_loop,
    )


def default_task() -> str:
    return """
Generate RAPID code for an ABB robot:
- move from home position to pPick
- close gripper using a digital output
- move to pPlace
- open gripper
- return home
""".strip()


def main():
    args = parse_args()
    task = " ".join(args.task).strip() or default_task()

    manuals = discover_manual_dirs(
        manual_root=args.manual_root,
        languages=args.languages,
        doc_dirs=args.doc_dirs,
        manual_dir=args.manual_dir,
    )

    code, retrieved, trace = generate_rapid(
        task,
        db_dir=args.db_dir,
        manuals=manuals,
        language=args.language,
        fallback_language=args.fallback_language,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        embedding_model=args.embedding_model,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        rerank_enabled=args.rerank,
        rerank_model=args.rerank_model,
        llm_model=args.model,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        max_loop=args.max_loop,
    )

    print("===== Generated RAPID =====")
    print(code)

    print("\n===== Retrieved Sources =====")
    for index, item in enumerate(retrieved, start=1):
        meta = item.get("metadata") or {}
        seg = meta.get("segment", "-")
        rrf = item.get("rrf_score")
        rerank_score = item.get("rerank_score")
        scores = []
        if rrf is not None:
            scores.append(f"rrf={rrf:.4f}")
        if rerank_score is not None:
            scores.append(f"rerank={rerank_score:.4f}")
        score_text = " | ".join(scores) if scores else "score=-"
        print(
                f"{index}. [{seg}] {meta.get('manual')} | {meta.get('title')} | {meta.get('section')} | "
                f"file={meta.get('file')} | {score_text}"
        )

    if args.show_trace:
        print("\n===== Closed-loop Trace =====")
        print(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
