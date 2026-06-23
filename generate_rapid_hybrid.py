import argparse
import os
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from rapid_rag.embeddings import DEFAULT_EMBEDDING_MODEL
from rapid_rag.hybrid_retriever import HybridRetriever
from rapid_rag.loaders import DEFAULT_DOC_DIRS, DEFAULT_MANUAL_ROOT, discover_manual_dirs
from rapid_rag.prompts import rapid_generation_prompt

load_dotenv()

DEFAULT_DB_DIR = "rapid_chroma_db_segmented"
DEFAULT_LLM_MODEL = "deepseek-chat"
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
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    return parser.parse_args()


def make_llm_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in the environment or .env file.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


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
    llm_model: str = DEFAULT_LLM_MODEL,
) -> Tuple[str, List[dict]]:
    retriever = HybridRetriever(
        db_dir=db_dir,
        manuals=manuals or [],
        embedding_model=embedding_model,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
    retrieved = retriever.retrieve(
        user_task,
        top_k=top_k,
        candidate_k=candidate_k,
        language=language,
        fallback_language=fallback_language,
    )
    prompt = rapid_generation_prompt(user_task, retrieved)

    response = make_llm_client().chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You generate ABB RAPID code. Be precise and avoid hallucinating APIs."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content, retrieved


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

    code, retrieved = generate_rapid(
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
        llm_model=args.model,
    )

    print("===== Generated RAPID =====")
    print(code)

    print("\n===== Retrieved Sources =====")
    for index, item in enumerate(retrieved, start=1):
        meta = item.get("metadata") or {}
        seg = meta.get("segment", "-")
        rrf = item.get("rrf_score", "-")
        print(
            f"{index}. [{seg}] {meta.get('manual')} | {meta.get('title')} | {meta.get('section')} | "
            f"file={meta.get('file')} | rrf={rrf:.4f}"
        )


if __name__ == "__main__":
    main()
