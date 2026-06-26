import argparse
import json
from typing import List, Tuple

from dotenv import load_dotenv

from rapid_rag.agent import generate_with_closed_loop
from rapid_rag.embeddings import DEFAULT_EMBEDDING_MODEL
from rapid_rag.llm import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_THINKING
from rapid_rag.retriever import RapidRetriever

load_dotenv()

DEFAULT_DB_DIR = "rapid_chroma_db"
DEFAULT_COLLECTION = "rapid_manual"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ABB RAPID code with manual-backed RAG")
    parser.add_argument("task", nargs="*", help="User requirement. If omitted, a built-in example is used.")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="Chroma database directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Must match the index embedding model")
    parser.add_argument("--language", default="en", help="Preferred manual language, e.g. en or zh-CN")
    parser.add_argument("--fallback-language", default="en", help="Fallback manual language when preferred results are sparse")
    parser.add_argument("--top-k", type=int, default=6, help="Number of chunks sent to the LLM")
    parser.add_argument("--candidate-k", type=int, default=12, help="Number of candidates retrieved before trimming")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model name")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=DEFAULT_THINKING)
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-loop", type=int, default=2, help="Maximum repair loops after the initial generation")
    parser.add_argument("--show-trace", action="store_true", help="Print closed-loop debug trace")
    return parser.parse_args()


def generate_rapid(
    user_task: str,
    language: str = "en",
    fallback_language: str = "en",
    top_k: int = 6,
    candidate_k: int = 12,
    db_dir: str = DEFAULT_DB_DIR,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    thinking: str = DEFAULT_THINKING,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_loop: int = 2,
) -> Tuple[str, List[dict], dict]:
    retriever = RapidRetriever(db_dir, collection, embedding_model)
    return generate_with_closed_loop(
        user_task=user_task,
        retrieve_fn=lambda query: retriever.retrieve(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            language=language,
            fallback_language=fallback_language,
        ),
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
    code, retrieved, trace = generate_rapid(
        task,
        language=args.language,
        fallback_language=args.fallback_language,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        db_dir=args.db_dir,
        collection=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.model,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        max_loop=args.max_loop,
    )

    print("===== Generated RAPID =====")
    print(code)

    print("\n===== Retrieved Sources =====")
    for index, item in enumerate(retrieved, start=1):
        metadata = item["metadata"] or {}
        print(
            f"{index}. {metadata.get('language')} | {metadata.get('title')} | "
            f"{metadata.get('section')} | {metadata.get('file')} | distance={item.get('distance')}"
        )

    if args.show_trace:
        print("\n===== Closed-loop Trace =====")
        print(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
