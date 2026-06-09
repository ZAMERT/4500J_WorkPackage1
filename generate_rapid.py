import argparse
import os
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from rapid_rag.embeddings import DEFAULT_EMBEDDING_MODEL
from rapid_rag.prompts import rapid_generation_prompt
from rapid_rag.retriever import RapidRetriever

load_dotenv()

DEFAULT_DB_DIR = "rapid_chroma_db"
DEFAULT_COLLECTION = "rapid_manual"
DEFAULT_LLM_MODEL = "deepseek-chat"


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
    return parser.parse_args()


def make_llm_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in the environment or .env file.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


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
) -> Tuple[str, List[dict]]:
    retriever = RapidRetriever(db_dir, collection, embedding_model)
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
            {
                "role": "system",
                "content": "You generate ABB RAPID code. Be precise and avoid hallucinating APIs.",
            },
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
    code, retrieved = generate_rapid(
        task,
        language=args.language,
        fallback_language=args.fallback_language,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        db_dir=args.db_dir,
        collection=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.model,
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


if __name__ == "__main__":
    main()
