"""One-shot launcher: starts a local llama.cpp OpenAI-compatible server,
loads the segmented + BM25 hybrid retriever once, then enters a REPL that
turns each stdin prompt into a full RAPID generation via the local model.

Example:
    python3 run_rapid_llama.py \
        --llama-server /Users/bytedance/Desktop/ABB/llama-infer-opt/build/bin/llama-server \
        --model /path/to/model.gguf \
        --served-model-name local-llama
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

from dotenv import load_dotenv

from rapid_rag.embeddings import DEFAULT_EMBEDDING_MODEL
from rapid_rag.hybrid_retriever import HybridRetriever
from rapid_rag.llm import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_THINKING
from rapid_rag.loaders import DEFAULT_DOC_DIRS, DEFAULT_MANUAL_ROOT, discover_manual_dirs
from rapid_rag.reranker import DEFAULT_RERANK_MODEL, rerank
from rapid_rag.agent import generate_with_closed_loop
from rapid_rag.llm import DeepSeekLLM

load_dotenv()

DEFAULT_DB_DIR = "rapid_chroma_db_segmented"
DEFAULT_TOP_K = 8
DEFAULT_CANDIDATE_K = 18
DEFAULT_LLAMA_PORT = 8080
DEFAULT_SERVED_MODEL_NAME = "local-llama"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run llama.cpp server + RAPID hybrid RAG REPL"
    )
    # llama server
    parser.add_argument(
        "--llama-server",
        default=os.getenv(
            "LLAMA_SERVER_BIN",
            "../llama-infer-opt/build/bin/llama-server",
        ),
        help="Path to llama.cpp server binary",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a .gguf model file to load in llama-server",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_LLAMA_PORT)
    parser.add_argument(
        "--served-model-name",
        default=DEFAULT_SERVED_MODEL_NAME,
        help="Alias the model will answer to (--alias in llama-server)",
    )
    parser.add_argument(
        "--llama-extra",
        default="",
        help="Extra CLI args forwarded to llama-server, e.g. '-c 8192 -ngl 99'",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=180,
        help="Seconds to wait for llama-server /health to become ready",
    )

    # retrieval / indexing
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
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)

    # generation
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="disabled",
        help="llama-server does not honor DeepSeek thinking; disabled by default",
    )
    parser.add_argument(
        "--reasoning-effort", choices=["high", "max"], default=DEFAULT_REASONING_EFFORT
    )
    parser.add_argument("--max-loop", type=int, default=2)
    parser.add_argument("--show-trace", action="store_true")
    return parser.parse_args()


# ---------- llama-server lifecycle ----------


def _health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def start_llama_server(
    binary: str,
    model_path: str,
    host: str,
    port: int,
    alias: str,
    extra: str,
) -> subprocess.Popen:
    if not os.path.isfile(binary):
        raise FileNotFoundError(f"llama-server binary not found: {binary}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")

    cmd = [
        binary,
        "-m", model_path,
        "--host", host,
        "--port", str(port),
        "--alias", alias,
    ]
    if extra.strip():
        cmd.extend(extra.strip().split())

    print(f"[llama] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    return proc


def wait_for_ready(host: str, port: int, timeout_s: int, proc: subprocess.Popen) -> None:
    url = _health_url(host, port)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early with code {proc.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print("[llama] server is ready", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"llama-server not ready after {timeout_s}s")


def stop_llama_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    print("[llama] stopping server ...", flush=True)
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
    except ProcessLookupError:
        pass


# ---------- retrieval + generation ----------


def build_retriever(args) -> HybridRetriever:
    manuals = discover_manual_dirs(
        manual_root=args.manual_root,
        languages=args.languages,
        doc_dirs=args.doc_dirs,
        manual_dir=args.manual_dir,
    )
    print(f"[rag] loading {len(manuals)} manual dir(s) for BM25", flush=True)
    retriever = HybridRetriever(
        db_dir=args.db_dir,
        manuals=manuals,
        embedding_model=args.embedding_model,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
    )
    print("[rag] retriever ready", flush=True)
    return retriever


def make_retrieve_fn(retriever: HybridRetriever, args):
    def retrieve_for_generation(query: str) -> List[dict]:
        retrieved = retriever.retrieve(
            query,
            top_k=args.candidate_k if args.rerank else args.top_k,
            candidate_k=args.candidate_k,
            language=args.language,
            fallback_language=args.fallback_language,
        )
        if args.rerank:
            retrieved = rerank(
                query,
                retrieved,
                top_k=args.top_k,
                model_name=args.rerank_model,
                enabled=True,
            )
        return retrieved

    return retrieve_for_generation


def print_result(code: str, retrieved: List[dict], trace: dict, show_trace: bool):
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
            f"{index}. [{seg}] {meta.get('manual')} | {meta.get('title')} | "
            f"{meta.get('section')} | file={meta.get('file')} | {score_text}"
        )

    if show_trace:
        print("\n===== Closed-loop Trace =====")
        print(json.dumps(trace, indent=2, ensure_ascii=False))


def repl(args, retriever: HybridRetriever, llm_client: DeepSeekLLM) -> None:
    retrieve_fn = make_retrieve_fn(retriever, args)
    print(
        "\nReady. Enter a RAPID task and press Enter. "
        "Type ':quit' to exit.\n",
        flush=True,
    )
    while True:
        try:
            line = input("task> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        task = line.strip()
        if not task:
            continue
        if task in {":q", ":quit", ":exit"}:
            break

        try:
            code, retrieved, trace = generate_with_closed_loop(
                user_task=task,
                retrieve_fn=retrieve_fn,
                llm_client=llm_client,
                llm_model=args.served_model_name,
                thinking=args.thinking,
                reasoning_effort=args.reasoning_effort,
                max_loop=args.max_loop,
            )
        except Exception as exc:
            print(f"[error] generation failed: {exc}", flush=True)
            continue

        print()
        print_result(code, retrieved, trace, args.show_trace)
        print()


def main():
    args = parse_args()

    proc = start_llama_server(
        binary=args.llama_server,
        model_path=args.model,
        host=args.host,
        port=args.port,
        alias=args.served_model_name,
        extra=args.llama_extra,
    )
    atexit.register(stop_llama_server, proc)

    def _sig_handler(signum, frame):
        stop_llama_server(proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    wait_for_ready(args.host, args.port, args.ready_timeout, proc)

    # point OpenAI client at local llama-server
    base_url = _base_url(args.host, args.port)
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    retriever = build_retriever(args)
    llm_client = DeepSeekLLM(base_url=base_url, api_key="dummy")

    repl(args, retriever, llm_client)


if __name__ == "__main__":
    main()
