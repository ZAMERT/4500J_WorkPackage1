from typing import Callable, List, Tuple

from .llm import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_THINKING, DeepSeekLLM
from .prompts import rapid_generation_prompt, rapid_repair_prompt, rapid_repair_query
from .validator import validate_rapid_code


def _merge_retrieved(primary: list[dict], extra: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for item in primary + extra:
        metadata = item.get("metadata") or {}
        key = (
            metadata.get("language"),
            metadata.get("manual"),
            metadata.get("file"),
            metadata.get("section"),
            metadata.get("section_instance", 0),
            metadata.get("chunk_id"),
            item.get("document", "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def generate_with_closed_loop(
    user_task: str,
    retrieve_fn: Callable[[str], List[dict]],
    llm_client: DeepSeekLLM | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    thinking: str = DEFAULT_THINKING,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_loop: int = 2,
) -> Tuple[str, List[dict], dict]:
    llm_client = llm_client or DeepSeekLLM()
    retrieved = retrieve_fn(user_task)
    prompt = rapid_generation_prompt(user_task, retrieved)
    result = llm_client.complete(
        prompt,
        model=llm_model,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    code = result.content
    validation = validate_rapid_code(code, retrieved)
    trace = {
        "iterations": [
            {
                "kind": "generate",
                "query": user_task,
                "retrieved_count": len(retrieved),
                "validation_issues": validation.issues,
                "model": result.model,
                "requested_model": result.requested_model,
                "thinking": result.thinking,
                "reasoning_effort": result.reasoning_effort,
                "reasoning_content": result.reasoning_content,
            }
        ]
    }

    for _ in range(max(0, max_loop)):
        if validation.ok:
            break
        repair_query = rapid_repair_query(user_task, validation.issues)
        repair_retrieved = retrieve_fn(repair_query)
        loop_retrieved = _merge_retrieved(retrieved, repair_retrieved)
        prompt = rapid_repair_prompt(user_task, code, validation.issues, loop_retrieved)
        result = llm_client.complete(
            prompt,
            model=llm_model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        code = result.content
        retrieved = loop_retrieved
        validation = validate_rapid_code(code, retrieved)
        trace["iterations"].append(
            {
                "kind": "repair",
                "query": repair_query,
                "retrieved_count": len(repair_retrieved),
                "total_retrieved_count": len(retrieved),
                "validation_issues": validation.issues,
                "model": result.model,
                "requested_model": result.requested_model,
                "thinking": result.thinking,
                "reasoning_effort": result.reasoning_effort,
                "reasoning_content": result.reasoning_content,
            }
        )

    trace["ok"] = validation.ok
    trace["final_issues"] = validation.issues
    return code, retrieved, trace
