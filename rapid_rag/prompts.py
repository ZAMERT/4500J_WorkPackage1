def build_context(retrieved: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(retrieved, start=1):
        metadata = item["metadata"] or {}
        blocks.append(
            f"[Source {index}]\n"
            f"Manual directory: {metadata.get('manual')}\n"
            f"Language: {metadata.get('language')}\n"
            f"Title: {metadata.get('title')}\n"
            f"Section: {metadata.get('section')}\n"
            f"File: {metadata.get('file')}\n"
            f"Distance: {item.get('distance')}\n\n"
            f"{item['document']}"
        )
    return "\n\n---\n\n".join(blocks)


def rapid_generation_prompt(user_task: str, retrieved: list[dict]) -> str:
    context_text = build_context(retrieved)
    return f"""
You are an expert ABB RAPID programmer.

Generate valid ABB RAPID code based on the user's requirement.
Use only the retrieved RAPID manual context when possible.
If an instruction, function, data type, or argument is not supported by the context, do not invent it. Add a RAPID comment explaining the uncertainty instead.

Retrieved RAPID manual context:
{context_text}

User requirement:
{user_task}

Output requirements:
1. Return a complete RAPID module.
2. Use clear variable declarations.
3. Include concise RAPID comments for important logic or uncertainty.
4. Do not output Python, pseudocode, Markdown, or explanation.
5. Output RAPID code only.
""".strip()
