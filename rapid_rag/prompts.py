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

Generate valid ABB RAPID code for the user's requirement.

Evidence policy:
- Use the retrieved RAPID manual context as the source of truth for instruction names, functions, data types, argument order, and required arguments.
- Prefer evidence from sections named Syntax, Arguments, Usage, Basic examples, More examples, Description, Definition, and Programming principles.
- If an instruction, function, data type, argument, or behavior is not supported by the retrieved context, do not invent it. Add a concise RAPID comment explaining what must be verified.

Symbol policy:
- Do not invent coordinates, joint values, tool geometry, workobject frames, payload data, or I/O signal names that the user did not provide.
- For user-named site data such as robtarget, jointtarget, tooldata, wobjdata, speeddata, zonedata, or signal names, reference the provided name when possible.
- If required site data is missing, add a RAPID TODO comment saying it must be defined in the robot system or configuration. Do not create fake placeholder values such as all-zero robtargets.

Built-in and common system data policy:
- Treat named ABB/common system data as existing references unless the user explicitly asks to define custom data.
- Do not redeclare common predefined values or conventional system objects such as v100, v200, fine, z10, z50, tool0, or wobj0 unless custom values are explicitly requested.
- If the user asks for a custom speed, zone, tool, workobject, target, or signal and provides enough values, declare it. If values are missing, leave a TODO comment instead of fabricating values.

Syntax policy:
- Before producing the final code, internally check every RAPID instruction call against the retrieved Syntax, Arguments, Usage, and examples.
- Verify instruction argument order, required arguments, data types, separators, module/procedure structure, and END statements.
- Prefer the simplest valid RAPID structure that satisfies the requirement.

Module policy:
- Return a complete RAPID module using MODULE ... ENDMODULE and a main procedure using PROC main() ... ENDPROC unless the user asks for a different routine shape.
- Declare only data that is required, non-predefined, and sufficiently specified by the user or context.
- Use concise RAPID comments for assumptions, missing site data, or uncertainty.

Output policy:
- Output RAPID code only.
- Do not output Markdown fences, Python, pseudocode, explanations, or prose outside RAPID comments.

Retrieved RAPID manual context:
{context_text}

User requirement:
{user_task}
""".strip()
