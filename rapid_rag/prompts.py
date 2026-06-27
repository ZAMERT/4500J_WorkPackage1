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

Agent engineering policy:
- Work as a closed-loop code generator: infer the required RAPID operations, ground each operation in retrieved manual evidence, then internally review the complete module before output.
- Prefer boring, verifiable RAPID over clever code. If evidence is thin, preserve correctness with a RAPID TODO comment instead of guessing.
- Treat the retrieved context as your current working memory. Do not rely on unsupported API memory when syntax or argument order matters.

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


def rapid_repair_prompt(
    user_task: str,
    previous_code: str,
    validation_issues: list[str],
    retrieved: list[dict],
) -> str:
    context_text = build_context(retrieved)
    issue_text = "\n".join(f"- {issue}" for issue in validation_issues) or "- No explicit validator issues."
    return f"""
You are an expert ABB RAPID programmer repairing generated RAPID code.

Repair the previous code so it satisfies the user's requirement and the validator issues.

Closed-loop repair policy:
- Use the retrieved RAPID manual context as the source of truth for instruction names, functions, data types, argument order, and required arguments.
- Check MODULE/ENDMODULE, PROC/ENDPROC, END statements, separators, required arguments, and data declarations.
- Remove Markdown fences, Python, pseudocode, and prose outside RAPID comments.
- If a RAPID instruction, data type, argument, I/O signal, target, tool, workobject, speed, or zone cannot be verified from the retrieved context or user request, keep the code valid and add a concise RAPID TODO comment.
- Do not fabricate coordinates, joint values, tool geometry, payload data, workobject frames, or signal names.
- Prefer a complete MODULE with PROC main() unless the user asked for a different routine shape.

Output policy:
- Output the repaired complete RAPID code only.
- Do not output Markdown fences or explanations outside RAPID comments.

Validator issues:
{issue_text}

Retrieved RAPID manual context:
{context_text}

User requirement:
{user_task}

Previous code:
{previous_code}
""".strip()


def rapid_repair_query(user_task: str, validation_issues: list[str]) -> str:
    issue_text = " ".join(validation_issues)
    return (
        f"{user_task} RAPID Syntax Arguments Usage Basic examples More examples "
        f"repair validation issues {issue_text}"
    ).strip()
