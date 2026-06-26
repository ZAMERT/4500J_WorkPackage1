import re
from dataclasses import dataclass


DECLARATION_KEYWORDS = {
    "ALIAS",
    "CONST",
    "FUNC",
    "LOCAL",
    "MODULE",
    "PERS",
    "PROC",
    "RECORD",
    "TASK",
    "TRAP",
    "VAR",
}

CONTROL_KEYWORDS = {
    "BACKWARD",
    "CASE",
    "CONNECT",
    "DEFAULT",
    "ELSE",
    "ELSEIF",
    "ENDFOR",
    "ENDFUNC",
    "ENDIF",
    "ENDMODULE",
    "ENDPROC",
    "ENDRECORD",
    "ENDTEST",
    "ENDTRAP",
    "ENDWHILE",
    "ERROR",
    "EXIT",
    "FOR",
    "FROM",
    "GOTO",
    "IF",
    "INOUT",
    "RAISE",
    "RETRY",
    "RETURN",
    "TEST",
    "THEN",
    "TO",
    "TRYNEXT",
    "UNDO",
    "WHILE",
    "WITH",
}

BUILTIN_DATA = {
    "FALSE",
    "TRUE",
    "fine",
    "tool0",
    "v100",
    "v200",
    "wobj0",
    "z10",
    "z50",
}


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]


def _strip_rapid_comment(line: str) -> str:
    return line.split("!", 1)[0]


def _contains_todo_for(name: str, code: str) -> bool:
    for line in code.splitlines():
        lower = line.lower()
        if "!" in line and ("todo" in lower or "verify" in lower) and name.lower() in lower:
            return True
    return False


def _evidence_terms(retrieved: list[dict]) -> set[str]:
    terms = set()
    for item in retrieved:
        metadata = item.get("metadata") or {}
        for value in metadata.values():
            if isinstance(value, str):
                terms.update(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", value))
        terms.update(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", item.get("document") or ""))
    return {term.lower() for term in terms}


def _called_symbols(code: str) -> list[tuple[int, str]]:
    calls = []
    for line_no, line in enumerate(code.splitlines(), start=1):
        stripped = _strip_rapid_comment(line).strip()
        if not stripped or stripped.startswith(("%", "#")):
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\b", stripped)
        if not match:
            continue
        symbol = match.group(1)
        upper = symbol.upper()
        if upper in DECLARATION_KEYWORDS or upper in CONTROL_KEYWORDS:
            continue
        if symbol in BUILTIN_DATA:
            continue
        if ":=" in stripped:
            continue
        calls.append((line_no, symbol))
    return calls


def validate_rapid_code(code: str, retrieved: list[dict]) -> ValidationResult:
    issues = []
    stripped = code.strip()
    if not stripped:
        issues.append("Output is empty.")
        return ValidationResult(ok=False, issues=issues)

    if "```" in code:
        issues.append("Output contains Markdown code fences.")

    first_line = next((line.strip() for line in code.splitlines() if line.strip()), "")
    if first_line and not first_line.upper().startswith("MODULE"):
        issues.append("First non-empty line should start with MODULE.")

    if not re.search(r"^\s*MODULE\s+\w+", code, re.IGNORECASE | re.MULTILINE):
        issues.append("Missing MODULE declaration.")
    if not re.search(r"^\s*ENDMODULE\b", code, re.IGNORECASE | re.MULTILINE):
        issues.append("Missing ENDMODULE.")
    if not re.search(r"^\s*PROC\s+\w+\s*\(", code, re.IGNORECASE | re.MULTILINE):
        issues.append("Missing PROC declaration.")
    if not re.search(r"^\s*ENDPROC\b", code, re.IGNORECASE | re.MULTILINE):
        issues.append("Missing ENDPROC.")

    prose_markers = ("here is", "below is", "this code", "```", "rapid code")
    for line_no, line in enumerate(code.splitlines(), start=1):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("!"):
            continue
        if any(marker in stripped_line.lower() for marker in prose_markers):
            issues.append(f"Line {line_no} appears to contain prose outside a RAPID comment.")
            break

    terms = _evidence_terms(retrieved)
    for line_no, symbol in _called_symbols(code):
        if symbol.lower() not in terms and not _contains_todo_for(symbol, code):
            issues.append(
                f"Line {line_no} calls '{symbol}' but retrieved evidence does not mention it; "
                "add evidence-backed syntax or a RAPID TODO comment."
            )

    return ValidationResult(ok=not issues, issues=issues)
