import json
import re
import time
from pathlib import Path
from typing import List, Optional

from .chunker import stable_id
from .loaders import html_files
from .parser import html_to_sections, parse_toc


CARD_COLLECTION = "rapid_cards"
PROGRESS_EVERY_FILES = 100

_EXAMPLE_SECTIONS = {
    "Basic examples",
    "Basic example",
    "More examples",
    "Examples",
    "Example",
    "Type examples",
}
_ERROR_SECTIONS = {"Error handling", "Error recovery"}
_RELATED_SECTIONS = {"Related information", "Related Information"}
_SYNTAX_SECTIONS = {"Syntax", "Predefined data"}
_ARGUMENT_SECTIONS = {"Arguments", "Components"}

_RAPID_CONTEXT_TYPES = {
    "bool",
    "confdata",
    "dionum",
    "errnum",
    "jointtarget",
    "loaddata",
    "num",
    "orient",
    "pos",
    "robtarget",
    "signalai",
    "signalao",
    "signaldi",
    "signaldo",
    "speeddata",
    "string",
    "switch",
    "syncident",
    "taskid",
    "tooldata",
    "triggdata",
    "wobjdata",
    "zonedata",
}


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


def _instruction_name(title: str) -> str:
    title = _clean_title(title)
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\b", title)
    return match.group(1) if match else title


def _instruction_type(sections: list[dict]) -> str:
    section_names = {section["section"] for section in sections}
    syntax = _section_text(sections, _SYNTAX_SECTIONS).lstrip().lower()
    joined = "\n".join(section["text"] for section in sections[:3]).lower()
    if section_names & {"Components", "Predefined data"}:
        return "data_type"
    if syntax.startswith("func"):
        return "function"
    if syntax.startswith("trap") or "trap routine" in joined:
        return "trap"
    return "instruction"


def _section_text(sections: list[dict], names: set[str]) -> str:
    parts = [section["text"].strip() for section in sections if section["section"] in names and section["text"].strip()]
    return "\n\n".join(parts)


def _section_list(sections: list[dict], names: set[str]) -> list[str]:
    return [section["text"].strip() for section in sections if section["section"] in names and section["text"].strip()]


def _required_context(*texts: str) -> list[str]:
    haystack = "\n".join(texts).lower()
    found = []
    for data_type in sorted(_RAPID_CONTEXT_TYPES):
        if re.search(rf"\b{re.escape(data_type.lower())}\b", haystack):
            found.append(data_type)
    return found


def _related(text: str) -> list[str]:
    names = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text):
        if token not in names:
            names.append(token)
    return names[:20]


def build_card_document(card: dict) -> str:
    return json.dumps(card, ensure_ascii=False, indent=2)


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def build_card_embedding_document(
    document: str,
    syntax_chars: int = 1000,
    argument_chars: int = 1400,
    usage_chars: int = 900,
    max_chars: int = 4000,
) -> str:
    """Build compact search text while preserving the full stored card."""
    card = json.loads(document)
    arguments = card.get("arguments") or []
    argument_text = _clip("\n".join(arguments), argument_chars)
    embedding_document = (
        f"Instruction: {card.get('instruction', '')}\n"
        f"Type: {card.get('type', '')}\n"
        f"Title: {card.get('title', '')}\n"
        f"Syntax:\n{_clip(card.get('syntax') or '', syntax_chars)}\n"
        f"Arguments:\n{argument_text}\n"
        f"Required context: {', '.join(card.get('required_context') or [])}\n"
        f"Related: {', '.join(card.get('related') or [])}\n"
        f"Usage:\n{_clip(card.get('usage') or '', usage_chars)}"
    )
    return _clip(embedding_document, max_chars)


def html_to_card(
    html_file: Path,
    toc_title: Optional[str],
    language: str,
    manual_name: str,
    manual_dir: Path,
) -> Optional[tuple[str, str, dict]]:
    sections = html_to_sections(html_file, toc_title)
    title = _clean_title(sections[0]["title"] if sections else (toc_title or html_file.stem))
    instruction = _instruction_name(title)
    rel_file = html_file.relative_to(manual_dir).as_posix()

    syntax = _section_text(sections, _SYNTAX_SECTIONS)
    arguments = _section_list(sections, _ARGUMENT_SECTIONS)
    examples = _section_list(sections, _EXAMPLE_SECTIONS)
    errors = _section_list(sections, _ERROR_SECTIONS)
    related_text = _section_text(sections, _RELATED_SECTIONS)
    usage = _section_text(sections, {"Usage", "Description", "Program execution", "Return value"})

    card = {
        "instruction": instruction,
        "type": _instruction_type(sections),
        "title": title,
        "syntax": syntax,
        "arguments": arguments,
        "required_context": _required_context(syntax, "\n".join(arguments), usage),
        "examples": examples,
        "related": _related(related_text),
        "common_errors": errors,
        "usage": usage,
        "source": {
            "language": language,
            "manual": manual_name,
            "file": rel_file,
            "path": str(html_file),
            "doc_version": "RobotWare 7.10",
        },
    }

    if not any([syntax, arguments, examples, errors]):
        return None

    metadata = {
        "language": language,
        "manual": manual_name,
        "title": title,
        "instruction": instruction,
        "type": card["type"],
        "file": rel_file,
        "path": str(html_file),
        "segment": "card",
        "doc_version": "RobotWare 7.10",
    }
    card_id = stable_id(language, manual_name, rel_file, "card", instruction)
    return card_id, build_card_document(card), metadata


def build_instruction_cards(manuals: List[dict]):
    ids = []
    docs = []
    metadatas = []
    total_files = 0

    for manual in manuals:
        language = manual["language"]
        manual_dir = Path(manual["manual_dir"])
        manual_name = manual["manual_name"]
        toc_titles = parse_toc(manual_dir / "toc.hhc")
        files = html_files(manual_dir)
        total_files += len(files)
        manual_start = time.monotonic()
        manual_cards_start = len(docs)
        print(
            f"{language}/{manual_name}: building canonical cards from {len(files)} HTML files",
            flush=True,
        )

        for file_index, html_file in enumerate(files, start=1):
            toc_title = toc_titles.get(html_file.name)
            try:
                card = html_to_card(html_file, toc_title, language, manual_name, manual_dir)
            except Exception as exc:
                print(f"Skipping card for unreadable file {html_file}: {exc}", flush=True)
                continue
            if card is None:
                if file_index % PROGRESS_EVERY_FILES == 0 or file_index == len(files):
                    elapsed = time.monotonic() - manual_start
                    print(
                        f"  cards progress: {file_index}/{len(files)} files, "
                        f"{len(docs) - manual_cards_start} cards from this manual, "
                        f"{elapsed:.1f}s elapsed",
                        flush=True,
                    )
                continue
            card_id, document, metadata = card
            ids.append(card_id)
            docs.append(document)
            metadatas.append(metadata)

            if file_index % PROGRESS_EVERY_FILES == 0 or file_index == len(files):
                elapsed = time.monotonic() - manual_start
                print(
                    f"  cards progress: {file_index}/{len(files)} files, "
                    f"{len(docs) - manual_cards_start} cards from this manual, "
                    f"{elapsed:.1f}s elapsed",
                    flush=True,
                )

        elapsed = time.monotonic() - manual_start
        print(
            f"{language}/{manual_name}: finished cards for {len(files)} files, "
            f"added {len(docs) - manual_cards_start} cards in {elapsed:.1f}s",
            flush=True,
        )

    print(f"Canonical cards: {len(docs)} from {total_files} HTML files", flush=True)
    return ids, docs, metadatas
