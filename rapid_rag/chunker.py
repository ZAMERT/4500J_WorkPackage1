import hashlib
import re
from pathlib import Path
from typing import List

from .code_detector import find_code_blocks
from .loaders import html_files
from .parser import html_to_sections, parse_toc


def _code_block_at(pos: int, code_blocks: list[tuple[int, int]]) -> tuple[int, int] | None:
    for block_start, block_end in code_blocks:
        if block_start < pos < block_end:
            return block_start, block_end
    return None


def _safe_chunk_end(
    text: str,
    start: int,
    end: int,
    max_chars: int,
    code_blocks: list[tuple[int, int]],
) -> int:
    block = _code_block_at(end, code_blocks)
    if not block:
        return end

    block_start, block_end = block
    min_useful_end = start + max_chars // 2

    if block_start > min_useful_end:
        return block_start
    if block_end - start <= max_chars + max_chars // 2:
        return block_end
    return end


def _safe_next_start(start: int, end: int, overlap: int, code_blocks: list[tuple[int, int]]) -> int:
    next_start = max(0, end - overlap)
    block = _code_block_at(next_start, code_blocks)
    if not block:
        return next_start

    block_start, block_end = block
    if block_start > start:
        return block_start
    if block_end > start:
        return block_end
    return next_start


def split_text(text: str, max_chars: int = 1800, overlap: int = 250) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if len(text) >= 50 else []

    code_blocks = find_code_blocks(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(";", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
            end = _safe_chunk_end(text, start, end, max_chars, code_blocks)
        chunk = text[start:end].strip()
        if len(chunk) >= 100:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = _safe_next_start(start, end, overlap, code_blocks)
    return chunks


def stable_id(*parts: str) -> str:
    raw = "::".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:80].strip("_")
    return f"{slug}_{digest}"


def build_records(manuals: List[dict], chunk_chars: int = 1800, overlap: int = 250):
    ids = []
    docs = []
    metadatas = []

    for manual in manuals:
        language = manual["language"]
        manual_dir = Path(manual["manual_dir"])
        manual_name = manual["manual_name"]
        toc_titles = parse_toc(manual_dir / "toc.hhc")
        files = html_files(manual_dir)
        print(f"{language}/{manual_name}: found {len(files)} HTML files")

        for html_file in files:
            rel_file = html_file.relative_to(manual_dir).as_posix()
            toc_title = toc_titles.get(html_file.name)
            try:
                sections = html_to_sections(html_file, toc_title)
            except Exception as exc:
                print(f"Skipping unreadable file {html_file}: {exc}")
                continue

            section_counts = {}
            for section_info in sections:
                title = section_info["title"]
                section = section_info["section"]
                section_instance = section_counts.get(section, 0)
                section_counts[section] = section_instance + 1
                chunks = split_text(section_info["text"], max_chars=chunk_chars, overlap=overlap)
                for chunk_index, chunk in enumerate(chunks):
                    doc = (
                        f"Manual: ABB RAPID\n"
                        f"Manual directory: {manual_name}\n"
                        f"Language: {language}\n"
                        f"Title: {title}\n"
                        f"Section: {section}\n"
                        f"Source file: {rel_file}\n\n"
                        f"{chunk}"
                    )
                    ids.append(stable_id(language, manual_name, rel_file, section, str(section_instance), str(chunk_index)))
                    docs.append(doc)
                    metadatas.append(
                        {
                            "language": language,
                            "manual": manual_name,
                            "title": title,
                            "section": section,
                            "file": rel_file,
                            "path": str(html_file),
                            "section_instance": section_instance,
                            "chunk_id": chunk_index,
                            "doc_version": "RobotWare 7.10",
                        }
                    )

    return ids, docs, metadatas


# --- Segmented indexing (3 collections) ---

_S1 = {
    "Usage",
    "Arguments",
    "Program execution",
    "Error handling",
    "Return value",
    "Description",
    "Limitations",
    "Limitation",
    "Characteristics",
    "Components",
    "Structure",
    "Definition",
    "Introduction",
    "Programming principles",
    "Parameters",
    "Instructions",
    "Data",
    "General",
    "Evaluation and termination",
    "Comments",
    "Comments in a record",
    "Late binding",
    "Arithmetic expressions",
    "Logical expressions",
    "Stationary TCPs",
}
_S2 = {"Syntax", "Syntax rules", "Predefined data"}
_S3 = {"Basic examples", "Basic example", "More examples", "Examples", "Example", "Type examples"}
_SKIP = {
    "Related information",
    "Related Information",
    "About this manual",
    "Revisions",
    "Prerequisites",
    "Organization of chapters",
    "Who should read this manual?",
    "References",
}


def _classify(section: str) -> str | None:
    if section in _SKIP or not section:
        return None
    if section in _S1:
        return "s1"
    if section in _S2:
        return "s2"
    if section in _S3:
        return "s3"
    return "s1"  # unknown sections default to definitions


def build_records_segmented(manuals: List[dict], chunk_chars: int = 1800, overlap: int = 250) -> dict:
    """Returns {"s1": (ids, docs, metadatas), "s2": ..., "s3": ...}"""
    segments: dict[str, tuple[list, list, list]] = {
        "s1": ([], [], []),
        "s2": ([], [], []),
        "s3": ([], [], []),
    }

    for manual in manuals:
        language = manual["language"]
        manual_dir = Path(manual["manual_dir"])
        manual_name = manual["manual_name"]
        toc_titles = parse_toc(manual_dir / "toc.hhc")
        files = html_files(manual_dir)
        print(f"{language}/{manual_name}: found {len(files)} HTML files")

        for html_file in files:
            rel_file = html_file.relative_to(manual_dir).as_posix()
            toc_title = toc_titles.get(html_file.name)
            try:
                sections = html_to_sections(html_file, toc_title)
            except Exception as exc:
                print(f"Skipping unreadable file {html_file}: {exc}")
                continue

            section_counts = {}
            for section_info in sections:
                title = section_info["title"]
                section = section_info["section"]
                section_instance = section_counts.get(section, 0)
                section_counts[section] = section_instance + 1
                seg = _classify(section)
                if seg is None:
                    continue

                ids, docs, metadatas = segments[seg]
                chunks = split_text(section_info["text"], max_chars=chunk_chars, overlap=overlap)
                for chunk_index, chunk in enumerate(chunks):
                    doc = (
                        f"Manual: ABB RAPID\n"
                        f"Manual directory: {manual_name}\n"
                        f"Language: {language}\n"
                        f"Title: {title}\n"
                        f"Section: {section}\n"
                        f"Source file: {rel_file}\n\n"
                        f"{chunk}"
                    )
                    ids.append(stable_id(language, manual_name, rel_file, section, str(section_instance), str(chunk_index)))
                    docs.append(doc)
                    metadatas.append({
                        "language": language,
                        "manual": manual_name,
                        "title": title,
                        "section": section,
                        "file": rel_file,
                        "path": str(html_file),
                        "section_instance": section_instance,
                        "chunk_id": chunk_index,
                        "segment": seg,
                        "doc_version": "RobotWare 7.10",
                    })

    for seg, (ids, docs, _) in segments.items():
        print(f"Segment {seg}: {len(docs)} chunks")
    return segments
