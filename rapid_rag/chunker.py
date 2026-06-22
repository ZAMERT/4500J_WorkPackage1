import hashlib
import re
import time
from pathlib import Path
from typing import List

from .loaders import html_files
from .parser import html_to_sections, parse_toc


PROGRESS_EVERY_FILES = 100


def split_text(text: str, max_chars: int = 1800, overlap: int = 250) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if len(text) >= 50 else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(";", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if len(chunk) >= 100:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
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

_S1 = {"Usage", "Arguments", "Program execution", "Error handling",
       "Return value", "Description", "Limitations", "Limitation",
       "Characteristics", "Components", "Structure"}
_S2 = {"Syntax", "Predefined data"}
_S3 = {"Basic examples", "Basic example", "More examples",
       "Examples", "Example", "Type examples"}
_SKIP = {"Related information", "Related Information", "About this manual",
         "Revisions", "Prerequisites", "Organization of chapters",
         "Who should read this manual?"}


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
        manual_start = time.monotonic()
        segment_starts = {seg: len(records[1]) for seg, records in segments.items()}
        print(f"{language}/{manual_name}: found {len(files)} HTML files", flush=True)

        for file_index, html_file in enumerate(files, start=1):
            rel_file = html_file.relative_to(manual_dir).as_posix()
            toc_title = toc_titles.get(html_file.name)
            try:
                sections = html_to_sections(html_file, toc_title)
            except Exception as exc:
                print(f"Skipping unreadable file {html_file}: {exc}", flush=True)
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
                        f"Language: {language}\n"
                        f"Title: {title}\n"
                        f"Section: {section}\n\n"
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

            if file_index % PROGRESS_EVERY_FILES == 0 or file_index == len(files):
                elapsed = time.monotonic() - manual_start
                segment_counts = ", ".join(
                    f"{seg}={len(records[1]) - segment_starts[seg]}"
                    for seg, records in segments.items()
                )
                print(
                    f"  chunk progress: {file_index}/{len(files)} files, "
                    f"{segment_counts} chunks from this manual, {elapsed:.1f}s elapsed",
                    flush=True,
                )

        elapsed = time.monotonic() - manual_start
        print(
            f"{language}/{manual_name}: finished chunking {len(files)} files in {elapsed:.1f}s",
            flush=True,
        )

    for seg, (ids, docs, _) in segments.items():
        print(f"Segment {seg}: {len(docs)} chunks", flush=True)
    return segments
