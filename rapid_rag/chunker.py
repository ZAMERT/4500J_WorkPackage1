import hashlib
import re
from pathlib import Path
from typing import List

from .loaders import html_files
from .parser import html_to_sections, parse_toc


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

            for section_info in sections:
                title = section_info["title"]
                section = section_info["section"]
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
                    ids.append(stable_id(language, manual_name, rel_file, section, str(chunk_index)))
                    docs.append(doc)
                    metadatas.append(
                        {
                            "language": language,
                            "manual": manual_name,
                            "title": title,
                            "section": section,
                            "file": rel_file,
                            "path": str(html_file),
                            "chunk_id": chunk_index,
                            "doc_version": "RobotWare 7.10",
                        }
                    )

    return ids, docs, metadatas
