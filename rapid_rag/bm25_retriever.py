from pathlib import Path
from typing import List, Optional

from .loaders import html_files
from .parser import html_to_sections, parse_toc


def _build_corpus(manuals: List[dict]) -> tuple[list[str], list[dict]]:
    """Returns (tokenized_docs, metadatas) by reading HTML files directly."""
    raw_docs = []
    metadatas = []

    for manual in manuals:
        language = manual["language"]
        manual_dir = Path(manual["manual_dir"])
        manual_name = manual["manual_name"]
        toc_titles = parse_toc(manual_dir / "toc.hhc")

        for html_file in html_files(manual_dir):
            rel_file = html_file.relative_to(manual_dir).as_posix()
            toc_title = toc_titles.get(html_file.name)
            try:
                sections = html_to_sections(html_file, toc_title)
            except Exception:
                continue

            section_counts = {}
            for section_info in sections:
                section = section_info["section"]
                section_instance = section_counts.get(section, 0)
                section_counts[section] = section_instance + 1
                text = section_info["text"].strip()
                if len(text) < 50:
                    continue
                raw_docs.append(text)
                metadatas.append({
                    "language": language,
                    "manual": manual_name,
                    "title": section_info["title"],
                    "section": section_info["section"],
                    "section_instance": section_instance,
                    "file": rel_file,
                })

    tokenized = [doc.lower().split() for doc in raw_docs]
    return tokenized, metadatas, raw_docs


class BM25Retriever:
    """Keyword retriever built on-the-fly from HTML files — no DB needed."""

    def __init__(self, manuals: List[dict]):
        from rank_bm25 import BM25Okapi

        print("Building BM25 index from HTML files...")
        tokenized, self.metadatas, self.raw_docs = _build_corpus(manuals)
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index ready: {len(self.raw_docs)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 12,
        language: Optional[str] = "en",
    ) -> List[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []
        for i in ranked:
            if scores[i] == 0:
                break
            meta = self.metadatas[i]
            if language and meta.get("language") != language:
                continue
            results.append({
                "document": self.raw_docs[i],
                "metadata": meta,
                "score": float(scores[i]),
                "distance": None,
            })
            if len(results) >= top_k:
                break

        return results
