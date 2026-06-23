from pathlib import Path
from typing import List

DEFAULT_MANUAL_ROOT = "rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation"
DEFAULT_DOC_DIRS = ("rapid_manual_html", "rapid_kernel", "rapid_overview")


def discover_manual_dirs(
    manual_root: str = DEFAULT_MANUAL_ROOT,
    languages: List[str] | None = None,
    doc_dirs: List[str] | None = None,
    manual_dir: str | None = None,
) -> List[dict]:
    if manual_dir:
        path = Path(manual_dir)
        language = path.parent.name if path.parent.name else "unknown"
        return [{"language": language, "manual_dir": path, "manual_name": path.name}]

    languages = languages or ["en"]
    doc_dirs = doc_dirs or list(DEFAULT_DOC_DIRS)
    root = Path(manual_root)
    manuals = []

    for language in languages:
        language_dir = root / language
        for doc_dir in doc_dirs:
            path = language_dir / doc_dir
            if path.exists():
                manuals.append({"language": language, "manual_dir": path, "manual_name": doc_dir})
            else:
                print(f"Skipping missing extracted HTML directory: {path}")

    return manuals


def html_files(manual_dir: Path) -> List[Path]:
    files = sorted(list(manual_dir.rglob("*.html")) + list(manual_dir.rglob("*.htm")))
    return [path for path in files if path.name.lower() not in {"toc.hhc"}]
