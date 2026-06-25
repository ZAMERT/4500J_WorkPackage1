import html
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag


def clean_text(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def clean_inline_text(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def node_text(node: Tag) -> str:
    return clean_inline_text(node.get_text(" ", strip=True))


def code_lines(node: Tag) -> List[str]:
    lines = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = clean_inline_text(str(child))
            if text:
                lines.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        if "computerscripts" in child.get("class", []):
            lines.extend(code_lines(child))
        elif child.name == "p":
            text = node_text(child)
            if text:
                lines.append(text)
        else:
            lines.extend(code_lines(child))
    return lines


def table_lines(node: Tag) -> List[str]:
    lines = []
    for row in node.find_all("tr"):
        cells = [node_text(cell) for cell in row.find_all(["td", "th"], recursive=False)]
        cells = [cell for cell in cells if cell]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def block_lines(node: Tag) -> List[str]:
    lines = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = clean_inline_text(str(child))
            if text:
                lines.append(text)
            continue
        if not isinstance(child, Tag):
            continue

        classes = child.get("class", [])
        if "computerscripts" in classes:
            lines.extend(code_lines(child))
        elif "titled-block-title" in classes:
            text = node_text(child)
            if text:
                lines.append(text)
        elif child.name == "p":
            text = node_text(child)
            if text:
                lines.append(text)
        elif child.name == "table":
            lines.extend(table_lines(child))
        elif child.name in {"div", "section"}:
            lines.extend(block_lines(child))
        elif child.name == "br":
            continue
        else:
            text = node_text(child)
            if text:
                lines.append(text)
    return lines


def body_text(node: Tag) -> str:
    return "\n".join(line for line in block_lines(node) if line)


def parse_toc(toc_file: Path) -> Dict[str, str]:
    if not toc_file.exists():
        return {}

    soup = BeautifulSoup(toc_file.read_text(errors="ignore"), "html.parser")
    title_by_file: Dict[str, str] = {}

    for obj in soup.find_all("object"):
        name = None
        local = None
        for param in obj.find_all("param"):
            key = param.get("name", "").lower()
            value = param.get("value", "")
            if key == "name":
                name = clean_text(value)
            elif key == "local":
                local = value.replace("\\", "/")

        if name and local and local.lower().endswith((".html", ".htm")):
            title_by_file[Path(local).name] = name

    return title_by_file


def page_title(soup: BeautifulSoup, fallback: str) -> str:
    labels = soup.select("span.maplabel")
    if labels:
        title = node_text(labels[0])
        if title:
            return title

    title_tag = soup.find("title")
    if title_tag and node_text(title_tag):
        return node_text(title_tag)

    return fallback


def section_rows(soup: BeautifulSoup) -> Iterable[tuple[str, str]]:
    for label in soup.select("span.blocklabel"):
        label_cell = label.find_parent("td")
        if not label_cell:
            continue
        row = label_cell.find_parent("tr")
        if not row:
            continue
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        section = node_text(label)
        body = body_text(cells[1])
        if section and body:
            yield section, body


def html_to_sections(path: Path, toc_title: Optional[str]) -> List[dict]:
    raw = path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    title = toc_title or page_title(soup, path.stem)
    sections = []
    seen = set()

    for section, body in section_rows(soup):
        key = (section, body)
        if key in seen:
            continue
        seen.add(key)
        sections.append({"title": title, "section": section, "text": body})

    if not sections:
        body = node_text(soup.body if soup.body else soup)
        if body:
            sections.append({"title": title, "section": "Body", "text": body})

    return sections
