from __future__ import annotations

import re
from pathlib import Path

from .jsonio import read_json
from .models import Document

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_-]+")
SECTION_RE = re.compile(r"(?:^|\n)\s*((?:\d+(?:\.\d+)*\.?|Приложение|Список литературы)[^\n]{0,160})")


def normalize_char(char: str) -> str:
    lowered = char.lower()
    return "е" if lowered == "ё" else lowered


def normalize_text(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    offset_map: list[int] = []
    for index, char in enumerate(text):
        normalized.append(normalize_char(char))
        offset_map.append(index)
    return "".join(normalized), offset_map


def normalize_query(text: str) -> str:
    return "".join(normalize_char(char) for char in text)


def containing_word(text: str, start: int, end: int) -> tuple[str, int, int, bool]:
    snippet = text[start:end]
    if any(char.isspace() for char in snippet):
        return snippet, start, end, False

    left = start
    while left > 0 and WORD_RE.match(text[left - 1]):
        left -= 1
    right = end
    while right < len(text) and WORD_RE.match(text[right]):
        right += 1

    host = text[left:right]
    return host, left, right, left != start or right != end


def context_window(text: str, start: int, end: int, before_chars: int, after_chars: int) -> tuple[str, str]:
    before = text[max(0, start - before_chars):start]
    after = text[end:min(len(text), end + after_chars)]
    return before, after


def section_for_position(text: str, start: int) -> str | None:
    section = None
    for match in SECTION_RE.finditer(text[:start]):
        section = " ".join(match.group(1).split())
    return section


def page_for_position(pages: tuple[tuple[int, int, int], ...], start: int) -> int | None:
    for page_number, page_start, page_end in pages:
        if page_start <= start < page_end:
            return page_number
    return None


def load_text_document(path: str | Path) -> Document:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    return Document(document_id=source.stem, title=source.stem, link="", path=str(source), text=text)


def load_pdf_document(path: str | Path) -> Document:
    import fitz

    source = Path(path)
    chunks: list[str] = []
    pages: list[tuple[int, int, int]] = []
    with fitz.open(source) as doc:
        for index, page in enumerate(doc, start=1):
            start = sum(len(chunk) for chunk in chunks)
            page_text = page.get_text("text")
            chunks.append(page_text)
            chunks.append("\n")
            pages.append((index, start, start + len(page_text)))
    text = "".join(chunks)
    return Document(document_id=source.stem, title=source.stem, link="", path=str(source), text=text, pages=tuple(pages))


def load_legacy_clinical_json(path: str | Path) -> list[Document]:
    source = Path(path)
    if not source.exists():
        return []
    data = read_json(source)
    recommendations = data.get("recommendations", data) if isinstance(data, dict) else {}
    documents: list[Document] = []
    if not isinstance(recommendations, dict):
        return documents
    for key, row in recommendations.items():
        if not isinstance(row, dict):
            continue
        text = str(row.get("Текст") or row.get("text") or "")
        if not text:
            continue
        documents.append(
            Document(
                document_id=str(key),
                title=str(row.get("Название") or row.get("title") or key),
                link=str(row.get("Ссылка") or row.get("link") or ""),
                path=str(source),
                text=text,
            )
        )
    return documents


def load_documents(path: str | Path) -> list[Document]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        return load_legacy_clinical_json(source)
    files = [source] if source.is_file() else sorted(source.rglob("*"))
    documents: list[Document] = []
    for item in files:
        suffix = item.suffix.lower()
        if suffix == ".txt":
            documents.append(load_text_document(item))
        elif suffix == ".pdf":
            documents.append(load_pdf_document(item))
    return documents
