from __future__ import annotations

import hashlib

from .aho import AhoCorasickMatcher
from .models import Document, RawMatch, Term, TermVariant
from .text import containing_word, context_window, normalize_text, page_for_position, section_for_position


def raw_match_id(document_id: str, start: int, end: int, variant: TermVariant) -> str:
    digest = hashlib.sha1(
        f"{document_id}|{start}|{end}|{variant.term_id}|{variant.variant}".encode("utf-8")
    ).hexdigest()[:16]
    return f"raw:{digest}"


def build_raw_matches(
    documents: list[Document],
    terms: list[Term],
    context_before: int = 1200,
    context_after: int = 1200,
) -> list[RawMatch]:
    matcher = AhoCorasickMatcher(terms)
    records: list[RawMatch] = []
    for document in documents:
        records.extend(match_document(document, matcher, context_before, context_after))
    return deduplicate_raw_matches(records)


def match_document(
    document: Document,
    matcher: AhoCorasickMatcher,
    context_before_chars: int,
    context_after_chars: int,
) -> list[RawMatch]:
    normalized, offset_map = normalize_text(document.text)
    records: list[RawMatch] = []
    for start_norm, end_norm, variant in matcher.finditer(normalized):
        char_start = offset_map[start_norm]
        char_end = offset_map[end_norm - 1] + 1
        matched_text = document.text[char_start:char_end]
        word_text, word_start, word_end, inside_word = containing_word(document.text, char_start, char_end)
        before, after = context_window(document.text, char_start, char_end, context_before_chars, context_after_chars)
        records.append(
            RawMatch(
                raw_match_id=raw_match_id(document.document_id, char_start, char_end, variant),
                document_id=document.document_id,
                document_title=document.title,
                document_link=document.link,
                term_id=variant.term_id,
                source=variant.source,
                canonical=variant.canonical,
                variant=variant.variant,
                matched_text=matched_text,
                char_start=char_start,
                char_end=char_end,
                word_text=word_text,
                word_start=word_start,
                word_end=word_end,
                inside_word=inside_word,
                page=page_for_position(document.pages, char_start),
                section=section_for_position(document.text, char_start),
                context_before=before,
                context_after=after,
                metadata=variant.metadata,
            )
        )
    return records


def deduplicate_raw_matches(records: list[RawMatch]) -> list[RawMatch]:
    seen: set[tuple[str, int, int, str, str]] = set()
    output: list[RawMatch] = []
    for record in sorted(records, key=lambda item: (item.document_id, item.char_start, item.char_end, item.source)):
        key = (record.document_id, record.char_start, record.char_end, record.source, record.canonical.lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output
