from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aho import AhoCorasickMatcher
from .dictionaries import load_terms
from .jsonio import write_json
from .models import Document, Term, TermVariant
from .text import containing_word, load_documents, normalize_query, normalize_text, page_for_position, section_for_position

EntryKey = tuple[str, str, str, str]
LOGGER = logging.getLogger("anal_russia_klinik.aho_report")
_WORKER_MATCHER: AhoCorasickMatcher | None = None


def configure_logging(verbose: bool = True) -> None:
    if logging.getLogger().handlers:
        return
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _variant_key(variant: TermVariant) -> EntryKey:
    return variant.source, variant.term_id, variant.canonical, variant.variant


def _new_entry(source: str, term_id: str, canonical: str, variant: str, normalized: str) -> dict[str, Any]:
    return {
        "source": source,
        "term_id": term_id,
        "canonical": canonical,
        "search_word": variant,
        "normalized_search_word": normalized,
        "match_count": 0,
        "inside_word_match_count": 0,
        "host_words": {},
        "inside_host_words": {},
    }


def _new_host(host_word: str, inside_word: bool) -> dict[str, Any]:
    return {"host_word": host_word, "inside_word": inside_word, "count": 0, "document_ids": set(), "occurrences": []}


def _occurrence(document: Document, start: int, end: int, word_start: int, word_end: int, matched_text: str) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "document_title": document.title,
        "document_link": document.link,
        "char_start": start,
        "char_end": end,
        "word_start": word_start,
        "word_end": word_end,
        "matched_text": matched_text,
        "page": page_for_position(document.pages, start),
        "section": section_for_position(document.text, start),
    }


def _add_host_match(entry: dict[str, Any], host_word: str, inside_word: bool, occurrence: dict[str, Any]) -> None:
    host = entry["host_words"].setdefault(host_word, _new_host(host_word, inside_word))
    host["count"] += 1
    host["document_ids"].add(occurrence["document_id"])
    host["occurrences"].append(occurrence)
    if inside_word:
        inside_host = entry["inside_host_words"].setdefault(host_word, _new_host(host_word, True))
        inside_host["count"] += 1
        inside_host["document_ids"].add(occurrence["document_id"])
        inside_host["occurrences"].append(occurrence)


def _init_worker(terms: list[Term]) -> None:
    global _WORKER_MATCHER
    _WORKER_MATCHER = AhoCorasickMatcher(terms)
    LOGGER.info("worker_initialized nodes=%s", len(_WORKER_MATCHER.nodes))


def _scan_chunk_worker(chunk_id: int, documents: list[Document], output_dir: str) -> dict[str, Any]:
    if _WORKER_MATCHER is None:
        raise RuntimeError("Aho matcher is not initialized in worker")
    partial_path = Path(output_dir) / f"chunk_{chunk_id:03d}.jsonl"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"chunk_id": chunk_id, "documents": len(documents), "chars": sum(len(item.text) for item in documents), "matches": 0}
    with partial_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            doc_matches = 0
            normalized_document, offset_map = normalize_text(document.text)
            for normalized_start, normalized_end, variant in _WORKER_MATCHER.finditer(normalized_document):
                start = offset_map[normalized_start]
                end = offset_map[normalized_end - 1] + 1
                matched_text = document.text[start:end]
                host_word, word_start, word_end, inside_word = containing_word(document.text, start, end)
                row = {
                    "key": list(_variant_key(variant)),
                    "entry": [variant.source, variant.term_id, variant.canonical, variant.variant, variant.normalized],
                    "host_word": host_word,
                    "inside_word": inside_word,
                    "occurrence": _occurrence(document, start, end, word_start, word_end, matched_text),
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                doc_matches += 1
            stats["matches"] += doc_matches
            LOGGER.info("chunk=%s document=%s chars=%s matches=%s", chunk_id, document.document_id, len(document.text), doc_matches)
    LOGGER.info("chunk_done chunk=%s documents=%s chars=%s matches=%s part=%s", chunk_id, stats["documents"], stats["chars"], stats["matches"], partial_path)
    stats["partial_path"] = str(partial_path)
    return stats


def _base_entries(terms: list[Term]) -> dict[EntryKey, dict[str, Any]]:
    entries: dict[EntryKey, dict[str, Any]] = {}
    for term in terms:
        for variant in term.variants:
            normalized = normalize_query(variant)
            if normalized:
                key = (term.source, term.term_id, term.canonical, variant)
                entries.setdefault(key, _new_entry(term.source, term.term_id, term.canonical, variant, normalized))
    return entries


def _chunk_documents(documents: list[Document], workers: int) -> list[list[Document]]:
    chunks: list[list[Document]] = [[] for _ in range(workers)]
    sizes = [0] * workers
    for document in sorted(documents, key=lambda item: len(item.text), reverse=True):
        index = sizes.index(min(sizes))
        chunks[index].append(document)
        sizes[index] += len(document.text)
    return [chunk for chunk in chunks if chunk]


def _merge_host(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["count"] += source["count"]
    target["document_ids"].update(source["document_ids"])
    target["occurrences"].extend(source["occurrences"])


def _merge_entry(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["match_count"] += source["match_count"]
    target["inside_word_match_count"] += source["inside_word_match_count"]
    for host_word, host in source["host_words"].items():
        _merge_host(target["host_words"].setdefault(host_word, _new_host(host_word, host["inside_word"])), host)
    for host_word, host in source["inside_host_words"].items():
        _merge_host(target["inside_host_words"].setdefault(host_word, _new_host(host_word, True)), host)


def _entry_from_row(row: dict[str, Any]) -> tuple[EntryKey, dict[str, Any]]:
    source, term_id, canonical, variant, normalized = row["entry"]
    key = tuple(row["key"])
    entry = _new_entry(source, term_id, canonical, variant, normalized)
    entry["match_count"] = 1
    if row["inside_word"]:
        entry["inside_word_match_count"] = 1
    _add_host_match(entry, row["host_word"], row["inside_word"], row["occurrence"])
    return key, entry


def _merge_partial_file(entries: dict[EntryKey, dict[str, Any]], path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            key, source_entry = _entry_from_row(json.loads(line))
            if key in entries:
                _merge_entry(entries[key], source_entry)
            else:
                entries[key] = source_entry
            count += 1
    LOGGER.info("merged_partial path=%s rows=%s", path, count)
    return count


def _serialize_host_words(values: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for host in values.values():
        rows.append(
            {
                "host_word": host["host_word"],
                "inside_word": host["inside_word"],
                "count": host["count"],
                "document_count": len(host["document_ids"]),
                "document_ids": sorted(host["document_ids"]),
                "occurrences": sorted(host["occurrences"], key=lambda item: (item["document_id"], item["char_start"], item["char_end"])),
            }
        )
    return sorted(rows, key=lambda row: (-row["count"], row["host_word"].lower()))


def _serialize_entries(entries: dict[EntryKey, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries.values():
        host_words = _serialize_host_words(entry["host_words"])
        inside_host_words = _serialize_host_words(entry["inside_host_words"])
        rows.append(
            {
                "source": entry["source"],
                "term_id": entry["term_id"],
                "canonical": entry["canonical"],
                "search_word": entry["search_word"],
                "normalized_search_word": entry["normalized_search_word"],
                "match_count": entry["match_count"],
                "inside_word_match_count": entry["inside_word_match_count"],
                "host_words_count": len(host_words),
                "inside_host_words_count": len(inside_host_words),
                "host_words": host_words,
                "inside_host_words": inside_host_words,
            }
        )
    return sorted(rows, key=lambda row: (row["source"], row["canonical"].lower(), row["search_word"].lower()))


def build_aho_host_word_report(
    documents: list[Document],
    terms: list[Term],
    *,
    workers: int = 1,
    partials_dir: str | Path = "reports/aho/partials",
) -> dict[str, Any]:
    entries = _base_entries(terms)
    worker_count = max(1, min(workers, len(documents) or 1))
    chunks = _chunk_documents(documents, worker_count)
    LOGGER.info("scan_start workers=%s chunks=%s documents=%s chars=%s terms=%s variants=%s", worker_count, len(chunks), len(documents), sum(len(d.text) for d in documents), len(terms), sum(len(t.variants) for t in terms))
    stats = []
    if worker_count == 1:
        _init_worker(terms)
        stats.append(_scan_chunk_worker(0, chunks[0], str(partials_dir)))
    else:
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker, initargs=(terms,)) as executor:
            futures = [executor.submit(_scan_chunk_worker, index, chunk, str(partials_dir)) for index, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                item = future.result()
                stats.append(item)
                LOGGER.info("progress chunks_done=%s/%s matches_done=%s", len(stats), len(chunks), sum(row["matches"] for row in stats))
    total_rows = 0
    for item in sorted(stats, key=lambda row: row["chunk_id"]):
        total_rows += _merge_partial_file(entries, item["partial_path"])
    rows = _serialize_entries(entries)
    return _report(rows, len(documents), worker_count, stats, total_rows)


def _report(rows: list[dict[str, Any]], document_count: int, workers: int, stats: list[dict[str, Any]], total_rows: int) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "document_count": document_count,
        "search_word_count": len(rows),
        "partial_rows_merged": total_rows,
        "chunks": sorted(stats, key=lambda row: row["chunk_id"]),
        "summary": {
            "matched_search_word_count": sum(1 for row in rows if row["match_count"]),
            "inside_matched_search_word_count": sum(1 for row in rows if row["inside_word_match_count"]),
            "total_match_count": sum(row["match_count"] for row in rows),
            "total_inside_word_match_count": sum(row["inside_word_match_count"] for row in rows),
        },
        "by_search_word": rows,
    }


def write_aho_host_word_report(
    output_path: str | Path,
    *,
    document_source: str | Path,
    preparations_path: str | Path,
    blacklist_path: str | Path,
    markers_path: str | Path,
    max_documents: int | None = None,
    workers: int = 16,
    partials_dir: str | Path | None = None,
    indent: int = 2,
) -> dict[str, Any]:
    configure_logging()
    output = Path(output_path)
    partials = Path(partials_dir) if partials_dir else output.parent / "partials"
    LOGGER.info("load_documents_start source=%s", document_source)
    documents = load_documents(document_source)
    if max_documents:
        documents = documents[:max_documents]
    LOGGER.info("load_documents_done documents=%s chars=%s", len(documents), sum(len(document.text) for document in documents))
    LOGGER.info("load_terms_start preparations=%s blacklist=%s markers=%s", preparations_path, blacklist_path, markers_path)
    terms = load_terms(preparations_path, blacklist_path, markers_path, expand_word_forms=False)
    LOGGER.info("load_terms_done terms=%s variants=%s", len(terms), sum(len(term.variants) for term in terms))
    report = build_aho_host_word_report(documents, terms, workers=workers, partials_dir=partials)
    LOGGER.info("write_report_start output=%s rows=%s matches=%s", output, report["search_word_count"], report["summary"]["total_match_count"])
    write_json(output, report, indent=indent)
    LOGGER.info("write_report_done output=%s bytes=%s", output, output.stat().st_size)
    return report
