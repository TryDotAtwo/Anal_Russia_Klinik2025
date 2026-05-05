from __future__ import annotations

import hashlib
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

from .jsonio import read_json
from .models import Term
from .text import normalize_query

RU_ENDINGS = (
    "",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "и",
    "ы",
    "ой",
    "ою",
    "ом",
    "ем",
    "ами",
    "ями",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ого",
    "ему",
    "ому",
    "ым",
    "им",
    "ых",
    "их",
)


def stable_term_id(source: str, canonical: str) -> str:
    digest = hashlib.sha1(f"{source}|{canonical}".encode("utf-8")).hexdigest()[:12]
    return f"{source}:{digest}"


def simple_word_forms(word: str) -> set[str]:
    value = word.lower().strip()
    if not value:
        return set()
    forms = {value}
    if not value.replace("-", "").isalpha() or len(value) <= 3:
        return forms
    stems = {value}
    for ending in sorted(RU_ENDINGS[1:], key=len, reverse=True):
        if value.endswith(ending) and len(value) > len(ending) + 2:
            stems.add(value[: -len(ending)])
            break
    for stem in stems:
        for ending in RU_ENDINGS:
            if stem:
                forms.add(stem + ending)
    return forms


def searchable_length(value: str) -> int:
    normalized = normalize_query(value).strip()
    return sum(1 for char in normalized if char.isalnum())


def keep_search_variant(variant: str, origin: str, *, generated: bool) -> bool:
    length = searchable_length(variant)
    if length >= 3:
        return True
    if not generated and length == 2 and searchable_length(origin) == 2:
        return True
    return False


def variants_for_term(canonical: str, aliases: Iterable[str] = (), *, expand_word_forms: bool = True) -> tuple[str, ...]:
    values: dict[str, str] = {}

    def add_variant(value: str) -> None:
        key = normalize_query(value)
        if key and key not in values:
            values[key] = value

    for value in chain([canonical], aliases):
        cleaned = str(value).strip()
        if not cleaned:
            continue
        if keep_search_variant(cleaned, cleaned, generated=False):
            add_variant(cleaned)
        if not expand_word_forms:
            continue
        parts = cleaned.split()
        if len(parts) == 1:
            for form in simple_word_forms(cleaned):
                if keep_search_variant(form, cleaned, generated=form != cleaned):
                    add_variant(form)
        else:
            joined = " ".join(parts)
            if keep_search_variant(joined, cleaned, generated=False):
                add_variant(joined)
            for part in parts:
                for form in simple_word_forms(part):
                    if keep_search_variant(form, cleaned, generated=True):
                        add_variant(form)
    return tuple(sorted(values.values(), key=lambda item: (len(item), item)))


def _dict_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _string_values(*values: Any) -> list[str]:
    output: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            output.append(value.strip())
        elif isinstance(value, list):
            output.extend(str(item).strip() for item in value if str(item).strip())
    return output


def _term_from_blacklist_entry(entry: dict[str, Any], *, expand_word_forms: bool = True) -> Term | None:
    canonical = str(_dict_value(entry, "Название препарата", "canonical", "name", "term") or "").strip()
    if not canonical:
        return None
    aliases_raw = _dict_value(entry, "Альтернативные названия", "aliases", "variants") or []
    aliases = [str(item).strip() for item in aliases_raw if str(item).strip()]
    metadata = {
        "description": str(_dict_value(entry, "Описание", "description") or ""),
        "entry": entry,
    }
    return Term(
        stable_term_id("blacklist", canonical),
        "blacklist",
        canonical,
        variants_for_term(canonical, aliases, expand_word_forms=expand_word_forms),
        metadata,
    )


def load_blacklist_terms(path: str | Path, *, expand_word_forms: bool = True) -> list[Term]:
    source = Path(path)
    if not source.exists():
        return []
    data = read_json(source)
    rows = data.get("terms", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    terms = [_term_from_blacklist_entry(row, expand_word_forms=expand_word_forms) for row in rows if isinstance(row, dict)]
    return [term for term in terms if term is not None]


def load_marker_terms(path: str | Path, *, expand_word_forms: bool = True) -> list[Term]:
    source = Path(path)
    if not source.exists():
        return []
    data = read_json(source)
    rows = data.get("terms", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    terms: list[Term] = []
    for value in rows:
        canonical = str(value.get("canonical") if isinstance(value, dict) else value).strip()
        if canonical:
            terms.append(
                Term(
                    stable_term_id("marker", canonical),
                    "marker",
                    canonical,
                    variants_for_term(canonical, expand_word_forms=expand_word_forms),
                )
            )
    return terms


def load_preparation_terms(path: str | Path, *, expand_word_forms: bool = True) -> list[Term]:
    source = Path(path)
    if not source.exists():
        return []
    data = read_json(source)
    rows = data.get("preparations", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    terms: list[Term] = []
    for row in rows:
        if isinstance(row, str):
            canonical = row.strip()
            aliases: list[str] = []
            metadata: dict[str, Any] = {}
        elif isinstance(row, dict):
            drug = row.get("drug") if isinstance(row.get("drug"), dict) else {}
            canonical = str(drug.get("name") or row.get("name") or row.get("canonical") or row.get("term") or "").strip()
            aliases = _string_values(
                row.get("aliases", []),
                row.get("variants", []),
                drug.get("aliases", []),
            )
            metadata = {"entry": row}
        else:
            continue
        if canonical:
            terms.append(
                Term(
                    stable_term_id("mediq", canonical),
                    "mediq",
                    canonical,
                    variants_for_term(canonical, aliases, expand_word_forms=expand_word_forms),
                    metadata,
                )
            )
    return terms


def load_terms(
    preparations_path: str | Path,
    blacklist_path: str | Path,
    markers_path: str | Path,
    *,
    expand_word_forms: bool = True,
) -> list[Term]:
    return deduplicate_terms(
        [
            *load_preparation_terms(preparations_path, expand_word_forms=expand_word_forms),
            *load_blacklist_terms(blacklist_path, expand_word_forms=expand_word_forms),
            *load_marker_terms(markers_path, expand_word_forms=expand_word_forms),
        ]
    )


def deduplicate_terms(terms: Iterable[Term]) -> list[Term]:
    seen: set[tuple[str, str]] = set()
    output: list[Term] = []
    for term in terms:
        key = (term.source, normalize_query(term.canonical))
        if key in seen:
            continue
        seen.add(key)
        output.append(term)
    return output
