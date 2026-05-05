from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    link: str
    path: str
    text: str
    pages: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Term:
    term_id: str
    source: str
    canonical: str
    variants: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TermVariant:
    term_id: str
    source: str
    canonical: str
    variant: str
    normalized: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawMatch:
    raw_match_id: str
    document_id: str
    document_title: str
    document_link: str
    term_id: str
    source: str
    canonical: str
    variant: str
    matched_text: str
    char_start: int
    char_end: int
    word_text: str
    word_start: int
    word_end: int
    inside_word: bool
    page: int | None
    section: str | None
    context_before: str
    context_after: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FilterDecision:
    term: str
    host_word: str
    action: str
    reason: str = ""
    confidence: float | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMResult:
    raw_match_id: str
    label: str
    confidence: float
    evidence_quote: str
    reason_short: str
    evidence_level: str
    recommendation_strength: str
    udd: str
    uur: str
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
