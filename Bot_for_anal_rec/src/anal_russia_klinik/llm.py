from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import requests

from .models import LLMResult

LABELS = (
    "recommended",
    "not_recommended",
    "contraindication",
    "literature_reference",
    "definition_or_scale",
    "general_mention",
    "false_positive",
    "uncertain",
)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def complete_json(self, match: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class FakeProvider(LLMProvider):
    name = "fake"

    def complete_json(self, match: dict[str, Any]) -> dict[str, Any]:
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "evidence_quote": match.get("matched_text", ""),
            "reason_short": "fake_provider_no_external_call",
            "evidence_level": "",
            "recommendation_strength": "",
            "udd": "",
            "uur": "",
        }


class G4FProvider(LLMProvider):
    name = "g4f"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("G4F_MODEL", "gpt-4o-mini")

    def complete_json(self, match: dict[str, Any]) -> dict[str, Any]:
        from g4f.client import Client

        client = Client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_prompt(match)}],
            web_search=False,
        )
        try:
            content = response.choices[0].message.content
        except Exception:
            content = str(response)
        return parse_llm_content(content)


class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL")
        if not self.api_key or not self.model:
            raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL are required for provider=openrouter")

    def complete_json(self, match: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": build_prompt(match)}],
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return parse_llm_content(content)


def build_provider(name: str | None) -> LLMProvider:
    provider_name = (name or "g4f").lower()
    if provider_name == "fake":
        return FakeProvider()
    if provider_name == "g4f":
        return G4FProvider()
    if provider_name == "openrouter":
        return OpenRouterProvider()
    raise ValueError(f"Unknown LLM provider: {name}")


def build_prompt(match: dict[str, Any]) -> str:
    payload = {
        "labels": LABELS,
        "match": {
            "drug_or_marker": match.get("canonical"),
            "source": match.get("source"),
            "matched_text": match.get("matched_text"),
            "host_word": match.get("word_text"),
            "context": f"{match.get('context_before', '')}{match.get('matched_text', '')}{match.get('context_after', '')}",
            "section": match.get("section"),
        },
        "instruction": (
            "Classify a Russian clinical guideline mention. Return only JSON with keys: "
            "label, confidence, evidence_quote, reason_short, evidence_level, recommendation_strength, udd, uur. "
            "label must be one of labels. udd is evidence certainty level if present. "
            "uur is recommendation strength if present. Do not invent missing levels."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_llm_content(content: str) -> dict[str, Any]:
    content = content.strip()
    parsed = extract_json_object(content)
    if parsed is not None:
        return normalize_llm_response(parsed)
    return normalize_llm_response(parse_legacy_level_response(content))


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_legacy_level_response(text: str) -> dict[str, Any]:
    udd = _regex_value(text, r"(?:УДД|РЈР”Р”)\s*:\s*([^\n\r;]+)")
    uur = _regex_value(text, r"(?:УУР|РЈРЈР )\s*:\s*([^\n\r;]+)")
    mention_type = _regex_value(text, r"(?:Тип|РўРёРї)\s*:\s*([^\n\r;]+)")
    label = "uncertain"
    lowered = mention_type.lower()
    if "рекоменд" in lowered or "recommend" in lowered:
        label = "recommended"
    elif "противопоказ" in lowered or "contra" in lowered:
        label = "contraindication"
    elif "литера" in lowered or "ссыл" in lowered or "reference" in lowered:
        label = "literature_reference"
    elif "ошиб" in lowered or "false" in lowered:
        label = "false_positive"
    return {
        "label": label,
        "confidence": 0.0,
        "evidence_quote": "",
        "reason_short": mention_type,
        "evidence_level": udd,
        "recommendation_strength": uur,
        "udd": udd,
        "uur": uur,
    }


def _regex_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def normalize_llm_response(response: dict[str, Any]) -> dict[str, Any]:
    label = str(response.get("label", "uncertain")).strip()
    if label not in LABELS:
        label = "uncertain"
    try:
        confidence = float(response.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "label": label,
        "confidence": confidence,
        "evidence_quote": str(response.get("evidence_quote", "")),
        "reason_short": str(response.get("reason_short", "")),
        "evidence_level": str(response.get("evidence_level") or response.get("udd") or ""),
        "recommendation_strength": str(response.get("recommendation_strength") or response.get("uur") or ""),
        "udd": str(response.get("udd") or response.get("evidence_level") or ""),
        "uur": str(response.get("uur") or response.get("recommendation_strength") or ""),
    }


def classify_matches(items: list[dict[str, Any]], provider: LLMProvider) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        match = item["match"]
        response = provider.complete_json(match)
        result = LLMResult(
            raw_match_id=item["raw_match_id"],
            label=response["label"],
            confidence=response["confidence"],
            evidence_quote=response["evidence_quote"],
            reason_short=response["reason_short"],
            evidence_level=response["evidence_level"],
            recommendation_strength=response["recommendation_strength"],
            udd=response["udd"],
            uur=response["uur"],
            provider=provider.name,
        )
        results.append(
            {
                "raw_match_id": item["raw_match_id"],
                "filter_decision": item.get("filter_decision", {}),
                "match": match,
                "llm": result.to_dict(),
            }
        )
    return results
