from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_value(name: str, legacy_name: str, default: str) -> str:
    return os.getenv(name) or os.getenv(legacy_name) or default


def _env_path(name: str, legacy_name: str, default: str) -> Path:
    value = _env_value(name, legacy_name, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class AnalysisConfig:
    project_root: Path = PROJECT_ROOT
    preparations_path: Path = _env_path("ANAL_RUSSIA_KLINIK_PREPARATIONS_PATH", "BOT_ANAL_PREPARATIONS_PATH", "data/input/drugs.json")
    blacklist_path: Path = _env_path("ANAL_RUSSIA_KLINIK_BLACKLIST_PATH", "BOT_ANAL_BLACKLIST_PATH", "data/input/blacklist_drugs.json")
    markers_path: Path = _env_path("ANAL_RUSSIA_KLINIK_MARKERS_PATH", "BOT_ANAL_MARKERS_PATH", "data/input/AXTUNG.Json")
    clinical_json_path: Path = _env_path("ANAL_RUSSIA_KLINIK_CLINICAL_JSON", "BOT_ANAL_CLINICAL_JSON", "data/input/clinical_recommendations.json")
    metadata_path: Path = _env_path("ANAL_RUSSIA_KLINIK_METADATA_PATH", "BOT_ANAL_METADATA_PATH", "data/input/MetaData.json")
    pdf_folder: Path = _env_path("ANAL_RUSSIA_KLINIK_PDF_FOLDER", "BOT_ANAL_PDF_FOLDER", "data/pdf")
    manual_filters_path: Path = _env_path("ANAL_RUSSIA_KLINIK_FILTERS_PATH", "BOT_ANAL_FILTERS_PATH", "config/manual_filters.csv")
    output_dir: Path = _env_path("ANAL_RUSSIA_KLINIK_OUTPUT_DIR", "BOT_ANAL_OUTPUT_DIR", "reports/default")
    context_before: int = int(_env_value("ANAL_RUSSIA_KLINIK_CONTEXT_BEFORE", "BOT_ANAL_CONTEXT_BEFORE", "1200"))
    context_after: int = int(_env_value("ANAL_RUSSIA_KLINIK_CONTEXT_AFTER", "BOT_ANAL_CONTEXT_AFTER", "1200"))
    provider: str = _env_value("ANAL_RUSSIA_KLINIK_PROVIDER", "BOT_ANAL_PROVIDER", "g4f")
    batch_size: int = int(_env_value("ANAL_RUSSIA_KLINIK_BATCH_SIZE", "BOT_ANAL_BATCH_SIZE", "1000"))
    json_indent: int = int(_env_value("ANAL_RUSSIA_KLINIK_JSON_INDENT", "BOT_ANAL_JSON_INDENT", "2"))
    max_documents: int | None = (
        int(_env_value("ANAL_RUSSIA_KLINIK_MAX_DOCUMENTS", "BOT_ANAL_MAX_DOCUMENTS", "0")) or None
    )


def default_config(**overrides: object) -> AnalysisConfig:
    values = AnalysisConfig().__dict__ | overrides
    return AnalysisConfig(**values)
