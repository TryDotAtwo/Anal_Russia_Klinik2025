from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AnalysisConfig
from .dashboard import write_dashboard
from .dictionaries import load_terms
from .exports import export_classified_csv, write_legacy_results
from .jsonio import read_json, write_json
from .llm import build_provider, classify_matches
from .manual_filters import apply_manual_filters, load_filter_dictionary
from .matcher import build_raw_matches
from .text import load_documents


@dataclass(frozen=True)
class PipelinePaths:
    output_dir: Path
    raw_matches: Path
    filtered_matches: Path
    rejected_matches: Path
    classified_matches: Path
    classified_csv: Path
    legacy_json: Path
    dashboard_html: Path


def output_paths(output_dir: str | Path) -> PipelinePaths:
    directory = Path(output_dir)
    return PipelinePaths(
        output_dir=directory,
        raw_matches=directory / "raw_matches.json",
        filtered_matches=directory / "filtered_matches.json",
        rejected_matches=directory / "rejected_matches.json",
        classified_matches=directory / "classified_matches.json",
        classified_csv=directory / "classified_matches.csv",
        legacy_json=directory / "Match_Clinick_results.json",
        dashboard_html=directory / "dashboard" / "index.html",
    )


def resolve_document_source(config: AnalysisConfig, explicit_source: str | Path | None = None) -> Path:
    if explicit_source:
        return Path(explicit_source)
    if config.clinical_json_path.exists():
        return config.clinical_json_path
    return config.pdf_folder


def load_limited_documents(source: str | Path, max_documents: int | None = None):
    documents = load_documents(source)
    return documents[:max_documents] if max_documents else documents


def run_pipeline(
    config: AnalysisConfig,
    document_source: str | Path | None = None,
    output_dir: str | Path | None = None,
    provider_name: str | None = None,
) -> PipelinePaths:
    paths = output_paths(output_dir or config.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_document_source(config, document_source)
    documents = load_limited_documents(source, config.max_documents)
    terms = load_terms(config.preparations_path, config.blacklist_path, config.markers_path)

    raw = [
        match.to_dict()
        for match in build_raw_matches(
            documents,
            terms,
            context_before=config.context_before,
            context_after=config.context_after,
        )
    ]
    write_json(paths.raw_matches, raw, indent=config.json_indent)

    decisions = load_filter_dictionary(config.manual_filters_path)
    kept, rejected = apply_manual_filters(raw, decisions)
    write_json(paths.filtered_matches, kept, indent=config.json_indent)
    write_json(paths.rejected_matches, rejected, indent=config.json_indent)

    provider = build_provider(provider_name or config.provider)
    classified = classify_matches(kept, provider)
    write_json(paths.classified_matches, classified, indent=config.json_indent)
    export_classified_csv(classified, paths.classified_csv)
    write_legacy_results(classified, paths.legacy_json, indent=config.json_indent)
    write_dashboard(paths.dashboard_html.parent, documents, classified, rejected)
    return paths


def run_match_stage(config: AnalysisConfig, document_source: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    documents = load_limited_documents(document_source, config.max_documents)
    terms = load_terms(config.preparations_path, config.blacklist_path, config.markers_path)
    raw = [
        match.to_dict()
        for match in build_raw_matches(
            documents,
            terms,
            context_before=config.context_before,
            context_after=config.context_after,
        )
    ]
    write_json(output_path, raw, indent=config.json_indent)
    return raw


def run_filter_stage(config: AnalysisConfig, input_path: str | Path, output_path: str | Path, rejected_path: str | Path) -> None:
    raw = read_json(input_path)
    decisions = load_filter_dictionary(config.manual_filters_path)
    kept, rejected = apply_manual_filters(raw, decisions)
    write_json(output_path, kept, indent=config.json_indent)
    write_json(rejected_path, rejected, indent=config.json_indent)


def run_classify_stage(config: AnalysisConfig, input_path: str | Path, output_path: str | Path, provider_name: str | None) -> None:
    kept = read_json(input_path)
    provider = build_provider(provider_name or config.provider)
    classified = classify_matches(kept, provider)
    write_json(output_path, classified, indent=config.json_indent)


def run_export_stage(
    config: AnalysisConfig,
    classified_path: str | Path,
    output_dir: str | Path,
    document_source: str | Path | None = None,
    rejected_path: str | Path | None = None,
) -> PipelinePaths:
    paths = output_paths(output_dir)
    classified = read_json(classified_path)
    rejected = read_json(rejected_path) if rejected_path and Path(rejected_path).exists() else []
    documents = load_limited_documents(resolve_document_source(config, document_source), config.max_documents)
    export_classified_csv(classified, paths.classified_csv)
    write_legacy_results(classified, paths.legacy_json, indent=config.json_indent)
    write_dashboard(paths.dashboard_html.parent, documents, classified, rejected)
    return paths
