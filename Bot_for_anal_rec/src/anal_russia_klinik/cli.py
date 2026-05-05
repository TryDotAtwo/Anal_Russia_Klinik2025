from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aho_report import write_aho_host_word_report
from .config import default_config
from .g4f_smoke import run_g4f_smoke
from .pipeline import run_classify_stage, run_export_stage, run_filter_stage, run_match_stage, run_pipeline


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def config_from_args(args: argparse.Namespace):
    overrides = {}
    for arg_name, field_name in (
        ("preparations", "preparations_path"),
        ("blacklist", "blacklist_path"),
        ("markers", "markers_path"),
        ("clinical_json", "clinical_json_path"),
        ("filter_file", "manual_filters_path"),
        ("output_dir", "output_dir"),
    ):
        value = getattr(args, arg_name, None)
        if value:
            overrides[field_name] = Path(value)
    if getattr(args, "context_before", None) is not None:
        overrides["context_before"] = args.context_before
    if getattr(args, "context_after", None) is not None:
        overrides["context_after"] = args.context_after
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "max_documents", None) is not None:
        overrides["max_documents"] = args.max_documents
    return default_config(**overrides)


def cmd_run(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    paths = run_pipeline(config, document_source=_path(args.texts), output_dir=args.output_dir, provider_name=args.provider)
    print(f"raw_matches={paths.raw_matches}")
    print(f"classified_matches={paths.classified_matches}")
    print(f"legacy_json={paths.legacy_json}")
    print(f"dashboard={paths.dashboard_html}")


def cmd_match(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    run_match_stage(config, args.texts or config.clinical_json_path, args.output)
    print(f"raw_matches={args.output}")


def cmd_filter(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    run_filter_stage(config, args.input, args.output, args.rejected_output)
    print(f"filtered_matches={args.output}")
    print(f"rejected_matches={args.rejected_output}")


def cmd_classify(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    run_classify_stage(config, args.input, args.output, args.provider)
    print(f"classified_matches={args.output}")


def cmd_export(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    paths = run_export_stage(config, args.input, args.output_dir, document_source=_path(args.texts), rejected_path=_path(args.rejected))
    print(f"legacy_json={paths.legacy_json}")
    print(f"dashboard={paths.dashboard_html}")


def cmd_aho_report(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    output_path = Path(args.output)
    report = write_aho_host_word_report(
        output_path,
        document_source=args.texts or config.clinical_json_path,
        preparations_path=config.preparations_path,
        blacklist_path=config.blacklist_path,
        markers_path=config.markers_path,
        max_documents=config.max_documents,
        workers=args.workers,
        partials_dir=args.partials_dir,
        indent=config.json_indent,
    )
    print(f"aho_host_words={output_path}")
    print(f"document_count={report['document_count']}")
    print(f"workers={report['workers']}")
    print(f"search_word_count={report['search_word_count']}")
    print(f"total_match_count={report['summary']['total_match_count']}")
    print(f"total_inside_word_match_count={report['summary']['total_inside_word_match_count']}")


def cmd_g4f_smoke(args: argparse.Namespace) -> None:
    result = run_g4f_smoke(args.output, model=args.model)
    print(f"g4f_smoke={args.output}")
    print(f"ok={result['ok']}")
    print(f"model={result['model']}")
    print(f"elapsed_sec={result['elapsed_sec']}")
    if result.get("error"):
        print(f"error={result['error']}")


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparations")
    parser.add_argument("--blacklist")
    parser.add_argument("--markers")
    parser.add_argument("--clinical-json")
    parser.add_argument("--filter-file")
    parser.add_argument("--context-before", type=int)
    parser.add_argument("--context-after", type=int)
    parser.add_argument("--max-documents", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anal-russia-klinik")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    add_common_paths(run)
    run.add_argument("--texts")
    run.add_argument("--provider", default="g4f", choices=["g4f", "openrouter", "fake"])
    run.add_argument("--output-dir", default="reports/default")
    run.set_defaults(func=cmd_run)

    match = subparsers.add_parser("match")
    add_common_paths(match)
    match.add_argument("--texts", required=True)
    match.add_argument("--output", required=True)
    match.set_defaults(func=cmd_match)

    filter_cmd = subparsers.add_parser("filter")
    add_common_paths(filter_cmd)
    filter_cmd.add_argument("--input", required=True)
    filter_cmd.add_argument("--output", required=True)
    filter_cmd.add_argument("--rejected-output", required=True)
    filter_cmd.set_defaults(func=cmd_filter)

    classify = subparsers.add_parser("classify")
    add_common_paths(classify)
    classify.add_argument("--input", required=True)
    classify.add_argument("--output", required=True)
    classify.add_argument("--provider", default="g4f", choices=["g4f", "openrouter", "fake"])
    classify.set_defaults(func=cmd_classify)

    export = subparsers.add_parser("export")
    add_common_paths(export)
    export.add_argument("--input", required=True)
    export.add_argument("--rejected")
    export.add_argument("--texts")
    export.add_argument("--output-dir", default="reports/default")
    export.set_defaults(func=cmd_export)

    aho_report = subparsers.add_parser("aho-report")
    add_common_paths(aho_report)
    aho_report.add_argument("--texts")
    aho_report.add_argument("--output", default="reports/aho/host_words_by_search_word.json")
    aho_report.add_argument("--workers", type=int, default=16)
    aho_report.add_argument("--partials-dir", default=None)
    aho_report.set_defaults(func=cmd_aho_report)

    g4f_smoke = subparsers.add_parser("g4f-smoke")
    g4f_smoke.add_argument("--model", default="gpt-4o-mini")
    g4f_smoke.add_argument("--output", default="reports/g4f/g4f_smoke.json")
    g4f_smoke.set_defaults(func=cmd_g4f_smoke)
    return parser


def main(argv: list[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        args_list = ["run"]
    parser = build_parser()
    args = parser.parse_args(args_list)
    args.func(args)
