from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anal_russia_klinik.aho_compact import configure_logging, write_filtered_report  # noqa: E402
from anal_russia_klinik.aho_filter_store import merge_filter_words  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(prog="apply-host-word-filters")
    parser.add_argument("--compact", default=str(Path(__file__).with_name("host_words_compact.json")))
    parser.add_argument("--filters", default=str(Path(__file__).with_name("host_word_filters.json")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("host_words_filtered_compact.json")))
    args = parser.parse_args()
    configure_logging()
    merge_filter_words(args.filters)
    report = write_filtered_report(args.compact, args.filters, args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "filter_host_word_count": report["filter_host_word_count"],
                "removed_match_count": report["removed_match_count"],
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
