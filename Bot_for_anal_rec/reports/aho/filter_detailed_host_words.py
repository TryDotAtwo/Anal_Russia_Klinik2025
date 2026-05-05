from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anal_russia_klinik.aho_compact import configure_logging  # noqa: E402
from anal_russia_klinik.aho_detailed_filter import write_filtered_detailed_report  # noqa: E402


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(prog="filter-detailed-host-words")
    parser.add_argument("--source", default=str(base_dir / "host_words_by_search_word.json"))
    parser.add_argument("--filters", default=str(base_dir / "host_word_filters.json"))
    parser.add_argument("--output", default=str(base_dir / "host_words_by_search_word_filtred.json"))
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=1000)
    args = parser.parse_args()
    configure_logging()
    result = write_filtered_detailed_report(
        args.source,
        args.filters,
        args.output,
        drop_empty=not args.keep_empty,
        progress_interval=args.progress_interval,
    )
    print(json.dumps({"output": args.output, **{key: result[key] for key in ("search_word_count", "filter_host_word_count", "removed_match_count", "summary")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
