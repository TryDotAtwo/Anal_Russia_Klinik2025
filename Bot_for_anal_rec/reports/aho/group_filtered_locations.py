from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anal_russia_klinik.aho_location_groups import write_location_group_report  # noqa: E402


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(prog="group-filtered-locations")
    parser.add_argument("--source", default=str(base_dir / "host_words_by_search_word_filtred.json"))
    parser.add_argument("--output", default=str(base_dir / "host_words_by_location_filtred.json"))
    args = parser.parse_args()
    report = write_location_group_report(args.source, args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "summary": report["summary"],
                "llm_review_unit": report["llm_review_unit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
