from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anal_russia_klinik.llm_review_cases import write_review_cases_report  # noqa: E402


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base_dir = Path(__file__).resolve().parent
    aho_dir = project_root / "reports" / "aho"
    parser = argparse.ArgumentParser(prog="build-llm-review-cases")
    parser.add_argument("--locations", default=str(aho_dir / "host_words_by_location_filtred.json"))
    parser.add_argument("--clinical", default=str(project_root / "data" / "input" / "clinical_recommendations.json"))
    parser.add_argument("--output", default=str(base_dir / "llm_review_cases.json"))
    parser.add_argument("--window-chars", type=int, default=2500)
    parser.add_argument("--block-gap-chars", type=int, default=100)
    args = parser.parse_args()
    report = write_review_cases_report(args.locations, args.clinical, args.output, window_chars=args.window_chars, block_gap_chars=args.block_gap_chars)
    print(
        json.dumps(
            {
                "output": args.output,
                "summary": report["summary"],
                "context_window_chars_each_side": report["context_window_chars_each_side"],
                "nearby_block_gap_chars": report["nearby_block_gap_chars"],
                "llm_review_unit": report["llm_review_unit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
