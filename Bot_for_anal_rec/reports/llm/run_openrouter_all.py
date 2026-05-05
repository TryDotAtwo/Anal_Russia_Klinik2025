from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anal_russia_klinik.llm_review_openrouter import run_openrouter_all  # noqa: E402


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(prog="run-openrouter-all")
    parser.add_argument("--review-cases", default=str(base_dir / "llm_review_cases.json"))
    parser.add_argument("--gold", default=str(base_dir / "llm_gold_40.json"))
    parser.add_argument("--output", default=str(base_dir / "openrouter_all_results.json"))
    parser.add_argument("--excluded-preparations", default=str(base_dir / "excluded_preparations.json"))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / "config" / "openrouter.env"))
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run_openrouter_all(
        args.review_cases,
        args.output,
        gold_path=args.gold,
        env_file=args.env_file,
        model=args.model,
        excluded_preparations_path=args.excluded_preparations,
        limit=args.limit,
        log_progress=not args.quiet,
        save_each=True,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "new_completed": report["new_completed"],
                "completed_all": report["completed_all"],
                "completed_visible": report["completed_visible"],
                "gold_missing_count": report["gold_missing_count"],
                "score": report["score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
