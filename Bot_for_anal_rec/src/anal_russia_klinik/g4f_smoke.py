from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .jsonio import write_json


def run_g4f_smoke(output_path: str | Path, model: str = "gpt-4o-mini") -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "provider": "g4f",
        "model": model,
        "ok": False,
        "response": "",
        "error": "",
        "elapsed_sec": 0.0,
    }
    try:
        from g4f.client import Client

        client = Client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say one short sentence in English."}],
            web_search=False,
        )
        result["response"] = response.choices[0].message.content
        result["ok"] = bool(str(result["response"]).strip())
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        finished = datetime.now(timezone.utc)
        result["finished_at"] = finished.isoformat()
        result["elapsed_sec"] = round((finished - started).total_seconds(), 3)
        write_json(output_path, result, indent=2)
    return result
