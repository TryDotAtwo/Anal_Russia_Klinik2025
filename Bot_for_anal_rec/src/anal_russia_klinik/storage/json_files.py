from __future__ import annotations

from pathlib import Path
from typing import Any

from ..jsonio import read_json, write_json


def load_json(path: str | Path) -> Any:
    return read_json(path)


def save_json(path: str | Path, data: Any, indent: int = 2) -> None:
    write_json(path, data, indent=indent)


def load_preparations_list(path: str | Path) -> list[str]:
    data = read_json(path)
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict) and isinstance(data.get("preparations"), list):
        return [str(item) for item in data["preparations"]]
    return []
