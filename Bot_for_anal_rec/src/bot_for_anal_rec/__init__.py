"""Compatibility package for the previous project name."""

from __future__ import annotations

from pathlib import Path

_CANONICAL_PACKAGE = Path(__file__).resolve().parent.parent / "anal_russia_klinik"
if _CANONICAL_PACKAGE.exists():
    __path__.append(str(_CANONICAL_PACKAGE))

from anal_russia_klinik import __version__  # noqa: E402,F401
