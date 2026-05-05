from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .aho_compact import (
    configure_logging,
    normalize_host_word,
    write_compact_report,
    write_filtered_report,
)
from .aho_dashboard_rows import row_payload
from .aho_filter_store import (
    DEFAULT_SHARD_SIZE,
    merge_filter_words,
    page_filter_words,
    read_filter_words,
    shard_filter_words,
    update_filter_words,
)
from .jsonio import read_json
from .aho_filter_dashboard_html import DASHBOARD_HTML

LOGGER = logging.getLogger("anal_russia_klinik.aho_filter_server")


def default_paths(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir).resolve()
    return {
        "base": base,
        "source": base / "host_words_by_search_word.json",
        "compact": base / "host_words_compact.json",
        "filters": base / "host_word_filters.json",
        "filters_parts": base / "host_word_filters_parts",
        "filtered": base / "host_words_filtered_compact.json",
        "dashboard": base / "dashboard" / "index.html",
    }


def write_dashboard_html(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(DASHBOARD_HTML, encoding="utf-8")


class DashboardState:
    def __init__(self, base_dir: str | Path) -> None:
        self.paths = default_paths(base_dir)
        self.compact: dict[str, Any] | None = None
        self.filters: set[str] = set()
        self.host_word_counts: dict[str, int] = {}

    def ensure(self, *, rebuild_compact: bool = False, rebuild_filtered: bool = False) -> None:
        if rebuild_compact or not self.paths["compact"].exists():
            write_compact_report(self.paths["source"], self.paths["compact"])
        if not self.paths["filters_parts"].exists():
            shard_filter_words(self.paths["filters"], read_filter_words(self.paths["filters"]))
        if rebuild_filtered or not self.paths["filtered"].exists():
            merge_filter_words(self.paths["filters"])
            write_filtered_report(self.paths["compact"], self.paths["filters"], self.paths["filtered"])
        write_dashboard_html(self.paths["dashboard"])
        self.reload()

    def reload(self) -> None:
        self.compact = read_json(self.paths["compact"])
        self.filters = read_filter_words(self.paths["filters"])
        self.host_word_counts = self._build_host_word_counts()
        LOGGER.info("dashboard_data_loaded entries=%s filters=%s", len(self.rows), len(self.filters))

    @property
    def rows(self) -> list[dict[str, Any]]:
        if not self.compact:
            return []
        return list(self.compact.get("by_search_word", []))

    def state_payload(self) -> dict[str, Any]:
        compact_size = self.paths["compact"].stat().st_size if self.paths["compact"].exists() else 0
        filtered_size = self.paths["filtered"].stat().st_size if self.paths["filtered"].exists() else 0
        filter_dirty = self._filter_dirty()
        return {
            "paths": {key: str(value) for key, value in self.paths.items() if key != "base"},
            "summary": (self.compact or {}).get("summary", {}),
            "filter_host_word_count": len(self.filters),
            "filter_shard_size": DEFAULT_SHARD_SIZE,
            "filter_shard_count": self._filter_shard_count(),
            "filter_dirty": filter_dirty,
            "estimated_removed_match_count": self.estimated_removed_match_count(),
            "compact_size_bytes": compact_size,
            "filtered_size_bytes": filtered_size,
        }

    def update_filter(self, host_word: str, enabled: bool) -> dict[str, Any]:
        normalized = normalize_host_word(host_word)
        if not normalized:
            raise ValueError("empty_host_word")
        result = update_filter_words(self.paths["filters"], [normalized], enabled=enabled)
        self.filters = read_filter_words(self.paths["filters"])
        return {
            "host_word": normalized,
            "enabled": normalized in self.filters,
            "filter_host_word_count": len(self.filters),
            "filter_shard_count": result["shard_count"],
            "filter_dirty": self._filter_dirty(),
            "estimated_removed_match_count": self.estimated_removed_match_count(),
            "filtered_path": str(self.paths["filtered"]),
        }

    def update_filter_bulk(self, host_words: list[str], enabled: bool) -> dict[str, Any]:
        clean_words = [word for word in (normalize_host_word(item) for item in host_words) if word]
        result = update_filter_words(self.paths["filters"], clean_words, enabled=enabled)
        self.filters = read_filter_words(self.paths["filters"])
        return {
            "requested_count": len(clean_words),
            "enabled": enabled,
            "filter_host_word_count": len(self.filters),
            "filter_shard_count": result["shard_count"],
            "changed_count": result["changed_count"],
            "filter_dirty": self._filter_dirty(),
            "estimated_removed_match_count": self.estimated_removed_match_count(),
        }

    def add_search_results_to_filter(self, query: dict[str, Any]) -> dict[str, Any]:
        words = self._collect_host_words_for_query(query)
        return {**self.update_filter_bulk(words, True), "collected_host_word_count": len(set(words))}

    def add_row_host_words_to_filter(self, row_key: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
        words = self._collect_host_words_for_row(row_key, query)
        return {**self.update_filter_bulk(words, True), "collected_host_word_count": len(set(words))}

    def rebuild_filtered(self) -> dict[str, Any]:
        merge_filter_words(self.paths["filters"])
        report = write_filtered_report(self.paths["compact"], self.paths["filters"], self.paths["filtered"])
        self.filters = read_filter_words(self.paths["filters"])
        return {
            "filter_host_word_count": len(self.filters),
            "removed_match_count": report["removed_match_count"],
            "filtered_summary": report["summary"],
            "filtered_path": str(self.paths["filtered"]),
        }

    def filters_page(self, query: dict[str, list[str]]) -> dict[str, Any]:
        offset = max(0, int(_one(query, "offset", "0") or 0))
        limit = min(500, max(1, int(_one(query, "limit", "200") or 200)))
        return page_filter_words(self.paths["filters"], q=_one(query, "q", ""), offset=offset, limit=limit)

    def search(self, query: dict[str, list[str]]) -> dict[str, Any]:
        text = normalize_host_word(_one(query, "q", ""))
        host_text = normalize_host_word(_one(query, "host", ""))
        source = _one(query, "source", "")
        inside = _one(query, "inside", "any")
        filtered = _one(query, "filtered", "all")
        offset = max(0, int(_one(query, "offset", "0") or 0))
        limit = min(500, max(1, int(_one(query, "limit", "100") or 100)))
        host_limit = min(20000, max(1, int(_one(query, "host_limit", "10000") or 10000)))
        needs_host_scan = bool(host_text or inside != "any" or filtered != "all")
        matches = []
        for row in self.rows:
            if source and row.get("source") != source:
                continue
            row_text = normalize_host_word(f"{row.get('canonical', '')} {row.get('search_word', '')}")
            if text and text not in row_text:
                continue
            hosts = None
            if needs_host_scan:
                hosts = self._visible_hosts(row, host_text, inside, filtered)
                if not hosts:
                    continue
            matches.append((row, hosts))
        page = self._page_items(matches, offset, limit, host_limit)
        return {
            "total": len(matches),
            "offset": offset,
            "limit": limit,
            "drug_total": len(matches),
            "items": page,
        }

    def _page_items(
        self,
        matches: list[tuple[dict[str, Any], list[dict[str, Any]] | None]],
        offset: int,
        limit: int,
        host_limit: int,
    ) -> list[dict[str, Any]]:
        end = offset + limit
        items: list[dict[str, Any]] = []
        for row, hosts in matches[offset:end]:
            visible_hosts = hosts if hosts is not None else self._visible_hosts(row, "", "any", "all")
            items.append(row_payload(row, visible_hosts, host_limit))
        return items

    def _collect_host_words_for_query(self, query: dict[str, Any]) -> list[str]:
        text = normalize_host_word(str(query.get("q", "")))
        host_text = normalize_host_word(str(query.get("host", "")))
        source = str(query.get("source", ""))
        inside = str(query.get("inside", "any"))
        filtered = str(query.get("filtered", "active") or "active")
        output: set[str] = set()
        for row in self.rows:
            if source and row.get("source") != source:
                continue
            row_text = normalize_host_word(f"{row.get('canonical', '')} {row.get('search_word', '')}")
            if text and text not in row_text:
                continue
            for host in self._visible_hosts(row, host_text, inside, filtered):
                word = normalize_host_word(host.get("host_word", ""))
                if word:
                    output.add(word)
        return sorted(output)

    def _collect_host_words_for_row(self, row_key: dict[str, Any], query: dict[str, Any]) -> list[str]:
        source = str(row_key.get("source", ""))
        term_id = str(row_key.get("term_id", ""))
        search_word = str(row_key.get("search_word", ""))
        host_text = normalize_host_word(str(query.get("host", "")))
        inside = str(query.get("inside", "any"))
        filtered = str(query.get("filtered", "active") or "active")
        output: set[str] = set()
        for row in self.rows:
            if source and row.get("source") != source:
                continue
            if term_id and row.get("term_id") != term_id:
                continue
            if search_word and row.get("search_word") != search_word:
                continue
            for host in self._visible_hosts(row, host_text, inside, filtered):
                word = normalize_host_word(host.get("host_word", ""))
                if word:
                    output.add(word)
        return sorted(output)

    def _visible_hosts(self, row: dict[str, Any], host_text: str, inside: str, filtered: str) -> list[dict[str, Any]]:
        hosts = []
        for host in row.get("host_words", []):
            normalized = normalize_host_word(host.get("host_word", ""))
            is_filtered = normalized in self.filters
            if host_text and host_text not in normalized:
                continue
            if inside == "inside" and not host.get("inside_word"):
                continue
            if inside == "whole" and host.get("inside_word"):
                continue
            if filtered == "filtered" and not is_filtered:
                continue
            if filtered == "active" and is_filtered:
                continue
            item = dict(host)
            item["filtered"] = is_filtered
            hosts.append(item)
        return hosts

    def _build_host_word_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            for host in row.get("host_words", []):
                key = normalize_host_word(host.get("host_word", ""))
                counts[key] = counts.get(key, 0) + int(host.get("count", 0) or 0)
        return counts

    def estimated_removed_match_count(self) -> int:
        return sum(self.host_word_counts.get(word, 0) for word in self.filters)

    def _filter_dirty(self) -> bool:
        if not self.paths["filtered"].exists():
            return True
        newest_filter_mtime = self.paths["filters"].stat().st_mtime if self.paths["filters"].exists() else 0.0
        if self.paths["filters_parts"].exists():
            mtimes = [item.stat().st_mtime for item in self.paths["filters_parts"].glob("*.json")]
            newest_filter_mtime = max([newest_filter_mtime, *mtimes]) if mtimes else newest_filter_mtime
        return newest_filter_mtime > self.paths["filtered"].stat().st_mtime

    def _filter_shard_count(self) -> int:
        parts = self.paths["filters_parts"]
        if not parts.exists():
            return 0
        return len(list(parts.glob("host_words_*.json")))


class Handler(BaseHTTPRequestHandler):
    state: DashboardState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_text(DASHBOARD_HTML, "text/html; charset=utf-8")
        elif parsed.path == "/api/state":
            self._send_json(self.state.state_payload())
        elif parsed.path == "/api/search":
            self._send_json(self.state.search(parse_qs(parsed.query)))
        elif parsed.path == "/api/filters":
            self._send_json(self.state.filters_page(parse_qs(parsed.query)))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if parsed.path == "/api/filter":
                self._send_json(self.state.update_filter(str(body.get("host_word", "")), bool(body.get("enabled", True))))
            elif parsed.path == "/api/filter-bulk":
                self._send_json(self.state.update_filter_bulk(list(body.get("host_words", [])), bool(body.get("enabled", True))))
            elif parsed.path == "/api/filter-search":
                self._send_json(self.state.add_search_results_to_filter(dict(body.get("query", {}))))
            elif parsed.path == "/api/filter-row":
                self._send_json(self.state.add_row_host_words_to_filter(dict(body.get("row", {})), dict(body.get("query", {}))))
            elif parsed.path == "/api/rebuild-filtered":
                self._send_json(self.state.rebuild_filtered())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("api_error path=%s", parsed.path)
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("http_client=%s message=%s", self.client_address[0], format % args)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, payload: str, content_type: str) -> None:
        data = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _one(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default


def run_server(base_dir: str | Path, host: str = "127.0.0.1", port: int = 8787, *, rebuild_compact: bool = False) -> None:
    state = DashboardState(base_dir)
    state.ensure(rebuild_compact=rebuild_compact)
    Handler.state = state
    server = ThreadingHTTPServer((host, port), Handler)
    LOGGER.info("dashboard_server_start url=http://%s:%s base=%s", host, port, state.paths["base"])
    server.serve_forever()


def build_outputs(base_dir: str | Path, *, rebuild_compact: bool = False) -> dict[str, Any]:
    state = DashboardState(base_dir)
    state.ensure(rebuild_compact=rebuild_compact)
    return state.state_payload()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aho-filter-dashboard")
    parser.add_argument("command", nargs="?", default="build", choices=["build", "serve"])
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[2] / "reports" / "aho"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--rebuild-compact", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()
    if args.command == "serve":
        run_server(args.base_dir, args.host, args.port, rebuild_compact=args.rebuild_compact)
    else:
        payload = build_outputs(args.base_dir, rebuild_compact=args.rebuild_compact)
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
