"""Local query cache for literature search results.

Caches search results by (query, source, limit) hash to avoid
redundant API calls. Cache entries expire after source-specific TTL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent / "results" / ".cache" / "search"
_TTL_SEC = 86400 * 7

_SOURCE_TTL: dict[str, float] = {
    "arxiv": 86400,
    "semantic_scholar": 86400 * 3,
    "openalex": 86400 * 3,
    "citation_verify": 86400 * 365,
}


def _cache_dir(base: Path | None = None) -> Path:
    d = base or _CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(query: str, source: str, limit: int) -> str:
    raw = f"{query.strip().lower()}|{source.strip().lower()}|{limit}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(
    query: str,
    source: str,
    limit: int,
    *,
    cache_base: Path | None = None,
    ttl: float | None = None,
) -> list[dict[str, Any]] | None:
    d = _cache_dir(cache_base)
    key = cache_key(query, source, limit)
    path = d / f"{key}.json"
    if not path.exists():
        return None

    effective_ttl = ttl if ttl is not None else _SOURCE_TTL.get(source, _TTL_SEC)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = data.get("timestamp", 0)
        if time.time() - ts > effective_ttl:
            return None
        papers = data.get("papers", [])
        return papers if isinstance(papers, list) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def put_cache(
    query: str,
    source: str,
    limit: int,
    papers: list[dict[str, Any]],
    *,
    cache_base: Path | None = None,
) -> None:
    d = _cache_dir(cache_base)
    key = cache_key(query, source, limit)
    path = d / f"{key}.json"
    payload = {
        "query": query,
        "source": source,
        "limit": limit,
        "timestamp": time.time(),
        "papers": papers,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_cache(*, cache_base: Path | None = None) -> int:
    d = _cache_dir(cache_base)
    count = 0
    for f in d.glob("*.json"):
        f.unlink()
        count += 1
    return count


def cache_stats(*, cache_base: Path | None = None) -> dict[str, Any]:
    d = _cache_dir(cache_base)
    files = list(d.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "total_bytes": total_bytes,
        "cache_dir": str(d),
    }
