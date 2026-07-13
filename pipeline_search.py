"""Unified literature search with deduplication.

Combines results from OpenAlex, Semantic Scholar, and arXiv,
deduplicates by DOI → arXiv ID → fuzzy title match, and returns
a merged list sorted by citation count (descending).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import httpx

from pipeline_cache import get_cached, put_cache
from pipeline_domain import Author, Paper

logger = logging.getLogger(__name__)

_DEFAULT_SOURCES = ("openalex", "semantic_scholar", "arxiv")


# ---------------------------------------------------------------------------
# OpenAlex client
# ---------------------------------------------------------------------------

_OPENALEX_URL = "https://api.openalex.org/works"
_OPENALEX_EMAIL = "research@minne.local"


async def _search_openalex(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
) -> list[Paper]:
    params: dict[str, str] = {
        "search": query,
        "per_page": str(min(limit, 50)),
        "mailto": _OPENALEX_EMAIL,
        "select": (
            "id,title,authorships,publication_year,primary_location,"
            "cited_by_count,doi,ids,abstract_inverted_index,type"
        ),
    }
    if year_min > 0:
        params["filter"] = f"from_publication_date:{year_min}-01-01"

    url = f"{_OPENALEX_URL}?{_encode_params(params)}"
    try:
        resp = await client.get(url, timeout=20.0)
        if resp.status_code != 200:
            logger.warning("OpenAlex HTTP %d for %r", resp.status_code, query)
            return []
        data = resp.json()
    except Exception as exc:
        logger.warning("OpenAlex error for %r: %s", query, exc)
        return []

    results = data.get("results", [])
    if not isinstance(results, list):
        return []

    papers: list[Paper] = []
    for item in results:
        try:
            papers.append(_parse_openalex_work(item))
        except Exception:
            logger.debug("Failed to parse OpenAlex work", exc_info=True)
    return papers


def _parse_openalex_work(item: dict[str, Any]) -> Paper:
    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
    authorships = item.get("authorships") or []
    authors = tuple(
        Author(
            name=str(a.get("author", {}).get("display_name", "Unknown")),
            affiliation=str(
                (a.get("institutions") or [{}])[0].get("display_name", "")
                if a.get("institutions") else ""
            ),
        )
        for a in authorships if isinstance(a, dict)
    )
    year = int(item.get("publication_year") or 0)
    abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
    primary_loc = item.get("primary_location") or {}
    source_info = primary_loc.get("source") or {}
    venue = str(source_info.get("display_name") or "").strip()
    if venue and re.match(r"^[a-z]{2,}\.[A-Z]{2}$", venue):
        venue = ""
    citation_count = int(item.get("cited_by_count") or 0)
    raw_doi = str(item.get("doi") or "").strip()
    doi = raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    ids = item.get("ids") or {}
    openalex_id = str(ids.get("openalex") or item.get("id") or "").strip()
    raw_arxiv = str(ids.get("arxiv") or "").strip()
    arxiv_id = ""
    if raw_arxiv:
        m = re.search(r"(\d{4}\.\d{4,5})", raw_arxiv)
        if m:
            arxiv_id = m.group(1)
    url = ""
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        url = f"https://doi.org/{doi}"
    elif openalex_id:
        url = openalex_id
    paper_id = f"oalex-{openalex_id.split('/')[-1]}" if openalex_id else f"oalex-{title[:20]}"
    return Paper(
        paper_id=paper_id, title=title, authors=authors, year=year,
        abstract=abstract, venue=venue, citation_count=citation_count,
        doi=doi, arxiv_id=arxiv_id, url=url, source="openalex",
    )


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


# ---------------------------------------------------------------------------
# Semantic Scholar client
# ---------------------------------------------------------------------------

_S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "paperId,title,abstract,year,venue,citationCount,authors,externalIds,url"


async def _search_semantic_scholar(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
    api_key: str = "",
) -> list[Paper]:
    params: dict[str, str] = {
        "query": query,
        "limit": str(min(limit, 100)),
        "fields": _S2_FIELDS,
    }
    if year_min > 0:
        params["year"] = f"{year_min}-"

    url = f"{_S2_URL}?{_encode_params(params)}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = await client.get(url, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            logger.warning("S2 HTTP %d for %r", resp.status_code, query)
            return []
        data = resp.json()
    except Exception as exc:
        logger.warning("S2 error for %r: %s", query, exc)
        return []

    raw_papers = data.get("data", [])
    if not isinstance(raw_papers, list):
        return []

    papers: list[Paper] = []
    for item in raw_papers:
        try:
            papers.append(_parse_s2_paper(item))
        except Exception:
            logger.debug("Failed to parse S2 paper", exc_info=True)
    return papers


def _parse_s2_paper(item: dict[str, Any]) -> Paper:
    ext_ids = item.get("externalIds") or {}
    authors_raw = item.get("authors") or []
    authors = tuple(
        Author(name=a.get("name", "Unknown"))
        for a in authors_raw if isinstance(a, dict)
    )
    return Paper(
        paper_id=f"s2-{item.get('paperId', '')}",
        title=str(item.get("title", "")).strip(),
        authors=authors,
        year=int(item.get("year") or 0),
        abstract=str(item.get("abstract") or "").strip(),
        venue=str(item.get("venue") or "").strip(),
        citation_count=int(item.get("citationCount") or 0),
        doi=str(ext_ids.get("DOI") or "").strip(),
        arxiv_id=str(ext_ids.get("ArXiv") or "").strip(),
        url=str(item.get("url") or "").strip(),
        source="semantic_scholar",
    )


# ---------------------------------------------------------------------------
# arXiv client  (uses the ``arxiv`` pip package when available)
# ---------------------------------------------------------------------------

try:
    import arxiv as _arxiv_lib
except ImportError:
    _arxiv_lib = None


_ARXIV_API = "https://export.arxiv.org/api/query"


async def _search_arxiv_httpx(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
) -> list[Paper]:
    url = f"{_ARXIV_API}?search_query=all:{_encode_query(query)}&max_results={min(limit, 100)}&sortBy=relevance&sortOrder=descending"
    try:
        resp = await client.get(url, timeout=30.0)
        if resp.status_code != 200:
            logger.warning("arXiv HTTP %d for %r", resp.status_code, query)
            return []
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)
        papers: list[Paper] = []
        for entry in entries:
            _id_el = entry.find("atom:id", ns)
            _title_el = entry.find("atom:title", ns)
            _summary_el = entry.find("atom:summary", ns)
            _published_el = entry.find("atom:published", ns)
            _authors_el = entry.findall("atom:author", ns)
            title = re.sub(r"\s+", " ", (_title_el.text or "").strip()) if _title_el is not None else ""
            summary = re.sub(r"\s+", " ", (_summary_el.text or "").strip()) if _summary_el is not None else ""
            entry_id = _id_el.text or "" if _id_el is not None else ""
            m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", entry_id)
            arxiv_id = m.group(1) if m else ""
            year = 0
            if _published_el is not None and _published_el.text:
                try:
                    from datetime import datetime
                    year = datetime.fromisoformat(_published_el.text.rstrip("Z")).year
                except (ValueError, TypeError):
                    pass
            authors = tuple(
                Author(name=a.find("atom:name", ns).text or "Unknown")
                for a in _authors_el
                if a.find("atom:name", ns) is not None
            )
            if year_min > 0 and year < year_min:
                continue
            papers.append(Paper(
                paper_id=f"arxiv-{arxiv_id}" if arxiv_id else "",
                title=title, authors=authors, year=year,
                abstract=summary, venue="", citation_count=0,
                doi="", arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                source="arxiv",
            ))
        return papers
    except Exception as exc:
        logger.warning("arXiv error for %r: %s", query, exc)
        return []


async def _search_arxiv_lib(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
) -> list[Paper]:
    if _arxiv_lib is None:
        return []
    search = _arxiv_lib.Search(query=query, max_results=min(limit, 300),
                                sort_by=_arxiv_lib.SortCriterion.Relevance)
    papers: list[Paper] = []
    client = _arxiv_lib.Client(page_size=100, delay_seconds=3.1, num_retries=3)
    try:
        for result in client.results(search):
            arxiv_id = ""
            if result.entry_id:
                m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", result.entry_id)
                if m:
                    arxiv_id = m.group(1)
            year = result.published.year if result.published else 0
            if year_min > 0 and year < year_min:
                continue
            papers.append(Paper(
                paper_id=f"arxiv-{arxiv_id}",
                title=result.title or "",
                authors=tuple(Author(name=a.name) for a in result.authors),
                year=year,
                abstract=result.summary or "",
                venue=result.primary_category or "",
                citation_count=0,
                doi=result.doi or "",
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else result.entry_id,
                source="arxiv",
            ))
    except Exception as exc:
        logger.warning("arXiv library error for %r: %s", query, exc)
    return papers


def _encode_query(query: str) -> str:
    return __import__("urllib.parse").quote(query)


def _encode_params(params: dict[str, str]) -> str:
    return __import__("urllib.parse").urlencode(params)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _normalise_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_title: dict[str, int] = {}
    result: list[Paper] = []

    def _update_indices(p: Paper, idx: int) -> None:
        if p.doi:
            seen_doi[p.doi.lower().strip()] = idx
        if p.arxiv_id:
            seen_arxiv[p.arxiv_id.strip()] = idx
        norm = _normalise_title(p.title)
        if norm:
            seen_title[norm] = idx

    def _replace_at(old: Paper, new: Paper, idx: int) -> None:
        if old.doi:
            old_doi = old.doi.lower().strip()
            new_doi = new.doi.lower().strip() if new.doi else ""
            if old_doi != new_doi and seen_doi.get(old_doi) == idx:
                del seen_doi[old_doi]
        if old.arxiv_id:
            old_ax = old.arxiv_id.strip()
            new_ax = new.arxiv_id.strip() if new.arxiv_id else ""
            if old_ax != new_ax and seen_arxiv.get(old_ax) == idx:
                del seen_arxiv[old_ax]
        old_norm = _normalise_title(old.title)
        new_norm = _normalise_title(new.title)
        if old_norm and old_norm != new_norm and seen_title.get(old_norm) == idx:
            del seen_title[old_norm]
        result[idx] = new
        _update_indices(new, idx)

    for paper in papers:
        is_dup = False
        if paper.doi:
            doi_key = paper.doi.lower().strip()
            if doi_key in seen_doi:
                idx = seen_doi[doi_key]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True
        if not is_dup and paper.arxiv_id:
            ax_key = paper.arxiv_id.strip()
            if ax_key in seen_arxiv:
                idx = seen_arxiv[ax_key]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True
        if not is_dup:
            norm = _normalise_title(paper.title)
            if norm and norm in seen_title:
                idx = seen_title[norm]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True
        if is_dup:
            continue
        _update_indices(paper, len(result))
        result.append(paper)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def search_papers(
    query: str,
    *,
    limit: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    deduplicate: bool = True,
    s2_api_key: str = "",
) -> list[Paper]:
    """Search multiple academic sources and return deduplicated results."""
    all_papers: list[Paper] = []
    source_stats: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for src in sources:
            src_lower = src.lower().replace("-", "_").replace(" ", "_")
            cache_source = "semantic_scholar" if src_lower in ("semantic_scholar", "s2") else src_lower

            # Check cache first
            cached = get_cached(query, cache_source, limit)
            if cached:
                papers = _dicts_to_papers(cached)
                all_papers.extend(papers)
                source_stats[src_lower] = len(papers)
                logger.info("[cache] HIT %s: %d papers for %r", src_lower, len(papers), query)
                continue

            papers: list[Paper] = []

            if src_lower == "openalex":
                papers = await _search_openalex(client, query, limit=limit, year_min=year_min)
            elif src_lower in ("semantic_scholar", "s2"):
                papers = await _search_semantic_scholar(client, query, limit=limit, year_min=year_min, api_key=s2_api_key)
            elif src_lower == "arxiv":
                papers = await _search_arxiv_httpx(client, query, limit=limit, year_min=year_min)

            if papers:
                put_cache(query, cache_source, limit, _papers_to_dicts(papers))
                all_papers.extend(papers)
                source_stats[src_lower] = len(papers)
                logger.info("%s: %d papers for %r", src_lower, len(papers), query)

            await asyncio.sleep(1.0)

    logger.info("Found %d papers from %s for %r", len(all_papers), sources, query)

    if deduplicate:
        all_papers = _deduplicate(all_papers)

    all_papers.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    return all_papers


async def search_papers_multi_query(
    queries: list[str],
    *,
    limit_per_query: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    s2_api_key: str = "",
    inter_query_delay: float = 1.5,
) -> list[Paper]:
    """Run multiple queries and return deduplicated union."""
    all_papers: list[Paper] = []
    for i, q in enumerate(queries):
        if i > 0:
            await asyncio.sleep(inter_query_delay)
        results = await search_papers(
            q, limit=limit_per_query, sources=sources,
            year_min=year_min, s2_api_key=s2_api_key, deduplicate=False,
        )
        all_papers.extend(results)
        logger.info("Query %d/%d %r → %d papers", i + 1, len(queries), q, len(results))

    deduped = _deduplicate(all_papers)
    deduped.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    return deduped


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    entries = [p.to_bibtex() for p in papers]
    return "\n\n".join(entries) + "\n"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _papers_to_dicts(papers: list[Paper]) -> list[dict[str, object]]:
    return [asdict(p) for p in papers]


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _dicts_to_papers(dicts: list[dict[str, object]]) -> list[Paper]:
    papers: list[Paper] = []
    for d in dicts:
        try:
            authors_raw = d.get("authors", ())
            if not isinstance(authors_raw, list):
                authors_raw = []
            authors = tuple(
                Author(
                    name=str(a.get("name", "")),
                    affiliation=str(a.get("affiliation", "")),
                )
                for a in authors_raw if isinstance(a, dict)
            )
            papers.append(Paper(
                paper_id=str(d["paper_id"]),
                title=str(d["title"]),
                authors=authors,
                year=_as_int(d.get("year", 0), 0),
                abstract=str(d.get("abstract", "")),
                venue=str(d.get("venue", "")),
                citation_count=_as_int(d.get("citation_count", 0), 0),
                doi=str(d.get("doi", "")),
                arxiv_id=str(d.get("arxiv_id", "")),
                url=str(d.get("url", "")),
                source=str(d.get("source", "")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return papers
