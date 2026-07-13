"""Citation verification helpers for the research pipeline.

Integrates verification logic from the original literature/verify.py:
  - arXiv ID + DOI resolution via Crossref (with DataCite fallback)
  - Jaccard word-overlap title similarity
  - Structured verification statuses and reports
  - File-based result caching
  - BibTeX parsing and post-processing helpers
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Title similarity  —  Jaccard word-overlap
# ---------------------------------------------------------------------------


def title_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard-ish similarity between two titles (0.0–1.0)."""

    def _words(t: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9\s]", "", t.lower()).split()) - {""}

    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


# ---------------------------------------------------------------------------
# Verification statuses
# ---------------------------------------------------------------------------


class VerifyStatus(str, Enum):
    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    HALLUCINATED = "hallucinated"
    SKIPPED = "skipped"


@dataclass
class CitationResult:
    """Verification result for one citation."""

    title: str
    status: VerifyStatus
    confidence: float
    method: str
    details: str = ""
    doi: str = ""
    arxiv_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "details": self.details,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
        }


@dataclass
class VerificationReport:
    """Aggregate report for all citations in a batch."""

    total: int = 0
    verified: int = 0
    suspicious: int = 0
    hallucinated: int = 0
    skipped: int = 0
    results: list[CitationResult] = field(default_factory=list)

    @property
    def integrity_score(self) -> float:
        verifiable = self.total - self.skipped
        if verifiable <= 0:
            return 1.0
        return round(self.verified / verifiable, 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "total": self.total,
                "verified": self.verified,
                "suspicious": self.suspicious,
                "hallucinated": self.hallucinated,
                "skipped": self.skipped,
                "integrity_score": self.integrity_score,
            },
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# BibTeX parser
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"@(\w+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)\s*\}(?=\s*(?:@|\Z))",
    re.DOTALL,
)

_FIELD_RE = re.compile(
    r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
    re.DOTALL,
)


def parse_bibtex_entries(bib_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for m in _ENTRY_RE.finditer(bib_text):
        entry: dict[str, str] = {
            "type": m.group(1).lower(),
            "key": m.group(2).strip(),
        }
        for fm in _FIELD_RE.finditer(m.group(3)):
            entry[fm.group(1).lower()] = fm.group(2).strip()
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / "results" / ".cache" / "verify"


def _cache_key(title: str) -> str:
    return hashlib.sha256(title.lower().strip().encode()).hexdigest()[:16]


def _read_cache(title: str) -> CitationResult | None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{_cache_key(title)}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return CitationResult(
                title=data.get("title", title),
                status=VerifyStatus(data["status"]),
                confidence=data["confidence"],
                method=data["method"],
                details=data.get("details", ""),
                doi=data.get("doi", ""),
                arxiv_id=data.get("arxiv_id", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    return None


def _write_cache(title: str, result: CitationResult) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{_cache_key(title)}.json"
    cache_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CitationVerifier
# ---------------------------------------------------------------------------


class CitationVerifier:
    """Verifica citazioni via arXiv API + Crossref (con fallback DataCite)."""

    ARXIV_URL = "http://export.arxiv.org/api/query"
    CROSSREF_URL = "https://api.crossref.org/works"
    DATACITE_URL = "https://api.datacite.org/dois"
    VERIFY_TIMEOUT = 300

    def __init__(self, use_cache: bool = True) -> None:
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._start_time = time.monotonic()
        self.use_cache = use_cache

    async def close(self) -> None:
        await self._client.aclose()

    # ── arXiv title search ──────────────────────────────────────

    async def verify_arxiv(self, title: str) -> CitationResult:
        query = quote(title)
        url = f"{self.ARXIV_URL}?search_query=ti:{query}&max_results=3"
        await asyncio.sleep(4)
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return CitationResult(title=title, status=VerifyStatus.SKIPPED,
                                      confidence=0.0, method="arxiv",
                                      details=f"arXiv API error {resp.status_code}")
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            if not entries:
                return CitationResult(title=title, status=VerifyStatus.HALLUCINATED,
                                      confidence=0.9, method="arxiv",
                                      details="Nessun risultato su arXiv")
            best_sim, best_title = 0.0, ""
            for entry in entries:
                etitle_el = entry.find("atom:title", ns)
                if etitle_el is not None and etitle_el.text:
                    etitle = re.sub(r"\s+", " ", etitle_el.text.strip())
                    sim = title_similarity(title, etitle)
                    if sim > best_sim:
                        best_sim, best_title = sim, etitle
            if best_sim >= 0.80:
                return CitationResult(title=title, status=VerifyStatus.VERIFIED,
                                      confidence=best_sim, method="arxiv",
                                      details=f"arXiv match: '{best_title}'")
            elif best_sim >= 0.50:
                return CitationResult(title=title, status=VerifyStatus.SUSPICIOUS,
                                      confidence=best_sim, method="arxiv",
                                      details=f"arXiv title debole (sim={best_sim:.2f}): '{best_title}'")
            else:
                return CitationResult(title=title, status=VerifyStatus.HALLUCINATED,
                                      confidence=max(0.5, 1.0 - best_sim), method="arxiv",
                                      details=f"Titolo arXiv non corrisponde (sim={best_sim:.2f})")
        except Exception as e:
            return CitationResult(title=title, status=VerifyStatus.SKIPPED,
                                  confidence=0.0, method="arxiv", details=f"arXiv error: {e}")

    # ── arXiv ID lookup ─────────────────────────────────────────

    async def verify_arxiv_id(self, arxiv_id: str, expected_title: str = "") -> CitationResult:
        if not arxiv_id:
            return CitationResult(title=expected_title, status=VerifyStatus.SKIPPED,
                                  confidence=0.0, method="arxiv_id", details="arXiv ID mancante")
        url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return CitationResult(title=expected_title, status=VerifyStatus.SKIPPED,
                                      confidence=0.0, method="arxiv_id",
                                      details=f"arXiv API error {resp.status_code}")
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            if not entries:
                return CitationResult(title=expected_title, status=VerifyStatus.HALLUCINATED,
                                      confidence=0.9, method="arxiv_id",
                                      details=f"arXiv ID {arxiv_id} non trovato")
            found_title_el = entries[0].find("atom:title", ns)
            found_title = re.sub(r"\s+", " ", (found_title_el.text or "").strip()) if found_title_el is not None else ""
            if not expected_title:
                return CitationResult(title=found_title or expected_title,
                                      status=VerifyStatus.VERIFIED, confidence=0.95,
                                      method="arxiv_id", details=f"arXiv ID {arxiv_id} valido",
                                      arxiv_id=arxiv_id)
            sim = title_similarity(expected_title, found_title)
            if sim >= 0.80:
                return CitationResult(title=expected_title, status=VerifyStatus.VERIFIED,
                                      confidence=sim, method="arxiv_id",
                                      details=f"arXiv ID match (sim={sim:.2f}): '{found_title}'",
                                      arxiv_id=arxiv_id)
            elif sim >= 0.50:
                return CitationResult(title=expected_title, status=VerifyStatus.SUSPICIOUS,
                                      confidence=sim, method="arxiv_id",
                                      details=f"arXiv ID exists but title differs (sim={sim:.2f}): '{found_title}'",
                                      arxiv_id=arxiv_id)
            else:
                return CitationResult(title=expected_title, status=VerifyStatus.SUSPICIOUS,
                                      confidence=sim, method="arxiv_id",
                                      details=f"arXiv ID exists but title mismatch (sim={sim:.2f}): '{found_title}'",
                                      arxiv_id=arxiv_id)
        except Exception as e:
            return CitationResult(title=expected_title, status=VerifyStatus.SKIPPED,
                                  confidence=0.0, method="arxiv_id", details=f"arXiv ID error: {e}")

    # ── PDF download ────────────────────────────────────────────

    async def fetch_full_text(self, arxiv_id: str) -> str:
        if not arxiv_id:
            return ""
        try:
            resp = await self._client.get(f"https://arxiv.org/pdf/{arxiv_id}")
            if resp.status_code != 200:
                return ""
            import fitz
            import io
            doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n\n".join(pages)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            return text.replace("\x00", "").strip()
        except Exception:
            return ""

    # ── arXiv abstract fallback ─────────────────────────────────

    async def fetch_arxiv_abstract(self, arxiv_id: str) -> str:
        if not arxiv_id:
            return ""
        url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return ""
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entry = root.find("atom:entry", ns)
            if entry is None:
                return ""
            summary_el = entry.find("atom:summary", ns)
            if summary_el is not None and summary_el.text:
                return re.sub(r"\s+", " ", summary_el.text.strip())
            return ""
        except Exception:
            return ""

    # ── DOI resolution — Crossref + DataCite fallback ────────────

    async def verify_doi(self, doi: str, expected_title: str) -> CitationResult:
        clean_doi = doi.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            clean_doi = clean_doi.removeprefix(prefix)
        clean_doi = clean_doi.strip()
        if not clean_doi:
            return CitationResult(title=expected_title, status=VerifyStatus.SKIPPED,
                                  confidence=0.0, method="doi", details="DOI mancante")

        # Crossref
        url = f"{self.CROSSREF_URL}/{quote(clean_doi)}"
        try:
            resp = await self._client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                titles = data.get("message", {}).get("title", [])
                if titles:
                    crossref_title = re.sub(r"\s+", " ", titles[0].strip())
                    sim = title_similarity(expected_title, crossref_title)
                    if sim >= 0.80:
                        return CitationResult(title=expected_title, status=VerifyStatus.VERIFIED,
                                              confidence=sim, method="doi",
                                              details=f"DOI match via Crossref (sim={sim:.2f})", doi=clean_doi)
                    elif sim >= 0.50:
                        return CitationResult(title=expected_title, status=VerifyStatus.SUSPICIOUS,
                                              confidence=sim, method="doi",
                                              details=f"DOI resolves but title differs (sim={sim:.2f})", doi=clean_doi)
                    else:
                        return CitationResult(title=expected_title, status=VerifyStatus.SUSPICIOUS,
                                              confidence=sim, method="doi",
                                              details=f"DOI resolves but title mismatch (sim={sim:.2f})", doi=clean_doi)
                else:
                    return CitationResult(title=expected_title, status=VerifyStatus.VERIFIED,
                                          confidence=0.85, method="doi",
                                          details=f"DOI {clean_doi} resolves via Crossref", doi=clean_doi)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                return CitationResult(title=expected_title, status=VerifyStatus.SKIPPED,
                                      confidence=0.0, method="doi",
                                      details=f"Crossref HTTP {exc.response.status_code}")
        except Exception as e:
            return CitationResult(title=expected_title, status=VerifyStatus.SKIPPED,
                                  confidence=0.0, method="doi", details=f"Crossref error: {e}")

        # DataCite fallback per arXiv / Zenodo DOIs
        if clean_doi.startswith("10.48550/") or clean_doi.startswith("10.5281/"):
            try:
                resp = await self._client.get(f"{self.DATACITE_URL}/{quote(clean_doi)}")
                if resp.status_code == 200:
                    attrs = resp.json().get("data", {}).get("attributes", {})
                    dc_titles = attrs.get("titles", [])
                    if dc_titles:
                        found = dc_titles[0].get("title", "")
                        if found:
                            sim = title_similarity(expected_title, found)
                            if sim >= 0.80:
                                return CitationResult(title=expected_title, status=VerifyStatus.VERIFIED,
                                                      confidence=sim, method="doi",
                                                      details=f"DOI match via DataCite (sim={sim:.2f})", doi=clean_doi)
                    else:
                        return CitationResult(title=expected_title, status=VerifyStatus.VERIFIED,
                                              confidence=0.85, method="doi",
                                              details=f"DOI {clean_doi} resolves via DataCite", doi=clean_doi)
            except Exception:
                pass

        return CitationResult(title=expected_title, status=VerifyStatus.HALLUCINATED,
                              confidence=0.9, method="doi",
                              details=f"DOI {clean_doi} not found", doi=clean_doi)

    # ── High-level article verification (tuple interface) ───────

    async def verify_article(self, title: str, doi: str, arxiv_id: str = "") -> tuple[bool, str]:
        has_doi = doi.lower().strip() not in ("", "doi not confirmed", "not available", "none", "n/a")
        if self.use_cache:
            cached = _read_cache(title)
            if cached is not None:
                ok = cached.status == VerifyStatus.VERIFIED
                return ok, f"{'OK' if ok else 'FAIL'}: [cache] {cached.details}"
        if not has_doi:
            return False, "DOI non disponibile"
        result = await self.verify_doi(doi, title)
        if self.use_cache:
            _write_cache(title, result)
        ok = result.status == VerifyStatus.VERIFIED
        return ok, f"{'OK' if ok else 'FAIL'}: {result.method} ({result.details})"

    # ── Batch verification ──────────────────────────────────────

    async def verify_articles_batch(
        self, articles: list[tuple[str, str, str]],
    ) -> VerificationReport:
        report = VerificationReport(total=len(articles))
        _status_map = {
            VerifyStatus.VERIFIED: "verified",
            VerifyStatus.SUSPICIOUS: "suspicious",
            VerifyStatus.HALLUCINATED: "hallucinated",
            VerifyStatus.SKIPPED: "skipped",
        }
        for title, doi, arxiv_id in articles:
            if time.monotonic() - self._start_time > self.VERIFY_TIMEOUT:
                report.results.append(CitationResult(
                    title=title, status=VerifyStatus.SKIPPED,
                    confidence=0.0, method="skipped",
                    details="Global verification timeout"))
                report.skipped += 1
                continue
            if not title:
                report.results.append(CitationResult(
                    title="", status=VerifyStatus.SKIPPED,
                    confidence=0.0, method="skipped", details="No title"))
                report.skipped += 1
                continue
            if self.use_cache:
                cached = _read_cache(title)
                if cached is not None:
                    cached.title = title
                    report.results.append(cached)
                    setattr(report, _status_map.get(cached.status, "skipped"),
                            getattr(report, _status_map.get(cached.status, "skipped"), 0) + 1)
                    continue
            result: CitationResult | None = None
            if result is None and arxiv_id:
                result = await self.verify_arxiv_id(arxiv_id, title)
            has_doi = doi.lower().strip() not in ("", "doi not confirmed", "not available", "none", "n/a")
            if result is None and has_doi:
                result = await self.verify_doi(doi, title)
            if result is None:
                result = await self.verify_arxiv(title)
            if result is None:
                result = CitationResult(title=title, status=VerifyStatus.SKIPPED,
                                        confidence=0.0, method="skipped",
                                        details="All methods failed")
            if self.use_cache and result.status != VerifyStatus.SKIPPED:
                _write_cache(title, result)
            report.results.append(result)
            setattr(report, _status_map.get(result.status, "skipped"),
                    getattr(report, _status_map.get(result.status, "skipped"), 0) + 1)
        return report


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def filter_verified_bibtex(
    bib_text: str, report: VerificationReport, *, include_suspicious: bool = True,
) -> str:
    keep_keys: set[str] = set()
    for r in report.results:
        if r.status == VerifyStatus.VERIFIED:
            keep_keys.add(r.title)
        elif r.status == VerifyStatus.SUSPICIOUS and include_suspicious:
            keep_keys.add(r.title)
        elif r.status == VerifyStatus.SKIPPED:
            keep_keys.add(r.title)
    kept: list[str] = []
    for m in _ENTRY_RE.finditer(bib_text):
        title = ""
        for fm in _FIELD_RE.finditer(m.group(3)):
            if fm.group(1).lower() == "title":
                title = fm.group(2).strip()
                break
        if title and title in keep_keys:
            kept.append(m.group(0))
        elif not title:
            key = m.group(2).strip()
            if key in keep_keys:
                kept.append(m.group(0))
    return "\n\n".join(kept) + "\n" if kept else ""


def annotate_paper_hallucinations(paper_text: str, report: VerificationReport) -> str:
    hallucinated_titles: set[str] = set()
    for r in report.results:
        if r.status == VerifyStatus.HALLUCINATED:
            hallucinated_titles.add(r.title)
    if not hallucinated_titles:
        return paper_text
    result = paper_text

    def _replace_latex(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        kept = [k for k in keys if k not in hallucinated_titles]
        return "\\cite{" + ", ".join(kept) + "}" if kept else ""

    result = re.sub(r"\\cite\{([^}]+)\}", _replace_latex, result)
    _CITE_KEY_PAT = r"[a-zA-Z]+\d{4}[a-zA-Z]*"

    def _replace_markdown(m: re.Match[str]) -> str:
        keys = [k.strip() for k in re.split(r"[,;]\s*", m.group(1))]
        kept = [k for k in keys if k not in hallucinated_titles]
        return "[" + ", ".join(kept) + "]" if kept else ""

    result = re.sub(rf"\[({_CITE_KEY_PAT}(?:\s*[,;]\s*{_CITE_KEY_PAT})*)\]", _replace_markdown, result)
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r"\(\s*\)", "", result)
    result = re.sub(r"\[\s*\]", "", result)
    return result
