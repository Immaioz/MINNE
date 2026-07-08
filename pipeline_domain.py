"""Domain models and parsing helpers for the research pipeline."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Literature domain models  (from literature/models.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Author:
    name: str
    affiliation: str = ""

    def last_name(self) -> str:
        parts = self.name.strip().split()
        raw = parts[-1] if parts else "unknown"
        nfkd = unicodedata.normalize("NFKD", raw)
        ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-zA-Z]", "", ascii_name).lower() or "unknown"


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[Author, ...] = ()
    year: int = 0
    abstract: str = ""
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    source: str = ""
    _bibtex_override: str = field(default="", repr=False)

    @property
    def cite_key(self) -> str:
        last = self.authors[0].last_name() if self.authors else "anon"
        yr = str(self.year) if self.year else "0000"
        kw = ""
        for word in self.title.split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
            if len(cleaned) > 3 and cleaned not in _STOPWORDS:
                kw = cleaned
                break
        return f"{last}{yr}{kw}"

    def to_bibtex(self) -> str:
        if self._bibtex_override:
            return self._bibtex_override.strip()
        key = self.cite_key
        authors_str = " and ".join(a.name for a in self.authors) or "Unknown"
        _venue = self.venue or ""
        _is_arxiv_category = bool(
            re.match(
                r"^(?:cs|math|stat|eess|physics|q-bio|q-fin|astro-ph|cond-mat|"
                r"gr-qc|hep-ex|hep-lat|hep-ph|hep-th|nlin|nucl-ex|nucl-th|"
                r"quant-ph)\.[A-Z]{2}$",
                _venue,
            )
        )
        if _venue and not _is_arxiv_category and any(
            kw in _venue.lower()
            for kw in ("conference", "proc", "workshop", "neurips", "icml", "iclr",
                       "aaai", "cvpr", "acl", "emnlp", "naacl", "eccv", "iccv",
                       "sigir", "kdd", "www", "ijcai")
        ):
            entry_type = "inproceedings"
            venue_field = f"  booktitle = {{{_venue}}},"
        elif self.arxiv_id and (not _venue or _is_arxiv_category):
            entry_type = "article"
            venue_field = f"  journal = {{arXiv preprint arXiv:{self.arxiv_id}}},"
        else:
            entry_type = "article"
            venue_field = f"  journal = {{{_venue or 'Unknown'}}}," if _venue else ""
        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{self.title}}},")
        lines.append(f"  author = {{{authors_str}}},")
        lines.append(f"  year = {{{self.year or 'Unknown'}}},")
        if venue_field:
            lines.append(venue_field)
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.arxiv_id:
            lines.append(f"  eprint = {{{self.arxiv_id}}},")
            lines.append("  archiveprefix = {arXiv},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": [{"name": a.name, "affiliation": a.affiliation} for a in self.authors],
            "year": self.year,
            "abstract": self.abstract,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "source": self.source,
            "cite_key": self.cite_key,
        }


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "upon", "about", "through", "using", "based", "towards", "toward",
    "between", "under", "more", "than", "when", "what", "which", "where",
    "does", "have", "been", "some", "each", "also", "much", "very",
    "learning",
})


# ---------------------------------------------------------------------------
# Pipeline domain models  (from pipeline_domain.py, extended)
# ---------------------------------------------------------------------------


@dataclass
class Article:
    title: str
    doi: str
    arxiv_id: str
    raw: str

    @property
    def has_doi(self) -> bool:
        d = self.doi.lower().strip()
        return d not in ("", "doi not confirmed", "not available", "none", "n/a")


def parse_articles(raw_text: str) -> list[Article]:
    articles: list[Article] = []
    blocks = re.split(r"### Article \d+", raw_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        t = re.search(r"- Title:\s*(.+)", block)
        d = re.search(r"- DOI:\s*(.+)", block)
        a = re.search(r"- arXiv:\s*(\S+)", block)
        title = t.group(1).strip() if t else ""
        doi = d.group(1).strip() if d else ""
        arxiv_id = a.group(1).strip() if a else ""
        if title:
            articles.append(Article(title=title, doi=doi, arxiv_id=arxiv_id, raw=block))
    return articles


@dataclass
class SubtopicResult:
    subtopic: str
    articles_raw: str
    summary_json: str


@dataclass
class PipelineResult:
    table: str
    draft: str
    novelties: str
    bibliography: str
    knowledge_base: str
    subtopic_results: list[SubtopicResult]


def build_articles_corpus(results: list[SubtopicResult]) -> str:
    parts = []
    num = 1
    for r in results:
        parts.append(f"--- Subtopic: {r.subtopic} ---")
        blocks = re.split(r"### Article \d+", r.articles_raw)
        for block in blocks:
            block = block.strip()
            if block:
                parts.append(f"[{num}]\n{block}")
                num += 1
    return "\n\n".join(parts)


def topic_slug(topic: str) -> str:
    slug = topic.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:60]
