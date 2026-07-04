"""Domain models and parsing helpers for the research pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass


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
    """Estrae articoli dal formato `### Article N` usato da ResearchAgent."""
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
    """Costruisce un corpus numerato [1], [2], ... con tutti gli articoli."""
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
    """Converte un topic in una stringa usabile come nome di cartella."""
    slug = topic.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:60]
