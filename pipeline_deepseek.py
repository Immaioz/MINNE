"""
Multi-Agent Scientific Research Pipeline (DeepSeek direct API)
Calls DeepSeek Chat API directly (no opencode subprocess).

Flow:
  OrchestratorAgent
    ├── ResearchAgent(topic_1) ──► CitationVerifier(L1+L2) ──► SummaryAgent(topic_1)
    ├── ResearchAgent(topic_2) ──► CitationVerifier(L1+L2) ──► SummaryAgent(topic_2)
    └── ResearchAgent(topic_N) ──► CitationVerifier(L1+L2) ──► SummaryAgent(topic_N)
  AggregatorAgent  →  Markdown table
  RelatedWorksDraftAgent  →  narrative draft with [N] citations
  NoveltyProposalAgent  →  novelty table with difficulty ratings
  BibliographyAgent  →  IEEE bibliography

Citation verification:
  L1 — arXiv API: query by title, verify paper existence
  L2 — DOI resolution: resolve DOI via Crossref API, compare title
  If a paper fails both levels, ResearchAgent retries with a replacement.

Each ResearchAgent finds articles from DIFFERENT Q1 journals (no separate JournalAssignerAgent).
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

# Carica .env se presente
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

if not DEEPSEEK_API_KEY:
    print(
        "[Warning] DEEPSEEK_API_KEY non impostata. "
        "Aggiungila al file .env:\n"
        "  DEEPSEEK_API_KEY=sk-...\n"
        "Oppure esportala come variabile d'ambiente.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# LLM Client  —  chiamate API dirette a DeepSeek
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """Calls DeepSeek Chat API directly via httpx."""

    TIMEOUT = 300

    ERROR_PATTERNS = [
        (["insufficient", "credit"], "Your DeepSeek account has run out of credits. Top up at https://platform.deepseek.com"),
        (["insufficient", "quota"], "Your DeepSeek API quota is exhausted. Check your plan at https://platform.deepseek.com"),
        (["insufficient", "balance"], "Your DeepSeek balance is too low. Add funds at https://platform.deepseek.com"),
        (["payment", "required"], "Payment required. Your DeepSeek account may be inactive or out of credits."),
        (["rate limit", "429"], "Rate-limited by DeepSeek API. Retry after a few seconds."),
        (["too many request"], "Too many requests to DeepSeek API. Retry after a few seconds."),
        (["token limit"], "The prompt exceeds DeepSeek's maximum context length. Try reducing the number of articles per subtopic."),
        (["maximum context"], "The prompt exceeds DeepSeek's maximum context length. Try reducing the number of articles per subtopic."),
        (["context length"], "The combined prompt is too long for DeepSeek's context window. Reduce --articles or --subtopics."),
        (["unauthorized", "key"], "Your DeepSeek API key is invalid. Check DEEPSEEK_API_KEY in .env"),
        (["invalid", "api key"], "Your DeepSeek API key is invalid. Check DEEPSEEK_API_KEY in .env"),
        (["authentication"], "Authentication failed. Verify your DEEPSEEK_API_KEY in .env"),
        (["server error", "500"], "DeepSeek API returned a server error (500). This is usually temporary; retry."),
        (["server error", "503"], "DeepSeek API is unavailable (503). This is usually temporary; retry."),
        (["502"], "DeepSeek API returned a bad gateway (502). This is usually temporary; retry."),
        (["timeout", "504"], "DeepSeek API timed out (504). This is usually temporary; retry."),
    ]

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def ask(self, prompt: str) -> str:
        return await self._ask_async(prompt)

    @staticmethod
    def _classify_error(status_code: int, body: str) -> str:
        combined = body.lower()
        for keywords, hint in DeepSeekClient.ERROR_PATTERNS:
            if all(kw in combined for kw in keywords):
                return hint
        if status_code == 401:
            return "Authentication failed. Verify your DEEPSEEK_API_KEY in .env"
        if status_code == 429:
            return "Rate-limited by DeepSeek API. Retry after a few seconds."
        if status_code == 402:
            return "Payment required. Your DeepSeek account may be out of credits."
        if 500 <= status_code < 600:
            return "DeepSeek API server error. This is usually temporary; retry."
        return ""

    async def _ask_async(self, prompt: str) -> str:
        safe = prompt.replace("\x00", "")
        if not DEEPSEEK_API_KEY:
            return "[Error] DEEPSEEK_API_KEY not set"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": safe},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.post(
                    f"{DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code != 200:
                    body = resp.text[:500]
                    hint = self._classify_error(resp.status_code, body)
                    msg = f"[Error] DeepSeek API returned {resp.status_code}: {body}"
                    if hint:
                        msg += f"\n\n💡 {hint}"
                    return msg
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return "[Error] DeepSeek API returned empty choices"
                return choices[0].get("message", {}).get("content", "").strip()
        except httpx.TimeoutException:
            return f"[Error] DeepSeek API timeout ({self.TIMEOUT}s)"
        except Exception as e:
            return f"[Error] {e}"


# ---------------------------------------------------------------------------
# Agente base
# ---------------------------------------------------------------------------

class Agent:
    """Minimal agent: delegates to DeepSeek API directly."""
    name = ""
    system_prompt = ""

    async def a_run(self, prompt: str, **kwargs) -> str:
        client = DeepSeekClient(self.system_prompt)
        return await client.ask(prompt)


# ---------------------------------------------------------------------------
# Agente 1 — OrchestratorAgent
#   Scompone il topic in N sottotemi distinti
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are a research orchestrator. Given a broad research topic, decompose it into
distinct, complementary subtopics suitable for parallel bibliographic search.

Reply with ONLY a numbered list — one subtopic per line, no extra commentary.
Example:
1. Deep learning architectures for image segmentation
2. Transfer learning in medical imaging
3. Self-supervised vision transformers
"""

class OrchestratorAgent(Agent):
    name = "OrchestratorAgent"
    system_prompt = ORCHESTRATOR_PROMPT


# ---------------------------------------------------------------------------
# Agente 2 — ResearchAgent
#   Trova N articoli recenti su N journal Q1 diversi
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """\
You are an expert scientific literature researcher with deep knowledge of academic
journals and bibliometric rankings.

You have access to web search and web fetch tools. USE THEM.

Task: given a research subtopic, use web search to find exactly
{num_articles} recent peer-reviewed articles (published in the last 3 years)
published in HIGH-QUALITY Q1 journals (according to Scimago/JCR).

Search strategy:
1. Use `websearch` to search for the subtopic on Google Scholar or Semantic Scholar.
2. For each candidate article, use `webfetch` to verify the DOI resolves correctly
   and that the arXiv ID exists.
3. Confirm each journal is indeed Q1 before including it.

Rules:
- Articles MUST be from DIFFERENT Q1 journals (no two from the same journal).
- All journals MUST be Q1 — confirm the quartile in the output.
- ALL articles MUST have a valid DOI.
- Provide REAL articles only — do NOT invent or fabricate.
- If web search returns no valid results, report fewer articles rather than inventing.
- Reply ONLY with the structured list below — no preamble, no commentary.

Format (repeat exactly for each article):

### Article 1
- Title: <full title>
- Authors: <last name, initials; ...>
- Year: <YYYY>
- Journal: <journal name> (Q1, IF ≈ <impact factor>)
- DOI: <doi>
- arXiv: <arXiv ID, e.g. 2301.12345>
- Abstract: <brief summary of key contribution>

### Article 2
...

### Article N
...
"""

class ResearchAgent(Agent):
    name = "ResearchAgent"
    system_prompt = RESEARCH_PROMPT

    def __init__(self, num_articles: int = 3):
        self.num_articles = num_articles

    async def a_run(self, prompt: str, **kwargs) -> str:
        system = self.system_prompt.format(num_articles=self.num_articles)
        full_prompt = system + "\n\n" + prompt
        full_prompt = full_prompt.replace("\x00", "")
        client = DeepSeekClient(system)
        return await client.ask(full_prompt)


# ---------------------------------------------------------------------------
# Agente 3 — SummaryAgent
#   Struttura i risultati del ResearchAgent in un JSON array (no tool)
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """\
You are a concise academic analyst.

You receive a structured list of scientific articles and must produce a JSON array
with one object per article. No tools, no web search — only analyse the text you receive.

Each JSON object MUST have exactly these six keys:
  "title"       : string — full article title
  "results"     : string — 1–2 sentences describing the KEY FINDINGS (max 30 words)
  "journal"     : string — journal name and quartile, e.g. "Nature Machine Intelligence (Q1)"
  "dataset"     : string — dataset(s) used, "none" if none, and whether it is public or private;
                           if public include the download link/URL
  "methodology" : string — brief description of the proposed method/approach (max 20 words)
  "code"        : string — "not available" if no public code; otherwise the URL to the
                           official code repository (GitHub, GitLab, etc.)

Reply with ONLY the raw JSON array — no markdown fences, no explanatory text.
"""

class SummaryAgent(Agent):
    name = "SummaryAgent"
    system_prompt = SUMMARY_PROMPT


# ---------------------------------------------------------------------------
# Agente 4 — AggregatorAgent
#   Consolida tutti i JSON in una tabella Markdown finale
# ---------------------------------------------------------------------------

AGGREGATOR_PROMPT = """\
You are a scientific report writer.

You receive multiple JSON arrays of article summaries, each prefixed with its subtopic.
Produce a single clean Markdown table with these columns:

| # | Title | Key Results | Journal | Methodology | Dataset | Code | Subtopic |
|---|-------|-------------|---------|-------------|---------|------|----------|

Rules:
- Include ALL articles from all subtopics.
- Keep "Key Results" under 25 words per cell.
- "Dataset" must state which dataset(s) were used and whether they are public or private;
  if a public download link exists, include it as a Markdown link.
- "Code" must contain a Markdown link to the official repository, or "Not available" if none.
- Sort rows by Subtopic, then alphabetically by Title.
- Reply with ONLY the Markdown table — no title, no preamble, no trailing text.
"""

class AggregatorAgent(Agent):
    name = "AggregatorAgent"
    system_prompt = AGGREGATOR_PROMPT


# ---------------------------------------------------------------------------
# Agente 5 — RelatedWorksDraftAgent
#   Produce una bozza narrativa dello stato dell'arte con citazioni [N],
#   pronta per una sezione introduttiva o related works.
# ---------------------------------------------------------------------------

RELATED_WORKS_PROMPT = """\
You are an expert scientific writer preparing a "Related Work" or introductory
section for a journal paper.

You receive:
1. A numbered corpus of articles, each with title, authors, year, journal, DOI,
   and abstract (already assigned citation numbers [1], [2], …).
2. A Markdown summary table of all articles.

Write a flowing academic narrative organised by subtopic. Use the pre-assigned
citation numbers to reference articles — e.g. "[1]" or "[1], [2]".

Rules:
- Use formal academic language.
- For each cited work, mention author and year in the text:
  e.g. "Singh et al. [1] proposed …" or "Recent work by Kim et al. [2] …"
- Group articles by subtopic and explain how they relate / differ.
- End with a brief paragraph identifying open challenges or research gaps.
- Do NOT fabricate claims beyond what the provided abstracts describe.
- Reply with ONLY the narrative text — no title, no preamble, no commentary.
"""

class RelatedWorksDraftAgent(Agent):
    name = "RelatedWorksDraftAgent"
    system_prompt = RELATED_WORKS_PROMPT


# ---------------------------------------------------------------------------
# Agente 6 — BibliographyAgent
#   Formatta la bibliografia in stile IEEE a partire dalla bozza e dal corpus.
# ---------------------------------------------------------------------------

BIBLIOGRAPHY_PROMPT = """\
You are a reference formatter. You receive:
1. A narrative draft with numbered citations [1], [2], …
2. The same numbered article corpus (title, authors, year, journal, DOI, abstract).

Produce a reference list in IEEE style:
[1] A. Author, B. Author, and C. Author, "Title," Journal Name, vol. X, no. Y, pp. Z–Z, Year, doi: XXX.

Rules:
- Include ONLY articles that are actually cited in the draft.
- Preserve the citation numbers from the draft.
- Use the real DOI where available; otherwise write "doi: not available".
- All entries must be real — do not invent details.
- Reply with ONLY the reference list — no preamble, no commentary.
"""

class BibliographyAgent(Agent):
    name = "BibliographyAgent"
    system_prompt = BIBLIOGRAPHY_PROMPT


# ---------------------------------------------------------------------------
# Agente 7 — NoveltyProposalAgent
#   Valuta lo stato dell'arte e propone possibili novità realizzabili,
#   con tabella riassuntiva e livello di difficoltà.
# ---------------------------------------------------------------------------

NOVELTY_PROMPT = """\
You are a research advisor evaluating a state-of-the-art survey to identify
feasible research novelties.

You receive:
1. A narrative draft of the state of the art (Related Work section), with
   numbered citations [1], [2], ….
2. The full numbered article corpus (title, authors, year, journal, abstract).

First, produce a summary table of the identified novelties:

| # | Novelty | Description | Difficulty | Rationale |
|---|---------|-------------|------------|-----------|

Rules for the table:
- "Difficulty" must be one of: ★ Easy, ★★ Medium, ★★★ Hard.
- "Description" must clearly state WHAT would be done and WHY it is novel
  (2–3 sentences).
- "Rationale" must cite the relevant articles from the draft [N] and explain
  why this fills a gap.
- All novelties MUST be realistically implementable (no purely theoretical
  or data-unavailable proposals).

Then, after the table, for EACH novelty provide a detailed discussion section
with the following structure:

### Novelty N: <title>

**Methodology.** Outline the proposed approach step by step. Describe the
architectural design, training procedure, and any key algorithmic innovations.

**Dataset.** Specify which dataset(s) should be used, whether they are public
or private, and include download links if public. If new data needs to be
collected, describe the collection strategy.

**Baselines & Comparisons.** List the state-of-the-art methods that should be
used as baselines. Specify evaluation protocols (e.g. cross-validation, held-out
test set) and the main metrics for comparison.

**Evaluation Metrics.** List the quantitative and qualitative metrics to assess
performance (e.g., accuracy, F1, throughput, latency, human evaluation, etc.).

**Implementation Roadmap.** Summarise the key steps needed to implement this
novelty, including any expected challenges and possible mitigation strategies.

Do NOT invent article details beyond what the corpus provides. Use formal
academic language throughout.
"""

class NoveltyProposalAgent(Agent):
    name = "NoveltyProposalAgent"
    system_prompt = NOVELTY_PROMPT


# ---------------------------------------------------------------------------
# Citation Verification Layer
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


class CitationVerifier:
    """L1 (arXiv API) + L2 (DOI resolution via Crossref) verification."""

    ARXIV_URL = "http://export.arxiv.org/api/query"
    CROSSREF_URL = "https://api.crossref.org/works"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    # ── L1: arXiv API ────────────────────────────────────────────
    async def verify_arxiv(self, title: str) -> tuple[bool, str]:
        """Cerca il titolo su arXiv. Restituisce (passato, messaggio)."""
        query = quote(title)
        url = f"{self.ARXIV_URL}?search_query=ti:{query}&max_results=3"
        await asyncio.sleep(4)  # rate-limit: 4s between arXiv queries
        # await asyncio.sleep(4)  # ← abilita per rispettare rate limit arXiv (1 req/4s)
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return False, f"arXiv API error {resp.status_code}"

            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            if not entries:
                return False, "Nessun risultato su arXiv"

            import difflib
            for entry in entries:
                etitle_el = entry.find("atom:title", ns)
                if etitle_el is not None and etitle_el.text:
                    etitle = etitle_el.text.strip().lower()
                    score = difflib.SequenceMatcher(
                        None, title.lower(), etitle,
                    ).ratio()
                    if score > 0.6:
                        return True, f"arXiv match (score={score:.2f})"
            return False, "Titolo non corrisponde su arXiv (arXiv titles differ)"
        except Exception as e:
            return False, f"arXiv error: {e}"

    # ── L1b: arXiv ID lookup ─────────────────────────────────────
    async def verify_arxiv_id(self, arxiv_id: str) -> tuple[bool, str]:
        """Verifica che un arXiv ID esista."""
        if not arxiv_id:
            return False, "arXiv ID mancante"
        url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
        # await asyncio.sleep(4)  # ← abilita per rispettare rate limit arXiv (1 req/4s)
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return False, f"arXiv API error {resp.status_code}"
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            if entries:
                return True, f"arXiv ID {arxiv_id} valido"
            return False, f"arXiv ID {arxiv_id} non trovato"
        except Exception as e:
            return False, f"arXiv ID error: {e}"

    # ── Fetch full paper text from arXiv PDF ──────────────────────
    async def fetch_full_text(self, arxiv_id: str) -> str:
        """Scarica il PDF da arXiv ed estrae il testo completo con PyMuPDF."""
        if not arxiv_id:
            return ""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            resp = await self._client.get(pdf_url)
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
            text = text.replace("\x00", "")  # rimuove null byte dai PDF
            return text.strip()
        except Exception:
            return ""

    # ── Fallback: abstract da arXiv API ──────────────────────────
    async def fetch_arxiv_abstract(self, arxiv_id: str) -> str:
        """Recupera l'abstract da arXiv API (fallback se PDF non disponibile)."""
        if not arxiv_id:
            return ""
        url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
        # await asyncio.sleep(4)  # ← abilita per rispettare rate limit arXiv (1 req/4s)
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
                text = re.sub(r"\s+", " ", summary_el.text.strip())
                return text
            return ""
        except Exception:
            return ""

    # ── L2: DOI resolution via Crossref ───────────────────────────
    async def verify_doi(self, doi: str, expected_title: str) -> tuple[bool, str]:
        """Risolve DOI via Crossref API e verifica il titolo."""
        clean_doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:").strip()
        if not clean_doi:
            return False, "DOI mancante"

        url = f"{self.CROSSREF_URL}/{quote(clean_doi)}"
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return False, f"Crossref error {resp.status_code}"

            data = resp.json()
            msg = data.get("message", {})
            titles = msg.get("title", [])
            if not titles:
                return False, "Nessun titolo in Crossref"

            import difflib
            crossref_title = titles[0].strip().lower()
            score = difflib.SequenceMatcher(
                None, expected_title.lower(), crossref_title,
            ).ratio()
            if score > 0.6:
                return True, f"DOI match (score={score:.2f})"
            return False, f"Titolo Crossref non corrisponde (score={score:.2f})"
        except Exception as e:
            return False, f"DOI error: {e}"

    # ── Two-level verification ────────────────────────────────────
    async def verify_article(self, title: str, doi: str, arxiv_id: str = "") -> tuple[bool, str]:
        """
        Verifica solo tramite DOI (Crossref). arXiv check disabilitato temporaneamente.
        """
        has_doi = doi.lower().strip() not in ("", "doi not confirmed", "not available", "none", "n/a")
        if not has_doi:
            return False, "DOI non disponibile"
        l2_ok, l2_msg = await self.verify_doi(doi, title)
        if l2_ok:
            return True, f"L2 OK ({l2_msg})"
        return False, f"L2: {l2_msg}"


# ---------------------------------------------------------------------------
# Pipeline per singolo sottotema: ResearchAgent → CitationVerifier → SummaryAgent
# ---------------------------------------------------------------------------

@dataclass
class SubtopicResult:
    subtopic: str
    articles_raw: str   # output grezzo del ResearchAgent
    summary_json: str   # JSON strutturato del SummaryAgent


@dataclass
class PipelineResult:
    table: str          # tabella Markdown da AggregatorAgent
    draft: str          # bozza narrativa da RelatedWorksDraftAgent
    novelties: str      # tabella novità da NoveltyProposalAgent
    bibliography: str   # bibliografia formattata da BibliographyAgent
    knowledge_base: str # knowledge base completo per chatbot
    subtopic_results: list[SubtopicResult]  # risultati grezzi per sottotema


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


async def run_subtopic_pipeline(
    subtopic: str, results_dir: Path, num_articles: int = 3,
) -> SubtopicResult:
    """
    Esegue ResearchAgent → salva articoli proposti (.md) → CitationVerifier (DOI) → SummaryAgent.
    L'intera catena viene lanciata in parallelo con le altre tramite asyncio.gather.
    """
    print(f"  🔍  [{subtopic[:50]}…] searching multiple Q1 journals")

    proposed_dir = results_dir / "proposed"
    proposed_dir.mkdir(parents=True, exist_ok=True)

    verifier = CitationVerifier()
    try:
        verified_blocks: list[str] = []
        used_titles: set[str] = set()
        prompt_prefix = f"Research subtopic: {subtopic}\n"

        for attempt in range(1):
            still_needed = num_articles - len(verified_blocks)
            if still_needed <= 0:
                break

            if attempt == 0:
                prompt = (
                    f"{prompt_prefix}"
                    f"Find {num_articles} articles from DIFFERENT Q1 journals."
                )
            else:
                exclude = "; ".join(sorted(used_titles)[:10])
                prompt = (
                    f"{prompt_prefix}"
                    f"Find {still_needed} DIFFERENT articles from DIFFERENT Q1 journals.\n"
                    f"Do NOT repeat any of these already-found articles:\n{exclude}"
                )

            researcher = ResearchAgent(num_articles=still_needed)
            raw = await researcher.a_run(prompt)
            articles = parse_articles(raw)

            if not articles:
                print(f"  ⚠️   [{subtopic[:50]}…] Researcher produced no valid articles (attempt {attempt+1})")
                continue

            # Save proposed articles to .md (first attempt only)
            if attempt == 0:
                subtopic_slug = re.sub(r"[^a-z0-9]+", "-", subtopic.lower()).strip("-")[:40]
                proposed_path = proposed_dir / f"{subtopic_slug}.md"
                header = f"# Proposed articles: {subtopic}\n\n"
                proposed_path.write_text(header + raw, encoding="utf-8")
                print(f"  📄  proposed articles saved → {proposed_path}")

            for art in articles:
                if art.title in used_titles:
                    continue
                ok, msg = await verifier.verify_article(art.title, art.doi, art.arxiv_id)
                if ok:
                    print(f"  📥  downloading PDF: {art.title[:55]}…")
                    full_text = await verifier.fetch_full_text(art.arxiv_id)
                    if full_text:
                        word_count = len(full_text.split())
                        enriched = (
                            f"{art.raw}\n"
                            f"- Full Text (from arXiv PDF, {word_count} words):\n"
                            f"{full_text}"
                        )
                        print(f"  ✓   PDF extracted ({word_count} words)")
                    else:
                        abs_text = await verifier.fetch_arxiv_abstract(art.arxiv_id)
                        if abs_text:
                            enriched = re.sub(
                                r"- Abstract:\s*(.+)",
                                f"- Abstract (full): {abs_text}",
                                art.raw,
                                count=1, flags=re.DOTALL,
                            )
                            enriched = re.sub(
                                r"- arXiv:\s*\S+",
                                f"- arXiv: {art.arxiv_id}",
                                enriched,
                            )
                            print(f"  ⚠️   PDF unavailable, using arXiv abstract")
                        else:
                            enriched = art.raw
                            print(f"  ⚠️   no alternative text available")

                    verified_blocks.append(enriched)
                    used_titles.add(art.title)
                    print(f"  ✓   verified: {art.title[:55]}…")
                else:
                    print(f"  ✗   not verified: {art.title[:60]}… — {msg}")

        if len(verified_blocks) < num_articles:
            print(f"  ⚠️   [{subtopic[:50]}…] only {len(verified_blocks)}/{num_articles} articles verified")

        articles_raw = "\n\n".join(
            f"### Article {i+1}\n{block}"
            for i, block in enumerate(verified_blocks[:num_articles])
        )
        if not articles_raw:
            articles_raw = "[No verified articles found]"

        print(f"  ✓   [{subtopic[:50]}…] {len(verified_blocks)} articles verified")

        # ── SummaryAgent ─────────────────────────────────────────
        summarizer = SummaryAgent()
        summary_json = await summarizer.a_run(
            f"Analyse and summarise these articles:\n\n{articles_raw}",
        )
        print(f"  ✓   [{subtopic[:50]}…] analysis complete")
    finally:
        await verifier.close()

    return SubtopicResult(
        subtopic=subtopic,
        articles_raw=articles_raw,
        summary_json=summary_json,
    )


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

async def run_pipeline(topic: str, num_subtopics: int = 3, num_articles: int = 3) -> PipelineResult:
    """
    1. OrchestratorAgent       → decompose into num_subtopics subtopics
    2. [ResearchAgent → proposals (.md) → CitationVerifier(DOI) → SummaryAgent] × N  → asyncio.gather
    3. AggregatorAgent         → Markdown table
    4. RelatedWorksDraftAgent  → narrative draft with citations
    5. NoveltyProposalAgent    → novelty table with difficulty
    6. BibliographyAgent       → formatted IEEE bibliography
    """
    print(f"\n{'═' * 62}")
    print(f"  TOPIC: {topic}")
    print(f"  Subtopics: {num_subtopics}")
    print(f"  Articles per subtopic: {num_articles}")
    print(f"{'═' * 62}\n")

    # global results directory
    topic_slug = _topic_slug(topic)
    results_dir = Path(__file__).parent / "results" / topic_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Orchestrator ─────────────────────────────────────
    print("📋  Orchestrator: topic decomposition…")
    orchestrator = OrchestratorAgent()
    raw_subtopics = await orchestrator.a_run(
        f"Decompose into exactly {num_subtopics} subtopics: {topic}",
    )

    # parsa lista numerata  "1. ...\n2. ...\n..."
    subtopics = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in raw_subtopics.strip().splitlines()
        if re.match(r"^\d+[\.\)]", line.strip())
    ][:num_subtopics]

    if not subtopics:
        subtopics = [l.strip() for l in raw_subtopics.splitlines() if l.strip()][:num_subtopics]

    print(f"  Subtopics:\n" + "\n".join(f"    {i+1}. {st}" for i, st in enumerate(subtopics)))

    # ── Step 2: Parallel agent pairs ─────────────────────────────
    print(f"\n🤖  Launching {len(subtopics)} ResearchAgent+SummaryAgent pairs in parallel…\n")
    results: list[SubtopicResult] = await asyncio.gather(
        *[
            run_subtopic_pipeline(st, results_dir, num_articles=num_articles)
            for st in subtopics
        ]
    )

    # ── Step 3: Aggregator ───────────────────────────────────────
    print("\n📊  Aggregator: building final table…")
    combined_input = "\n\n".join(
        f"=== Subtopic: {r.subtopic} ===\n{r.summary_json}"
        for r in results
    )
    aggregator = AggregatorAgent()
    table = await aggregator.a_run(combined_input)

    # ── Step 4: Related Works Draft ──────────────────────────────
    print("\n✍️   RelatedWorksDraftAgent: writing narrative draft…")
    corpus = build_articles_corpus(results)
    draft_input = (
        f"=== ARTICLES CORPUS ===\n{corpus}\n\n"
        f"=== AGGREGATED TABLE ===\n{table}"
    )
    draft_agent = RelatedWorksDraftAgent()
    draft = await draft_agent.a_run(draft_input)

    # ── Step 5: Novelty Proposal ─────────────────────────────────
    print("\n💡  NoveltyProposalAgent: proposing novelties…")
    novelty_input = (
        f"=== RELATED WORK DRAFT ===\n{draft}\n\n"
        f"=== ARTICLES CORPUS ===\n{corpus}"
    )
    novelty_agent = NoveltyProposalAgent()
    novelties = await novelty_agent.a_run(novelty_input)

    # ── Step 6: Bibliography ─────────────────────────────────────
    print("📚  BibliographyAgent: formatting bibliography…")
    bib_input = (
        f"=== DRAFT ===\n{draft}\n\n"
        f"=== ARTICLES CORPUS ===\n{corpus}"
    )
    bib_agent = BibliographyAgent()
    bibliography = await bib_agent.a_run(bib_input)

    # ── Step 7: Knowledge Base ────────────────────────────────────
    print("📚  Generazione knowledge base per chatbot…")
    kb_parts = [f"# Knowledge Base: {topic}\n"]
    kb_parts.append(f"## Topic\n\n{topic}\n")
    subtopics_list = "\n".join(f"- {r.subtopic}" for r in results)
    kb_parts.append(f"## Subtopics\n\n{subtopics_list}\n")
    kb_parts.append(f"## Aggregated Table\n\n{table}\n")
    kb_parts.append("## Articles\n")
    for r in results:
        kb_parts.append(f"### Subtopic: {r.subtopic}\n\n{r.articles_raw}\n")
    kb_parts.append("## Summaries\n")
    for r in results:
        kb_parts.append(f"### {r.subtopic}\n\n{r.summary_json}\n")
    kb_parts.append(f"## Related Work\n\n{draft}\n")
    kb_parts.append(f"## Novelties\n\n{novelties}\n")
    kb_parts.append(f"## Bibliography\n\n{bibliography}\n")
    knowledge_base = "\n".join(kb_parts)

    print("✅  Pipeline complete.\n")
    return PipelineResult(
        table=table, draft=draft, novelties=novelties, bibliography=bibliography,
        knowledge_base=knowledge_base, subtopic_results=results,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PDF generation with wide-table handling
# ---------------------------------------------------------------------------

def _table_to_list(md_table: str) -> str:
    """
    Convert a wide Markdown table into a per-row key-value list format
    that renders well in PDF regardless of column count.
    """
    lines = md_table.strip().splitlines()
    if not lines:
        return md_table

    header_idx = None
    delim_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if "---" in line:
                delim_idx = i
            elif header_idx is None and delim_idx is None:
                header_idx = i
            if header_idx is not None and delim_idx is not None:
                break

    if header_idx is None or delim_idx is None:
        return md_table

    headers = [h.strip() for h in lines[header_idx].strip().strip("|").split("|")]
    num_cols = len(headers)

    if num_cols <= 5:
        return md_table

    out_parts: list[str] = []
    for line in lines[delim_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        while len(cells) < num_cols:
            cells.append("")
        cells = cells[:num_cols]

        if all(c == "" or c == "..." for c in cells):
            continue

        item_parts: list[str] = []
        for h, c in zip(headers, cells):
            if h in ("#", ""):
                continue
            if c and c != "...":
                item_parts.append(f"  - **{h}:** {c}")
        if item_parts:
            out_parts.append("\n".join(item_parts))

    if not out_parts:
        return md_table

    return "\n\n".join(out_parts) + "\n"


def _fix_wide_tables(markdown_text: str, max_cols: int = 5) -> str:
    """
    Find all Markdown tables in *markdown_text* and convert those with more
    than *max_cols* columns to a row-by-row key-value list format.
    """
    result: list[str] = []
    in_table = False
    table_lines: list[str] = []

    for line in markdown_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(line)
            in_table = True
        else:
            if in_table:
                table_text = "".join(table_lines)
                result.append(_table_to_list(table_text))
                table_lines = []
                in_table = False
            result.append(line)

    if in_table and table_lines:
        table_text = "".join(table_lines)
        result.append(_table_to_list(table_text))

    return "".join(result)


# ---------------------------------------------------------------------------
# Dataset link enrichment via web search
# ---------------------------------------------------------------------------

DATASET_SEARCH_PROMPT = """\
You are a research data librarian. Given a dataset name, find its official
download URL. Reply with ONLY the URL — no preamble, no commentary.

If you cannot find a URL, reply with "NOT FOUND".

Use web search to locate the dataset.
"""


def _find_public_datasets(table_md: str) -> list[tuple[int, str, str]]:
    """
    Parse the aggregated Markdown table and return a list of
    (row_index, dataset_name, current_cell) tuples for datasets
    marked as public but missing a download link/URL.
    """
    lines = table_md.strip().splitlines()
    if not lines:
        return []

    header_idx = None
    delim_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            if "---" in s:
                delim_idx = i
            elif header_idx is None and delim_idx is None:
                header_idx = i
        if header_idx is not None and delim_idx is not None:
            break

    if header_idx is None or delim_idx is None:
        return []

    headers = [h.strip().lower() for h in lines[header_idx].strip().strip("|").split("|")]

    ds_col = None
    for i, h in enumerate(headers):
        if h == "dataset":
            ds_col = i
            break
    if ds_col is None:
        return []

    needs_link: list[tuple[int, str, str]] = []
    for row_idx, line in enumerate(lines[delim_idx + 1:], start=delim_idx + 1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) <= ds_col:
            continue
        cell = cells[ds_col]
        cell_lower = cell.lower()
        has_url = "http://" in cell or "https://" in cell or "[link]" in cell.lower() or "](http" in cell
        is_public = "public" in cell_lower
        if is_public and not has_url and "none" not in cell_lower:
            name = cell.split("(")[0].split(",")[0].strip()
            if name and name.lower() not in ("public", "dataset", "yes", "n/a", ""):
                needs_link.append((row_idx, name, cell))
    return needs_link


def _update_dataset_cell(table_md: str, row_idx: int, new_cell: str) -> str:
    """Replace the Dataset cell at *row_idx* with *new_cell* in the table."""
    lines = table_md.splitlines()
    line = lines[row_idx]
    cells = line.strip().strip("|").split("|")
    header_idx = None
    delim_idx = None
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("|") and s.endswith("|"):
            if "---" in s:
                delim_idx = i
            elif header_idx is None and delim_idx is None:
                header_idx = i
        if header_idx is not None and delim_idx is not None:
            break
    headers = [h.strip().lower() for h in lines[header_idx].strip().strip("|").split("|")]
    ds_col = None
    for i, h in enumerate(headers):
        if h == "dataset":
            ds_col = i
            break
    if ds_col is None:
        return table_md

    cells[ds_col] = f" {new_cell} "
    indent = line[:len(line) - len(line.lstrip())]
    lines[row_idx] = indent + "|" + "|".join(cells) + "|"
    return "\n".join(lines)


async def _enrich_dataset_links_deepseek(table_md: str) -> str:
    """
    Search the web for public dataset names missing download links
    and update the table accordingly (uses DeepSeek API directly).
    """
    needs = _find_public_datasets(table_md)
    if not needs:
        return table_md

    print(f"  🔍  ricerca link per {len(needs)} dataset pubblici...")
    result = table_md
    for row_idx, ds_name, _old_cell in needs:
        client = DeepSeekClient(DATASET_SEARCH_PROMPT)
        url = await client.ask(f"Dataset name: {ds_name}")
        url = url.strip()
        if url and url != "NOT FOUND" and not url.startswith("[Error"):
            url = url.rstrip(".,;")
            if url.startswith("http://") or url.startswith("https://"):
                old = _old_cell
                new = f"{old} ([link]({url}))"
                result = _update_dataset_cell(result, row_idx, new)
                print(f"    ✓ {ds_name} → {url[:60]}...")
            else:
                print(f"    - {ds_name}: risultato non valido ({url[:40]}...)")
        else:
            print(f"    - {ds_name}: link non trovato")

    return result


def _topic_slug(topic: str) -> str:
    """Converte un topic in una stringa usabile come nome di cartella."""
    slug = topic.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:60]


async def main():
    parser = argparse.ArgumentParser(
        description="Scientific Research Pipeline (DeepSeek API) — multi-agent literature survey",
    )
    parser.add_argument("topic", type=str,
                        help="Research topic (e.g. 'Deep learning for medical imaging')")
    parser.add_argument("--subtopics", type=int, default=3,
                        help="Number of parallel subtopics (default: 3)")
    parser.add_argument("--articles", type=int, default=3,
                        help="Number of articles to find per subtopic (default: 3)")
    args = parser.parse_args()

    result = await run_pipeline(
        args.topic, num_subtopics=args.subtopics, num_articles=args.articles,
    )

    # ── Enrich public dataset links via web search ──────────────
    table = await _enrich_dataset_links_deepseek(result.table)
    # Also update knowledge_base with enriched table
    knowledge_base = result.knowledge_base
    knowledge_base = knowledge_base.replace(result.table, table)

    sep = "─" * 62
    print(sep)
    print("FINAL RESULTS")
    print(sep)

    print("\n📊  AGGREGATED TABLE\n")
    print(table)

    print("\n✍️  RELATED WORKS DRAFT\n")
    print(result.draft)

    print("\n💡  PROPOSED NOVELTIES\n")
    print(result.novelties)

    print("\n📚  BIBLIOGRAPHY\n")
    print(result.bibliography)
    print(sep)

    # ── Save to file ───────────────────────────────────────────
    out_dir = Path(__file__).parent / "results" / _topic_slug(args.topic)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "aggregated_table.md": table,
        "related_work.md":    result.draft,
        "novelties.md":       result.novelties,
        "bibliography.md":    result.bibliography,
        "knowledge_base.md":  knowledge_base,
    }
    for name, content in files.items():
        (out_dir / name).write_text(content.strip() + "\n", encoding="utf-8")

    # Combined
    combined = out_dir / "results.md"
    combined.write_text(
        f"# Scientific Research Results\n\n"
        f"**Topic:** {args.topic}\n\n"
        f"## Aggregated Table\n\n{table}\n\n"
        f"## Related Work\n\n{result.draft}\n\n"
        f"## Proposed Novelties\n\n{result.novelties}\n\n"
        f"## References\n\n{result.bibliography}\n",
        encoding="utf-8",
    )

    print(f"\n💾  Results saved to: {out_dir}/")
    for name in files:
        print(f"      {name}")

    # ── Convert to PDF (with wide-table fix) ────────────────────
    try:
        from markdown_pdf import MarkdownPdf, Section

        pdf_path = out_dir / "results.pdf"
        pdf_md = _fix_wide_tables(combined.read_text(encoding="utf-8"), max_cols=5)

        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(pdf_md, paper_size="A4"))
        pdf.save(str(pdf_path))

        print(f"📄  PDF generated: {pdf_path}")
    except ImportError:
        print("⚠️  markdown-pdf not installed. Skipping PDF.")
        print("    Install with: pip install markdown-pdf")
    except Exception as e:
        print(f"⚠️  PDF generation error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
