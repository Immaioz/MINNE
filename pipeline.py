"""
Multi-Agent Scientific Research Pipeline (opencode)
Usa opencode per orchestrare agenti ricercatori e analisti in parallelo.

Flusso:
  OrchestratorAgent
    └── JournalAssignerAgent  (assegna journal Q1 a ogni sottotema)
          ├── ResearchAgent(topic_1, journal_A) ──► CitationVerifier(L1+L2) ──► SummaryAgent(topic_1)
          ├── ResearchAgent(topic_2, journal_B) ──► CitationVerifier(L1+L2) ──► SummaryAgent(topic_2)
          └── ResearchAgent(topic_N, journal_N) ──► CitationVerifier(L1+L2) ──► SummaryAgent(topic_N)
  AggregatorAgent  →  tabella Markdown
  RelatedWorksDraftAgent  →  bozza narrativa con citazioni [N]
  NoveltyProposalAgent  →  tabella novità con difficoltà
  BibliographyAgent  →  bibliografia IEEE

Verifica citazioni:
  L1 — arXiv API: query per titolo, verifica esistenza paper
  L2 — DOI resolution: risolve DOI via Crossref API, confronta titolo
  Se un paper non supera entrambi i livelli, il ResearchAgent cerca un sostituto.
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

# Carica .env se presente (per OPENCODE_API_KEY e altre variabili)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# LLM Client  —  chiamate API tramite `opencode run`
# ---------------------------------------------------------------------------

class OpencodeClient:
    """Invia prompt a opencode via `opencode run --agent <name>`."""

    TIMEOUT = 300  # secondi

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def ask(self, prompt: str) -> str:
        return await asyncio.to_thread(self._ask_sync, prompt)

    def _ask_sync(self, prompt: str) -> str:
        cmd = ["opencode", "run", "--agent", self.agent_name, "--", prompt]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=Path(__file__).parent,
            )
            if result.returncode != 0:
                return (
                    f"[Error] opencode exit {result.returncode}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            return result.stdout.strip()
        except FileNotFoundError:
            return (
                "[Error] opencode non trovato. Installalo con:\n"
                "  curl -fsSL https://opencode.ai/install | bash\n"
                "oppure:\n  npm install -g opencode-ai"
            )
        except subprocess.TimeoutExpired:
            return f"[Error] opencode ha superato il timeout di {self.TIMEOUT}s"
        except Exception as e:
            return f"[Error] {e}"


# ---------------------------------------------------------------------------
# Agente base
# ---------------------------------------------------------------------------

class Agent:
    """Agente minimale: delega a opencode via `opencode run --agent <name>`."""
    name = ""
    system_prompt = ""  # usato solo dalla pipeline Python, non da opencode
    opencode_agent = ""  # nome del subagente in .opencode/agents/

    async def a_run(self, prompt: str, **kwargs) -> str:
        client = OpencodeClient(self.opencode_agent)
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
    opencode_agent = "orchestrator"


# ---------------------------------------------------------------------------
# Agente 1b — JournalAssignerAgent
#   Assegna un journal Q1 diverso a ciascun sottotema
# ---------------------------------------------------------------------------

JOURNAL_ASSIGNER_PROMPT = """\
You are a research journal assigner. Given a list of research subtopics, assign
to each one a different high-quality Q1 journal according to Scimago Journal
Rankings (SJR) or JCR that is highly relevant to that subtopic.

Rules:
- Each subtopic must get a DIFFERENT journal.
- All journals must be Q1.
- Assign journals that publish research closely related to the subtopic.
- Reply with ONLY the mapping — one line per subtopic, no extra commentary.

Format:
1. <subtopic> → <Journal Name>
2. <subtopic> → <Journal Name>
...
"""

class JournalAssignerAgent(Agent):
    name = "JournalAssignerAgent"
    system_prompt = JOURNAL_ASSIGNER_PROMPT
    opencode_agent = "journal-assigner"


# ---------------------------------------------------------------------------
# Agente 2 — ResearchAgent
#   Trova N articoli recenti da uno specifico journal Q1
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """\
You are an expert scientific literature researcher with deep knowledge of academic
journals and bibliometric rankings.

Task: given a research subtopic and a TARGET journal, identify exactly
{num_articles} recent peer-reviewed articles (published in the last 3 years)
published SPECIFICALLY in that target journal.

Rules:
- ALL articles MUST be from the specified target journal.
- The target journal is Q1 — confirm the quartile in the output.
- ALL articles MUST have a valid DOI.
- ALL articles MUST be available on arXiv as preprints (they were posted on arXiv
  before or after journal publication).
- Provide REAL articles you are confident about (title, authors, year, journal, DOI).
- Do NOT invent articles or fabricate titles.
- Reply ONLY with the structured list below — no preamble, no commentary.

Format (repeat exactly for each article):

### Article 1
- Title: <full title>
- Authors: <last name, initials; ...>
- Year: <YYYY>
- Journal: <journal name> (Q1, IF ≈ <impact factor>)
- DOI: <doi>
- arXiv: <arXiv ID, e.g. 2301.12345>
- Abstract: <2–3 sentences on key contribution and findings>

### Article 2
...

### Article N
...
"""

class ResearchAgent(Agent):
    name = "ResearchAgent"
    system_prompt = RESEARCH_PROMPT
    opencode_agent = "researcher"

    def __init__(self, num_articles: int = 3, target_journal: str = ""):
        self.num_articles = num_articles
        self.target_journal = target_journal

    async def a_run(self, prompt: str, **kwargs) -> str:
        client = OpencodeClient(self.opencode_agent)
        full_prompt = (
            self.system_prompt.format(num_articles=self.num_articles)
            + "\n\n"
            + prompt
        )
        return await client.ask(full_prompt)


# ---------------------------------------------------------------------------
# Agente 3 — SummaryAgent
#   Struttura i risultati del ResearchAgent in un JSON array (no tool)
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """\
You are a concise academic analyst.

You receive a structured list of scientific articles and must produce a JSON array
with one object per article. No tools, no web search — only analyse the text you receive.

Each JSON object MUST have exactly these three keys:
  "title"   : string — full article title
  "results" : string — 1–2 sentences describing the KEY FINDINGS (max 30 words)
  "journal" : string — journal name and quartile, e.g. "Nature Machine Intelligence (Q1)"

Reply with ONLY the raw JSON array — no markdown fences, no explanatory text.
"""

class SummaryAgent(Agent):
    name = "SummaryAgent"
    system_prompt = SUMMARY_PROMPT
    opencode_agent = "summarizer"


# ---------------------------------------------------------------------------
# Agente 4 — AggregatorAgent
#   Consolida tutti i JSON in una tabella Markdown finale
# ---------------------------------------------------------------------------

AGGREGATOR_PROMPT = """\
You are a scientific report writer.

You receive multiple JSON arrays of article summaries, each prefixed with its subtopic.
Produce a single clean Markdown table with these columns:

| # | Title | Key Results | Journal | Subtopic |
|---|-------|-------------|---------|----------|

Rules:
- Include ALL articles from all subtopics.
- Keep "Key Results" under 25 words per cell.
- Sort rows by Subtopic, then alphabetically by Title.
- Reply with ONLY the Markdown table — no title, no preamble, no trailing text.
"""

class AggregatorAgent(Agent):
    name = "AggregatorAgent"
    system_prompt = AGGREGATOR_PROMPT
    opencode_agent = "aggregator"


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
    opencode_agent = "related-works-draft"


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
    opencode_agent = "bibliography"


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

Identify 3–5 concrete research directions / novelties that build on the gaps
and challenges described in the draft. For each novelty produce:

| # | Novelty | Description | Difficulty | Rationale |
|---|---------|-------------|------------|-----------|

Rules:
- "Difficulty" must be one of: ★ Easy, ★★ Medium, ★★★ Hard.
- "Description" must clearly state WHAT would be done and WHY it is novel
  (2–3 sentences).
- "Rationale" must cite the relevant articles from the draft [N] and explain
  why this fills a gap.
- All novelties MUST be realistically implementable (no purely theoretical
  or data-unavailable proposals).
- Reply with ONLY the Markdown table — no preamble, no commentary.
- Do NOT invent article details beyond what the corpus provides.
"""

class NoveltyProposalAgent(Agent):
    name = "NoveltyProposalAgent"
    system_prompt = NOVELTY_PROMPT
    opencode_agent = "novelty-proposal"


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
        Esegue L1 (arXiv) e L2 (DOI).
        L1 usa arXiv ID se fornito, altrimenti cerca per titolo.
        Un paper è verificato solo se SUPERA ENTRAMBI i livelli.
        Se non ha DOI, il L2 fallisce automaticamente.
        """
        if arxiv_id:
            l1_ok, l1_msg = await self.verify_arxiv_id(arxiv_id)
        else:
            l1_ok, l1_msg = await self.verify_arxiv(title)

        has_doi = doi.lower().strip() not in ("", "doi not confirmed", "not available", "none", "n/a")
        l2_ok, l2_msg = await self.verify_doi(doi, title) if has_doi else (False, "DOI non disponibile")

        if l1_ok and l2_ok:
            return True, f"L1 OK ({l1_msg}) · L2 OK ({l2_msg})"
        return False, f"L1: {l1_msg} | L2: {l2_msg}"


# ---------------------------------------------------------------------------
# Pipeline per singolo sottotema: ResearchAgent → CitationVerifier → SummaryAgent
# ---------------------------------------------------------------------------

@dataclass
class SubtopicResult:
    subtopic: str
    target_journal: str
    articles_raw: str   # output grezzo del ResearchAgent
    summary_json: str   # JSON strutturato del SummaryAgent


@dataclass
class PipelineResult:
    table: str          # tabella Markdown da AggregatorAgent
    draft: str          # bozza narrativa da RelatedWorksDraftAgent
    novelties: str      # tabella novità da NoveltyProposalAgent
    bibliography: str   # bibliografia formattata da BibliographyAgent


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
    subtopic: str, target_journal: str = "", num_articles: int = 3,
) -> SubtopicResult:
    """
    Esegue ResearchAgent → CitationVerifier (L1+L2, con retry) → SummaryAgent.
    L'intera catena viene lanciata in parallelo con le altre tramite asyncio.gather.
    """
    print(f"  🔍  [{subtopic[:50]}…] ricerca su «{target_journal}»")

    verifier = CitationVerifier()
    try:
        # ── ResearchAgent (con retry finché non abbiamo N articoli verificati) ──
        verified_blocks: list[str] = []
        used_titles: set[str] = set()
        prompt_prefix = (
            f"Research subtopic: {subtopic}\n"
            f"Target journal: {target_journal}\n"
        )

        for attempt in range(5):
            still_needed = num_articles - len(verified_blocks)
            if still_needed <= 0:
                break

            if attempt == 0:
                prompt = (
                    f"{prompt_prefix}"
                    f"Find {num_articles} articles published in {target_journal}."
                )
            else:
                exclude = "; ".join(sorted(used_titles)[:10])
                prompt = (
                    f"{prompt_prefix}"
                    f"Find {still_needed} DIFFERENT articles published in {target_journal}.\n"
                    f"Do NOT repeat any of these already-found articles:\n{exclude}"
                )

            researcher = ResearchAgent(
                num_articles=still_needed, target_journal=target_journal,
            )
            raw = await researcher.a_run(prompt)
            articles = parse_articles(raw)

            if not articles:
                print(f"  ⚠️   [{subtopic[:50]}…] Researcher non ha prodotto articoli validi (tentativo {attempt+1})")
                continue

            for art in articles:
                if art.title in used_titles:
                    continue
                ok, msg = await verifier.verify_article(art.title, art.doi, art.arxiv_id)
                if ok:
                    verified_blocks.append(art.raw)
                    used_titles.add(art.title)
                    print(f"  ✓   verificato: {art.title[:60]}…")
                else:
                    print(f"  ✗   non verificato: {art.title[:60]}… — {msg}")

        if len(verified_blocks) < num_articles:
            print(f"  ⚠️   [{subtopic[:50]}…] solo {len(verified_blocks)}/{num_articles} articoli verificati")

        # ricostruisce il testo completo solo con gli articoli verificati
        articles_raw = "\n\n".join(
            f"### Article {i+1}\n{block}"
            for i, block in enumerate(verified_blocks[:num_articles])
        )
        if not articles_raw:
            articles_raw = "[No verified articles found]"

        print(f"  ✓   [{subtopic[:50]}…] {len(verified_blocks)} articoli verificati")

        # ── SummaryAgent ─────────────────────────────────────────
        summarizer = SummaryAgent()
        summary_json = await summarizer.a_run(
            f"Analyse and summarise these articles:\n\n{articles_raw}",
        )
        print(f"  ✓   [{subtopic[:50]}…] analisi completata")
    finally:
        await verifier.close()

    return SubtopicResult(
        subtopic=subtopic,
        target_journal=target_journal,
        articles_raw=articles_raw,
        summary_json=summary_json,
    )


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

async def run_pipeline(topic: str, num_subtopics: int = 3, num_articles: int = 3) -> PipelineResult:
    """
    1. OrchestratorAgent       → scompone in num_subtopics sottotemi
    2. JournalAssignerAgent     → assegna un journal Q1 a ogni sottotema
    3. [ResearchAgent → CitationVerifier(L1+L2, retry) → SummaryAgent] × N  → asyncio.gather
    4. AggregatorAgent         → tabella Markdown
    5. RelatedWorksDraftAgent  → bozza narrativa con citazioni
    6. NoveltyProposalAgent    → tabella novità con difficoltà
    7. BibliographyAgent       → bibliografia IEEE formattata
    """
    print(f"\n{'═' * 62}")
    print(f"  TOPIC: {topic}")
    print(f"  Sottotemi: {num_subtopics}")
    print(f"  Articoli per sottotema: {num_articles}")
    print(f"{'═' * 62}\n")

    # ── Step 1: Orchestratore ────────────────────────────────────
    print("📋  Orchestratore: scomposizione del topic…")
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

    print(f"  Sottotemi:\n" + "\n".join(f"    {i+1}. {st}" for i, st in enumerate(subtopics)))

    # ── Step 2: Journal Assigner ─────────────────────────────────
    print("\n📌  JournalAssigner: assegnazione journal target…")
    assigner = JournalAssignerAgent()
    raw_journals = await assigner.a_run(
        "Assign one different Q1 journal to each of these subtopics:\n"
        + "\n".join(f"{i+1}. {st}" for i, st in enumerate(subtopics))
    )

    # parsa "N. subtopic → Journal Name"
    journal_map: dict[str, str] = {}
    for line in raw_journals.strip().splitlines():
        m = re.match(r"\d+[\.\)]\s*(.*?)\s*→\s*(.*)", line.strip())
        if m:
            st = m.group(1).strip()
            jn = m.group(2).strip()
            journal_map[st] = jn

    # fallback: assegna vuoto se il parse fallisce
    target_journals = [journal_map.get(st, "") for st in subtopics]

    for i, st in enumerate(subtopics):
        j = target_journals[i]
        print(f"    {i+1}. {st[:50]}… → {j or '(nessun journal assegnato)'}")

    # ── Step 3: Coppie agenti in parallelo ───────────────────────
    print(f"\n🤖  Avvio di {len(subtopics)} coppie ResearchAgent+SummaryAgent in parallelo…\n")
    results: list[SubtopicResult] = await asyncio.gather(
        *[
            run_subtopic_pipeline(st, target_journal=j, num_articles=num_articles)
            for st, j in zip(subtopics, target_journals)
        ]
    )

    # ── Step 3: Aggregatore ──────────────────────────────────────
    print("\n📊  Aggregatore: composizione tabella finale…")
    combined_input = "\n\n".join(
        f"=== Subtopic: {r.subtopic} ===\n{r.summary_json}"
        for r in results
    )
    aggregator = AggregatorAgent()
    table = await aggregator.a_run(combined_input)

    # ── Step 4: Related Works Draft ──────────────────────────────
    print("\n✍️   RelatedWorksDraftAgent: scrittura bozza narrativa…")
    corpus = build_articles_corpus(results)
    draft_input = (
        f"=== ARTICLES CORPUS ===\n{corpus}\n\n"
        f"=== AGGREGATED TABLE ===\n{table}"
    )
    draft_agent = RelatedWorksDraftAgent()
    draft = await draft_agent.a_run(draft_input)

    # ── Step 5: Novelty Proposal ─────────────────────────────────
    print("\n💡  NoveltyProposalAgent: proposta novità…")
    novelty_input = (
        f"=== RELATED WORK DRAFT ===\n{draft}\n\n"
        f"=== ARTICLES CORPUS ===\n{corpus}"
    )
    novelty_agent = NoveltyProposalAgent()
    novelties = await novelty_agent.a_run(novelty_input)

    # ── Step 6: Bibliography ─────────────────────────────────────
    print("📚  BibliographyAgent: formattazione bibliografia…")
    bib_input = (
        f"=== DRAFT ===\n{draft}\n\n"
        f"=== ARTICLES CORPUS ===\n{corpus}"
    )
    bib_agent = BibliographyAgent()
    bibliography = await bib_agent.a_run(bib_input)

    print("✅  Pipeline completata.\n")
    return PipelineResult(
        table=table, draft=draft, novelties=novelties, bibliography=bibliography,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _topic_slug(topic: str) -> str:
    """Converte un topic in una stringa usabile come nome di cartella."""
    slug = topic.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:60]


async def main():
    parser = argparse.ArgumentParser(
        description="Scientific Research Pipeline — multi-agent literature survey",
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

    sep = "─" * 62
    print(sep)
    print("RISULTATI FINALI")
    print(sep)

    print("\n📊  TABELLA AGGREGATA\n")
    print(result.table)

    print("\n✍️  BOZZA RELATED WORKS\n")
    print(result.draft)

    print("\n💡  NOVITÀ PROPOSTE\n")
    print(result.novelties)

    print("\n📚  BIBLIOGRAFIA\n")
    print(result.bibliography)
    print(sep)

    # ── Salva su file ──────────────────────────────────────────
    out_dir = Path(__file__).parent / "results" / _topic_slug(args.topic)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "aggregated_table.md": result.table,
        "related_work.md":    result.draft,
        "novelties.md":       result.novelties,
        "bibliography.md":    result.bibliography,
    }
    for name, content in files.items():
        (out_dir / name).write_text(content.strip() + "\n", encoding="utf-8")

    # Combinato
    combined = out_dir / "results.md"
    combined.write_text(
        f"# Scientific Research Results\n\n"
        f"**Topic:** {args.topic}\n\n"
        f"## Aggregated Table\n\n{result.table}\n\n"
        f"## Related Work\n\n{result.draft}\n\n"
        f"## Proposed Novelties\n\n{result.novelties}\n\n"
        f"## References\n\n{result.bibliography}\n",
        encoding="utf-8",
    )

    print(f"\n💾  Risultati salvati in: {out_dir}/")
    for name in files:
        print(f"      {name}")

    # ── Converti in PDF ──────────────────────────────────────────
    try:
        from markdown_pdf import MarkdownPdf, Section

        pdf_path = out_dir / "results.pdf"
        md_content = (out_dir / "results.md").read_text(encoding="utf-8")

        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(md_content, paper_size="A4"))
        pdf.save(str(pdf_path))

        print(f"📄  PDF generato: {pdf_path}")
    except ImportError:
        print("⚠️  markdown-pdf non installato. Salta PDF.")
        print("    Installa con: pip install markdown-pdf")
    except Exception as e:
        print(f"⚠️  Errore generazione PDF: {e}")


if __name__ == "__main__":
    asyncio.run(main())
