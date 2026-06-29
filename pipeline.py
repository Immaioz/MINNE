"""
Multi-Agent Scientific Research Pipeline (opencode)
Usa opencode per orchestrare agenti ricercatori e analisti in parallelo.

Flusso:
  OrchestratorAgent
    ├── ResearchAgent(topic_1) ──► SummaryAgent(topic_1)
    ├── ResearchAgent(topic_2) ──► SummaryAgent(topic_2)  (asyncio.gather)
    └── ResearchAgent(topic_N) ──► SummaryAgent(topic_N)
  AggregatorAgent  →  tabella Markdown
  RelatedWorksDraftAgent  →  bozza narrativa con citazioni [N]
  NoveltyProposalAgent  →  tabella novità con difficoltà
  BibliographyAgent  →  bibliografia IEEE
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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
# Agente 2 — ResearchAgent
#   Trova 3 articoli recenti da journal Q1 diversi (knowledge interna, no tool)
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """\
You are an expert scientific literature researcher with deep knowledge of academic
journals and bibliometric rankings.

Task: given a research subtopic, identify exactly {num_articles} recent peer-reviewed
articles (published in the last 3 years) from HIGH-QUALITY journals that are ranked Q1
according to Scimago Journal Rankings (SJR) or JCR.

Rules:
- Each article must come from a DIFFERENT journal.
- All journals must be at least Q1.
- Provide REAL articles you are confident about (title, authors, year, journal, DOI).
- If you are uncertain about the DOI, write "DOI not confirmed".
- Do NOT invent articles or fabricate titles.
- Reply ONLY with the structured list below — no preamble, no commentary.

Format (repeat exactly for each article):

### Article 1
- Title: <full title>
- Authors: <last name, initials; ...>
- Year: <YYYY>
- Journal: <journal name> (Q1, IF ≈ <impact factor>)
- DOI: <doi or "DOI not confirmed">
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

    def __init__(self, num_articles: int = 3):
        self.num_articles = num_articles

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
# Pipeline per singolo sottotema: ResearchAgent → SummaryAgent
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


async def run_subtopic_pipeline(subtopic: str, num_articles: int = 3) -> SubtopicResult:
    """
    Esegue in sequenza ResearchAgent e SummaryAgent per un singolo sottotema.
    L'intera coppia viene lanciata in parallelo con le altre tramite asyncio.gather.
    """
    print(f"  🔍  [{subtopic[:55]}…] ricerca avviata")

    # ── ResearchAgent ────────────────────────────────────────────
    researcher = ResearchAgent(num_articles=num_articles)
    articles_raw = await researcher.a_run(
        f"Find {num_articles} recent Q1 journal articles about: {subtopic}",
    )
    print(f"  ✓   [{subtopic[:55]}…] articoli trovati")

    # ── SummaryAgent ─────────────────────────────────────────────
    summarizer = SummaryAgent()
    summary_json = await summarizer.a_run(
        f"Analyse and summarise these articles:\n\n{articles_raw}",
    )
    print(f"  ✓   [{subtopic[:55]}…] analisi completata")

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
    1. OrchestratorAgent  → scompone in num_subtopics sottotemi
    2. [ResearchAgent + SummaryAgent] × N  → asyncio.gather (parallelo)
    3. AggregatorAgent    → tabella Markdown
    4. RelatedWorksDraftAgent → bozza narrativa con citazioni
    5. NoveltyProposalAgent  → tabella novità con difficoltà
    6. BibliographyAgent  → bibliografia IEEE formattata
    """
    print(f"\n{'═' * 62}")
    print(f"  TOPIC: {topic}")
    print(f"  Sottotemi: {num_subtopics}")
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

    # ── Step 2: Coppie agenti in parallelo ───────────────────────
    print(f"\n🤖  Avvio di {len(subtopics)} coppie ResearchAgent+SummaryAgent in parallelo…\n")
    results: list[SubtopicResult] = await asyncio.gather(
        *[run_subtopic_pipeline(st, num_articles=num_articles) for st in subtopics]
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


if __name__ == "__main__":
    asyncio.run(main())
