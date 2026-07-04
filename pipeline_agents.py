"""Agent definitions and prompt templates for the research pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

# Carica .env se presente (per OPENCODE_API_KEY e altre variabili)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Usa la chiave API da .env invece della sessione opencode salvata
_AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"


def ensure_opencode_api_key_from_env() -> None:
    """Sostituisce la chiave API in auth.json con quella da .env."""
    env_key = os.environ.get("OPENCODE_API_KEY", "")
    if not env_key:
        return
    try:
        if _AUTH_PATH.exists():
            auth = json.loads(_AUTH_PATH.read_text(encoding="utf-8"))
            auth.setdefault("opencode", {})["key"] = env_key
            _AUTH_PATH.write_text(json.dumps(auth, indent=2), encoding="utf-8")
    except Exception:
        pass


# Keep import-time behavior for backward compatibility.
ensure_opencode_api_key_from_env()


class OpencodeClient:
    """Invia prompt a opencode via `opencode run --agent <name>`."""

    TIMEOUT = 300  # secondi

    ERROR_PATTERNS = [
        (["insufficient", "credit"], "Your API account has run out of credits/tokens. Top up at https://opencode.ai/auth"),
        (["insufficient", "quota"], "Your API quota is exhausted. Check your plan at https://opencode.ai/auth"),
        (["insufficient", "balance"], "Your API balance is too low. Add funds at https://opencode.ai/auth"),
        (["insufficient", "fund"], "Your API account has insufficient funds. Add credits at https://opencode.ai/auth"),
        (["payment", "required"], "Payment required. Your API account may be inactive or out of credits."),
        (["billing"], "A billing issue with your API account. Check https://opencode.ai/auth"),
        (["rate limit", "429"], "Rate-limited by the API. The pipeline will retry automatically on the next run."),
        (["too many request"], "Too many requests. The pipeline will retry automatically on the next run."),
        (["token limit"], "The prompt exceeds the model's maximum context length. Try reducing the number of articles per subtopic."),
        (["maximum context"], "The prompt exceeds the model's maximum context length. Try reducing the number of articles per subtopic."),
        (["context length"], "The combined prompt is too long for the model's context window. Reduce --articles or --subtopics."),
        (["unauthorized", "key"], "Your API key is invalid or unauthorized. Check your OPENCODE_API_KEY in .env"),
        (["invalid", "api key"], "Your API key is invalid. Check your OPENCODE_API_KEY in .env"),
        (["authentication"], "Authentication failed. Verify your API key in .env or at https://opencode.ai/auth"),
        (["server error", "500"], "The LLM provider returned a server error (500). This is usually temporary; retry the pipeline."),
        (["server error", "503"], "The LLM provider is unavailable (503). This is usually temporary; retry the pipeline."),
        (["502"], "The LLM provider returned a bad gateway (502). This is usually temporary; retry the pipeline."),
        (["timeout", "504"], "The LLM provider timed out (504). This is usually temporary; retry the pipeline."),
        (["deadline"], "The API request exceeded the deadline. Try again or reduce the prompt size."),
    ]

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def ask(self, prompt: str) -> str:
        return await asyncio.to_thread(self._ask_sync, prompt)

    @staticmethod
    def _classify_error(stderr: str, stdout: str) -> str:
        combined = (stderr + " " + stdout).lower()
        for keywords, hint in OpencodeClient.ERROR_PATTERNS:
            if all(kw in combined for kw in keywords):
                return hint
        return ""

    def _ask_sync(self, prompt: str) -> str:
        safe = prompt.replace("\x00", "")
        cmd = ["opencode", "run", "--agent", self.agent_name]
        try:
            result = subprocess.run(
                cmd,
                input=safe,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=Path(__file__).parent,
            )
            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip()
                hint = self._classify_error(result.stderr, result.stdout)
                if hint:
                    return f"[Error] opencode exit {result.returncode}: {err}\n\n💡 {hint}"
                return f"[Error] opencode exit {result.returncode}: {err}"
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


class Agent:
    """Minimal agent: delegates to opencode via `opencode run --agent <name>`."""

    name = ""
    system_prompt = ""
    opencode_agent = ""

    async def a_run(self, prompt: str, **kwargs) -> str:
        client = OpencodeClient(self.opencode_agent)
        full = (self.system_prompt + "\n\n" + prompt) if self.system_prompt else prompt
        return await client.ask(full)


ORCHESTRATOR_PROMPT = """\
You are a research orchestrator. Given a broad research topic, decompose it into
distinct, complementary subtopics suitable for parallel bibliographic search.

Each subtopic should:
- Be focused and specific enough for a targeted literature search
- Cover a different aspect of the broader topic (minimal overlap)
- Be recognisable as a standalone research direction

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
        full_prompt = full_prompt.replace("\x00", "")
        return await client.ask(full_prompt)


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
    opencode_agent = "summarizer"


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
    opencode_agent = "aggregator"


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
- Sort by citation number (ascending).
- All entries must be real — do not invent details.
- Reply with ONLY the reference list — no preamble, no commentary.
"""


class BibliographyAgent(Agent):
    name = "BibliographyAgent"
    system_prompt = BIBLIOGRAPHY_PROMPT
    opencode_agent = "bibliography"


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
    opencode_agent = "novelty-proposal"


DATASET_SEARCH_PROMPT = """\
You are a research data librarian. Given a dataset name, find its official
download URL. Reply with ONLY the URL — no preamble, no commentary.

If you cannot find a URL, reply with "NOT FOUND".

Use web search to locate the dataset.
"""
