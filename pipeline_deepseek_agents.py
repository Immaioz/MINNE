"""Agent definitions and prompt templates for the DeepSeek pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


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
                        msg += f"\n\n\U0001f4a1 {hint}"
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


class Agent:
    """Minimal agent: delegates to DeepSeek API directly."""
    name = ""
    system_prompt = ""

    async def a_run(self, prompt: str, **kwargs) -> str:
        client = DeepSeekClient(self.system_prompt)
        return await client.ask(prompt)


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


SUMMARY_PROMPT = """\
You are a concise academic analyst.

You receive a structured list of scientific articles and must produce a JSON array
with one object per article. No tools, no web search — only analyse the text you receive.

Each JSON object MUST have exactly these six keys:
  "title"       : string — full article title
  "results"     : string — 1-2 sentences describing the KEY FINDINGS (max 30 words)
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
  (2-3 sentences).
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


DATASET_SEARCH_PROMPT = """\
You are a research data librarian. Given a dataset name, find its official
download URL. Reply with ONLY the URL — no preamble, no commentary.

If you cannot find a URL, reply with "NOT FOUND".

Use web search to locate the dataset.
"""
