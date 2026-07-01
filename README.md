# Multi-Agent Scientific Research Pipeline

Orchestrates AI agents via `opencode` to search, verify, and analyze scientific papers from Q1 journals, producing structured Markdown reports and PDFs.

---

## Architecture

```
OrchestratorAgent (opencode run --agent orchestrator)
    └── asyncio.gather (parallel)
          ├── ResearchAgent(topic_1)
          │     └── literature.search_papers()       # OpenAlex + Semantic Scholar + arXiv
          │     └── CitationVerifier (L1: arXiv DOI, L2: Crossref title match)
          │     └── PDF download + text extraction
          │     └── SummaryAgent(topic_1)
          ├── ResearchAgent(topic_2)
          │     └── ...
          └── ResearchAgent(topic_N)
                └── ...
  AggregatorAgent        → Markdown table
  RelatedWorksDraftAgent → narrative draft with [N] citations
  NoveltyProposalAgent   → novelty table with difficulty (★–★★★)
  BibliographyAgent      → IEEE-formatted bibliography
```

Each agent is an `opencode` subprocess running the corresponding agent definition in `.opencode/agents/`. The prompt is passed via **stdin** (not CLI arguments) to avoid Linux `ARG_MAX` limits on large contexts (~20+ papers).

### Agents

| Agent | Role |
|-------|------|
| `OrchestratorAgent` | Decomposes the topic into N distinct subtopics |
| `ResearchAgent` | Finds N recent Q1 articles per subtopic (different journals) via OpenAlex + S2 + arXiv |
| `CitationVerifier` | Two-level verification: L1 — arXiv API metadata, L2 — Crossref DOI/title match |
| `SummaryAgent` | Structures validated papers into JSON |
| `AggregatorAgent` | Merges all JSON arrays into a Markdown table |
| `RelatedWorksDraftAgent` | Writes a narrative state-of-the-art with inline [N] citations |
| `NoveltyProposalAgent` | Proposes 4–5 feasible novelties with difficulty ratings |
| `BibliographyAgent` | Formats references in IEEE style |

### Literature search sources (fallback chain)

1. **OpenAlex** — primary Q1 journal search, filtered by type, year, and citation count
2. **Semantic Scholar** — fallback if OpenAlex is unavailable (rate-limited, retries up to 3×)
3. **arXiv** — final fallback via direct API + `arxiv` Python library

All requests use polite pool with `mailto` (OpenAlex) and exponential backoff.

### Citation verification

Each paper undergoes two independent checks before acceptance:

- **L1 (arXiv)** — resolves the arXiv ID or queries by title; paper must exist
- **L2 (Crossref)** — resolves the DOI, compares returned title with expected title (fuzzy match via `difflib`)

Both levels must pass, otherwise the ResearchAgent retries with a replacement (up to 5 attempts).

---

## Setup

### Requirements

```bash
pip install flask httpx markdown markdown-pdf arxiv python-dotenv PyMuPDF
```

Create a `.env` file in the project root:

```env
OPENCODE_API_KEY=sk-your-key-here
```

The API key is automatically loaded at startup and injected into `opencode`'s credentials (`~/.local/share/opencode/auth.json`), overriding any saved session.

### Pipeline CLI

```bash
python pipeline.py "Research topic" --subtopics 3 --articles 3
```

Arguments:
- `topic` (positional) — research topic string
- `--subtopics` — number of parallel subtopics (default: 3)
- `--articles` — articles per subtopic (default: 3)

The pipeline searches for papers, verifies citations, downloads PDFs, summarizes, aggregates, drafts a related-work section, proposes novelties, and formats the bibliography — all in one run.

### Webapp

```bash
python webapp.py
# → http://127.0.0.1:5050
```

Browser interface for:
- Launching the pipeline via form (topic, subtopics, articles)
- Real-time execution status
- Browsing past results (rendered Markdown → HTML)
- Downloading raw files (Markdown, PDF)

---

## Agent definitions

System prompts for each agent live in `.opencode/agents/*.md`. These are invoked by `opencode run --agent <name>` and are the single source of truth for agent behavior. Editing an agent file changes its behavior across the pipeline.

| File | Agent |
|------|-------|
| `.opencode/agents/orchestrator.md` | Topic decomposition |
| `.opencode/agents/researcher.md` | Q1 journal article search |
| `.opencode/agents/summarizer.md` | Article analysis & JSON output |
| `.opencode/agents/aggregator.md` | JSON consolidation into table |
| `.opencode/agents/related-works-draft.md` | Narrative draft with citations |
| `.opencode/agents/novelty-proposal.md` | Novelty proposals |
| `.opencode/agents/bibliography.md` | IEEE bibliography formatting |

---

## Output

```
results/<topic-slug>/
├── aggregated_table.md    # summary article table
├── related_work.md        # narrative draft with [N] citations
├── novelties.md           # 4–5 novelty proposals with difficulty (★–★★★)
├── bibliography.md        # IEEE-style references
├── results.md             # unified full report
└── results.pdf            # PDF export (via markdown-pdf)
```

Intermediate files per subtopic are saved under `results/<topic-slug>/proposed/`.

---

## Technical notes

- Each agent runs as an `opencode` subprocess (`opencode run --agent <name>`) with the prompt piped via **stdin** to avoid command-line length limits.
- The API key is read from `.env` (`OPENCODE_API_KEY`) and written into `~/.local/share/opencode/auth.json` at startup, overriding any previously saved session credentials.
- Parallelism for ResearchAgent → CitationVerifier → SummaryAgent pairs is managed via `asyncio.gather`.
- If no papers pass verification for a subtopic, the pipeline retries with rejection feedback (up to 5 attempts).
- PDF download is attempted via the Unpaywall DOI proxy (`doi.org`) and full-text is extracted with `PyMuPDF` (fitz); if unavailable, a fallback sentence is used.
- PDF output requires `markdown-pdf` (optional; skipped if not installed).
- All agents in `.opencode/agents/*.md` use `mode: all`.
