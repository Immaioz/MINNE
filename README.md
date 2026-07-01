# **M**ulti-agent **I**ntelligence for **N**ovelty **N**avigation and **E**xploration

Orchestrates AI agents via `opencode` (or DeepSeek API directly) to search, verify, and analyze scientific papers from Q1 journals, producing structured Markdown reports, PDFs, and an interactive chatbot knowledge base.

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
  KnowledgeBaseGen       → chatbot-ready knowledge_base.md
```

Each agent runs as an `opencode` subprocess (or a direct DeepSeek API call in `pipeline_deepseek.py`). The prompt is passed via **stdin** (not CLI arguments) to avoid Linux `ARG_MAX` limits on large contexts (~20+ papers).

### Agents

| Agent | Role |
|-------|------|
| `OrchestratorAgent` | Decomposes the topic into N distinct subtopics |
| `ResearchAgent` | Finds N recent Q1 articles per subtopic (different journals) via OpenAlex + S2 + arXiv |
| `CitationVerifier` | Two-level verification: L1 — arXiv API metadata, L2 — Crossref DOI/title match |
| `SummaryAgent` | Structures validated papers into JSON (title, results, journal, methodology, dataset, code) |
| `AggregatorAgent` | Merges all JSON arrays into a Markdown table (with Methodology, Dataset, Code columns) |
| `RelatedWorksDraftAgent` | Writes a narrative state-of-the-art with inline [N] citations |
| `NoveltyProposalAgent` | Proposes 4–5 feasible novelties with difficulty ratings and detailed discussion per novelty |
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
# For opencode pipeline (pipeline.py)
OPENCODE_API_KEY=sk-your-opencode-key-here

# For DeepSeek pipeline (pipeline_deepseek.py)
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
DEEPSEEK_MODEL=deepseek-chat
```

The opencode API key is automatically loaded at startup and injected into `opencode`'s credentials (`~/.local/share/opencode/auth.json`), overriding any saved session.

### Pipeline CLI (opencode)

```bash
python pipeline.py "Research topic" --subtopics 3 --articles 3
```

Arguments:
- `topic` (positional) — research topic string
- `--subtopics` — number of parallel subtopics (default: 3)
- `--articles` — articles per subtopic (default: 3)

Uses `opencode run --agent <name>` for each agent step.

### Pipeline CLI (DeepSeek direct)

```bash
python pipeline_deepseek.py "Research topic" --subtopics 3 --articles 3
```

Identical interface but calls the DeepSeek Chat API directly (no opencode dependency). Useful if you already have a DeepSeek API key and want to avoid the opencode subscription.

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
- **Chat interface** — interactive Q&A with an AI agent on any completed result, powered by the generated knowledge base

---

## Chatbot Interface

Each completed pipeline run generates a `knowledge_base.md` that powers an **interactive chatbot** at `/chat/<topic-slug>`.

### How it works

1. The pipeline produces `knowledge_base.md` containing all articles, summaries, related work, novelties, and bibliography
2. The webapp exposes a chat page where you can ask questions about the research topic
3. The agent answers **only** based on the knowledge base content (no external search)
4. Chat history is kept in `current.json` during the session

### Save & Enrich

When you close a chat session via the **Save & Close** button:
- A dated chat report is saved to `results/<slug>/chats/chat_YYYYMMDD_HHMMSS.md`
- The Q&A content is **appended to the knowledge base** (`knowledge_base.md`), so future chat sessions benefit from all previous interactions
- The current session is cleared for a fresh start

Saved chat reports are listed on the chat page for later review.

---

## Output

```
results/<topic-slug>/
├── aggregated_table.md    # summary article table (with Methodology, Dataset, Code columns)
├── related_work.md        # narrative draft with [N] citations
├── novelties.md           # novelty table + detailed discussions (methodology, dataset, baselines, metrics, roadmap)
├── bibliography.md        # IEEE-style references
├── results.md             # unified full report
├── results.pdf            # PDF export (wide tables auto-converted to bullet lists)
├── knowledge_base.md      # chatbot-ready knowledge base with all content
├── proposed/              # intermediate raw article proposals per subtopic
└── chats/                 # saved chat reports (one .md per session)
```

Intermediate files per subtopic are saved under `results/<topic-slug>/proposed/`.

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

## Technical notes

- Each agent runs as an `opencode` subprocess (`opencode run --agent <name>`) with the prompt piped via **stdin** to avoid command-line length limits.
- `pipeline_deepseek.py` uses direct DeepSeek API calls instead of opencode subprocesses, but follows the same workflow.
- The API key is read from `.env` (`OPENCODE_API_KEY`) and written into `~/.local/share/opencode/auth.json` at startup, overriding any previously saved session credentials.
- Parallelism for ResearchAgent → CitationVerifier → SummaryAgent pairs is managed via `asyncio.gather`.
- If no papers pass verification for a subtopic, the pipeline retries with rejection feedback (up to 5 attempts).
- PDF download is attempted via the Unpaywall DOI proxy (`doi.org`) and full-text is extracted with `PyMuPDF` (fitz); if unavailable, a fallback sentence is used.
- **Aggregated table** includes columns for **Methodology**, **Dataset** (with download links for public datasets), and **Code** (with repository links when available).
- **Dataset link enrichment:** After the aggregated table is built, the pipeline automatically detects public datasets missing download URLs and uses web search to find and insert them.
- **PDF wide-table handling:** If the aggregated table has more than 5 columns, it is automatically converted to a per-row bullet list before PDF generation to ensure all columns are readable.
- **Knowledge base generation:** After all pipeline steps, a comprehensive `knowledge_base.md` is built containing topics, subtopics, article corpus, summaries, related work, novelties, and bibliography — ready for the chatbot interface.
- **Chat save enriches KB:** When a chat session is saved, its Q&A is appended to `knowledge_base.md`, so the agent's context grows with every conversation.
- **Novelty proposals** include a summary table plus a detailed discussion section per novelty: methodology, dataset(s) with links, baselines & comparisons, evaluation metrics, and an implementation roadmap with expected challenges and mitigation strategies.
- PDF output requires `markdown-pdf` (optional; skipped if not installed).
- All agents in `.opencode/agents/*.md` use `mode: all`.
