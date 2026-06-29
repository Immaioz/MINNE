# Multi-Agent Scientific Research Pipeline

Sistema multi-agente basato su opencode per cercare, verificare e analizzare articoli scientifici Q1, con generazione di report Markdown e PDF.

---

## Architettura

```
OrchestratorAgent
    └── JournalAssignerAgent  (assegna un journal Q1 diverso a ogni sottotema)
          └── asyncio.gather (parallelo)
                ├── ResearchAgent(topic_1, journal_A)
                │     └── CitationVerifier (L1: arXiv API + L2: DOI/C Crossref)
                │           └── SummaryAgent(topic_1)
                ├── ResearchAgent(topic_2, journal_B)
                │     └── CitationVerifier (L1 + L2)
                │           └── SummaryAgent(topic_2)
                └── ResearchAgent(topic_N, journal_N)
                      └── CitationVerifier (L1 + L2)
                            └── SummaryAgent(topic_N)
  AggregatorAgent          → tabella Markdown
  RelatedWorksDraftAgent   → bozza narrativa con citazioni [N]
  NoveltyProposalAgent     → tabella novità con difficoltà
  BibliographyAgent        → bibliografia IEEE formattata
```

### Agenti

| Agente | Compito |
|--------|---------|
| `OrchestratorAgent` | Scompone il topic in N sottotemi distinti |
| `JournalAssignerAgent` | Assegna un journal Q1 diverso a ciascun sottotema |
| `ResearchAgent` | Trova N articoli recenti su un journal target specifico (con arXiv ID e DOI) |
| `CitationVerifier` | Verifica ogni articolo: L1 (arXiv API) + L2 (DOI resolution via Crossref) |
| `SummaryAgent` | Struttura i risultati in JSON |
| `AggregatorAgent` | Consolida tutti i JSON in una tabella Markdown |
| `RelatedWorksDraftAgent` | Scrive bozza narrativa dello stato dell'arte con citazioni [N] |
| `NoveltyProposalAgent` | Propone novità realizzabili con tabella e livello di difficoltà |
| `BibliographyAgent` | Formatta la bibliografia in stile IEEE |

### Verifica citazioni

Prima che un articolo passi al SummaryAgent, viene verificato su due livelli:

- **L1 — arXiv API**: cerca il paper su arXiv per arXiv ID (o per titolo se ID non disponibile) e conferma l'esistenza
- **L2 — DOI resolution**: risolve il DOI via Crossref API e confronta il titolo restituito con quello atteso (similarità ≥ 60%)

Se un articolo non supera **entrambi** i livelli, il ResearchAgent cerca un sostituto (fino a 5 tentativi).

---

## Setup & Utilizzo

### Requisiti

```bash
pip install flask httpx markdown markdown-pdf
```

### Pipeline CLI

```bash
python pipeline.py "Research topic" --subtopics 3 --articles 3
```

Argomenti:
- `topic` (posizionale) — topic di ricerca
- `--subtopics` — numero di sottotemi (default: 3)
- `--articles` — articoli per sottotema (default: 3)

### Webapp

```bash
python webapp.py
# → http://127.0.0.1:5050
```

Interfaccia browser per:
- Avviare la pipeline con un form (topic, sottotemi, articoli)
- Visualizzare in tempo reale lo stato di esecuzione
- Sfogliare tutti i risultati passati (Markdown renderizzato in HTML)
- Scaricare file raw (Markdown, PDF)

### Subagenti singoli (opencode TUI)

```
@orchestrator Decompose: <topic>
@journal-assigner Assign journals to: <subtopics>
@researcher Find articles about: <subtopic> in <journal>
@summarizer Analyse these articles: ...
@aggregator Combine these JSON arrays: ...
@related-works-draft Write related work from: ...
@novelty-proposal Propose novelties from: ...
@bibliography Format references from: ...
```

---

## Configurazione

Modello LLU configurato in `opencode.json`. Attualmente: `opencode/deepseek-v4-flash-free`.

I system prompt di ogni agente sono nei file `.opencode/agents/*.md`.

---

## Output

```
results/<topic-slug>/
├── aggregated_table.md    # tabella riassuntiva articoli
├── related_work.md        # bozza narrativa con citazioni [N]
├── novelties.md           # 3–5 novità con difficoltà (★–★★★)
├── bibliography.md        # riferimenti in stile IEEE
├── results.md             # report completo unificato
└── results.pdf            # report in formato PDF
```

---

## Note tecniche

- Ogni agente invoca `opencode run --agent <name> "prompt"` via subprocess.
- Il parallelismo delle coppie ResearchAgent → CitationVerifier → SummaryAgent è gestito da `asyncio.gather`.
- Se il ResearchAgent non produce articoli verificabili, il CitationVerifier lo riavvia con un prompt che esclude i titoli già falliti (fino a 5 tentativi).
- I modelli `github-copilot/*` NON funzionano via `opencode run` (solo in TUI). Usare modelli `opencode/*`.
- La conversione PDF usa `markdown-pdf` (PyMuPDF).
- Tutti gli agenti in `.opencode/agents/*.md` devono avere `mode: all`.
