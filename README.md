# Multi-Agent Scientific Research Pipeline

Sistema multi-agente opencode per cercare, analizzare e proporre novità da articoli scientifici Q1.

---

## Architettura degli agenti

```
                    ┌─────────────────────┐
                    │  OrchestratorAgent  │  scompone il topic in N sottotemi
                    └──────────┬──────────┘
                               │  asyncio.gather (parallelo)
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
  │ ResearchAgent │    │ ResearchAgent │    │ ResearchAgent │
  │  (subtopic 1) │    │  (subtopic 2) │    │  (subtopic N) │
  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
          │ testo articoli     │                    │
          ▼                    ▼                    ▼
  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
  │ SummaryAgent  │    │ SummaryAgent  │    │ SummaryAgent  │
  │  (subtopic 1) │    │  (subtopic 2) │    │  (subtopic N) │
  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
          │ JSON               │ JSON               │ JSON
          └────────────────────┼────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  AggregatorAgent    │  tabella Markdown
                    └──────────┬──────────┘
                               │ tabella + articoli grezzi
                               ▼
                    ┌─────────────────────┐
                    │ RelatedWorksDraft-  │  bozza narrativa
                    │     Agent           │  con citazioni [N]
                    └──────────┬──────────┘
                               │ bozza + corpus articoli
                    ┌──────────┴──────────┐
                    ▼                     ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │ NoveltyProposalAgent │  │  BibliographyAgent   │
    │ tabella novità +     │  │  bibliografia IEEE   │
    │ livello difficoltà   │  │                      │
    └──────────────────────┘  └──────────────────────┘
```

### Ruolo di ogni agente

| Agente | Compito |
|--------|---------|
| `OrchestratorAgent` | Scompone il topic in N sottotemi distinti |
| `ResearchAgent` | Trova 3 articoli Q1 recenti per sottotema (knowledge interna) |
| `SummaryAgent` | Struttura i risultati del ResearchAgent in JSON |
| `AggregatorAgent` | Consolida tutti i JSON in una tabella Markdown |
| `RelatedWorksDraftAgent` | Scrive bozza narrativa dello stato dell'arte con citazioni [N] |
| `NoveltyProposalAgent` | Propone novità realizzabili con tabella e livello di difficoltà |
| `BibliographyAgent` | Formatta la bibliografia in stile IEEE |

> Tutti gli agenti usano chiamate LLM dirette (nessuna dipendenza esterna oltre a Python stdlib).

---

## Setup & Utilizzo

### Pipeline completa (consigliata)

```bash
# 1. Assicurati di avere opencode installato e autenticato
opencode --version
opencode auth login   # configura un provider (es. Anthropic, OpenAI)

# 2. (Opzionale) Crea .env con la API key di opencode
cp .env.example .env
# modifica .env → OPENCODE_API_KEY=sk-...

# 3. Avvia la pipeline end-to-end
python pipeline.py
```

La pipeline:
1. Scompone il topic in sottotemi
2. Cerca articoli Q1 in parallelo
3. Genera tabella → bozza related works → novità → bibliografia
4. **Salva tutto in `results/<topic-slug>/`** (un file per output)

### Subagenti singoli (opencode)

In una sessione opencode, invoca qualsiasi agente singolarmente:

- `@orchestrator Decompose: <topic>`
- `@researcher Find articles about: <subtopic>`
- `@summarizer Analyse these articles: ...`
- `@aggregator Combine these JSON arrays: ...`
- `@related-works-draft Write related work from: ...`
- `@novelty-proposal Propose novelties from: ...`
- `@bibliography Format references from: ...`

---

## Configurazione

All'inizio di `pipeline.py`:

```python
TOPIC = "Transformer-based architectures for speech and audio processing"
NUM_SUBTOPICS = 3   # numero di coppie ResearchAgent+SummaryAgent in parallelo
```

---

## Output

La pipeline stampa tutto nel terminale e salva i file in `results/<topic-slug>/`:

```
results/<topic-slug>/
├── aggregated_table.md    # tabella articoli
├── related_work.md        # bozza narrativa con citazioni [N]
├── novelties.md           # 3–5 novità con difficoltà (★–★★★)
├── bibliography.md        # riferimenti IEEE
└── results.md             # report completo unificato
```

---

## Note tecniche

- Ogni agente invoca `opencode run --agent <name> "prompt"` via subprocess.
- Le chiamate LLM sono gestite interamente da opencode — usa il modello/provider configurato.
- I system prompt sono definiti nei file `.opencode/agents/*.md`.
- Il parallelismo delle coppie ResearchAgent+SummaryAgent è gestito da `asyncio.gather`.
- Gli agenti finali (AggregatorAgent, RelatedWorksDraftAgent, NoveltyProposalAgent, BibliographyAgent) vengono eseguiti in sequenza.
- I subagenti sono anche invocabili manualmente in una sessione opencode via `@agentname`.
- **Zero dipendenze Python** oltre alla stdlib (python-dotenv è opzionale).
