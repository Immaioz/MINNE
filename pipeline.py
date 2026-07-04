"""
Multi-Agent Scientific Research Pipeline.

The main orchestration stays here; agents, verifier, and markdown utilities are
split into dedicated modules for maintainability.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from pipeline_agents import (
    AggregatorAgent,
    BibliographyAgent,
    NoveltyProposalAgent,
    OrchestratorAgent,
    RelatedWorksDraftAgent,
    ResearchAgent,
    SummaryAgent,
    ensure_opencode_api_key_from_env,
)
from pipeline_domain import (
    PipelineResult,
    SubtopicResult,
    build_articles_corpus,
    parse_articles,
    topic_slug,
)
from pipeline_markdown import enrich_dataset_links, fix_wide_tables
from pipeline_verifier import CitationVerifier


async def run_subtopic_pipeline(
    subtopic: str, results_dir: Path, num_articles: int = 3,
) -> SubtopicResult:
    """
    Esegue ResearchAgent → salva articoli proposti (.md) → CitationVerifier (DOI) → SummaryAgent.
    L'intera catena viene lanciata in parallelo con le altre tramite asyncio.gather.
    """
    print(f"  🔍  [{subtopic[:50]}…] ricerca su journal Q1 multipli")

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
                print(f"  ⚠️   [{subtopic[:50]}…] Researcher non ha prodotto articoli validi (tentativo {attempt+1})")
                continue

            if attempt == 0:
                subtopic_slug = re.sub(r"[^a-z0-9]+", "-", subtopic.lower()).strip("-")[:40]
                proposed_path = proposed_dir / f"{subtopic_slug}.md"
                header = f"# Articoli proposti: {subtopic}\n\n"
                proposed_path.write_text(header + raw, encoding="utf-8")
                print(f"  📄  articoli proposti salvati → {proposed_path}")

            for art in articles:
                if art.title in used_titles:
                    continue
                ok, msg = await verifier.verify_article(art.title, art.doi, art.arxiv_id)
                if ok:
                    print(f"  📥  download PDF: {art.title[:55]}…")
                    full_text = await verifier.fetch_full_text(art.arxiv_id)
                    if full_text:
                        word_count = len(full_text.split())
                        enriched = (
                            f"{art.raw}\n"
                            f"- Full Text (from arXiv PDF, {word_count} words):\n"
                            f"{full_text}"
                        )
                        print(f"  ✓   PDF estratto ({word_count} parole)")
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
                            print("  ⚠️   PDF non disponibile, usato abstract arXiv")
                        else:
                            enriched = art.raw
                            print("  ⚠️   nessun testo alternativo disponibile")

                    verified_blocks.append(enriched)
                    used_titles.add(art.title)
                    print(f"  ✓   verificato: {art.title[:55]}…")
                else:
                    print(f"  ✗   non verificato: {art.title[:60]}… — {msg}")

        if len(verified_blocks) < num_articles:
            print(f"  ⚠️   [{subtopic[:50]}…] solo {len(verified_blocks)}/{num_articles} articoli verificati")

        articles_raw = "\n\n".join(
            f"### Article {i+1}\n{block}"
            for i, block in enumerate(verified_blocks[:num_articles])
        )
        if not articles_raw:
            articles_raw = "[No verified articles found]"
        print(f"  ✓   [{subtopic[:50]}…] {len(verified_blocks)} articoli verificati")

        summarizer = SummaryAgent()
        summary_json = await summarizer.a_run(
            f"Analyse and summarise these articles:\n\n{articles_raw}",
        )
        print(f"  ✓   [{subtopic[:50]}…] analisi completata")
    finally:
        await verifier.close()

    return SubtopicResult(
        subtopic=subtopic,
        articles_raw=articles_raw,
        summary_json=summary_json,
    )


async def run_pipeline(topic: str, num_subtopics: int = 3, num_articles: int = 3) -> PipelineResult:
    """
    1. OrchestratorAgent       → scompone in num_subtopics sottotemi
    2. [ResearchAgent → proposte (.md) → CitationVerifier(DOI) → SummaryAgent] × N  → asyncio.gather
    3. AggregatorAgent         → tabella Markdown
    4. RelatedWorksDraftAgent  → bozza narrativa con citazioni
    5. NoveltyProposalAgent    → tabella novità con difficoltà
    6. BibliographyAgent       → bibliografia IEEE formattata
    """
    print(f"\n{'═' * 62}")
    print(f"  TOPIC: {topic}")
    print(f"  Sottotemi: {num_subtopics}")
    print(f"  Articoli per sottotema: {num_articles}")
    print(f"{'═' * 62}\n")

    results_dir = Path(__file__).parent / "results" / topic_slug(topic)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("📋  Orchestratore: scomposizione del topic…")
    orchestrator = OrchestratorAgent()
    raw_subtopics = await orchestrator.a_run(
        f"Decompose into exactly {num_subtopics} subtopics: {topic}",
    )

    subtopics = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in raw_subtopics.strip().splitlines()
        if re.match(r"^\d+[\.\)]", line.strip())
    ][:num_subtopics]

    if not subtopics:
        subtopics = [l.strip() for l in raw_subtopics.splitlines() if l.strip()][:num_subtopics]

    print(f"  Sottotemi:\n" + "\n".join(f"    {i+1}. {st}" for i, st in enumerate(subtopics)))

    print(f"\n🤖  Avvio di {len(subtopics)} coppie ResearchAgent+SummaryAgent in parallelo…\n")
    results: list[SubtopicResult] = await asyncio.gather(
        *[
            run_subtopic_pipeline(st, results_dir, num_articles=num_articles)
            for st in subtopics
        ]
    )

    print("\n📊  Aggregatore: composizione tabella finale…")
    combined_input = "\n\n".join(
        f"=== Subtopic: {r.subtopic} ===\n{r.summary_json}"
        for r in results
    )
    aggregator = AggregatorAgent()
    table = await aggregator.a_run(combined_input)

    print("\n✍️   RelatedWorksDraftAgent: scrittura bozza narrativa…")
    corpus = build_articles_corpus(results)
    draft_input = (
        f"=== ARTICLES CORPUS ===\n{corpus}\n\n"
        f"=== AGGREGATED TABLE ===\n{table}"
    )
    draft_agent = RelatedWorksDraftAgent()
    draft = await draft_agent.a_run(draft_input)

    print("\n💡  NoveltyProposalAgent: proposta novità…")
    novelty_input = (
        f"=== RELATED WORK DRAFT ===\n{draft}\n\n"
        f"=== ARTICLES CORPUS ===\n{corpus}"
    )
    novelty_agent = NoveltyProposalAgent()
    novelties = await novelty_agent.a_run(novelty_input)

    print("📚  BibliographyAgent: formattazione bibliografia…")
    bib_input = (
        f"=== DRAFT ===\n{draft}\n\n"
        f"=== ARTICLES CORPUS ===\n{corpus}"
    )
    bib_agent = BibliographyAgent()
    bibliography = await bib_agent.a_run(bib_input)

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

    print("✅  Pipeline completata.\n")
    return PipelineResult(
        table=table,
        draft=draft,
        novelties=novelties,
        bibliography=bibliography,
        knowledge_base=knowledge_base,
        subtopic_results=results,
    )


async def main() -> None:
    # Explicitly prefer OPENCODE_API_KEY from .env over any logged session key.
    ensure_opencode_api_key_from_env()

    parser = argparse.ArgumentParser(
        description="Scientific Research Pipeline — multi-agent literature survey",
    )
    parser.add_argument(
        "topic",
        type=str,
        help="Research topic (e.g. 'Deep learning for medical imaging')",
    )
    parser.add_argument(
        "--subtopics",
        type=int,
        default=3,
        help="Number of parallel subtopics (default: 3)",
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=3,
        help="Number of articles to find per subtopic (default: 3)",
    )
    args = parser.parse_args()

    result = await run_pipeline(
        args.topic, num_subtopics=args.subtopics, num_articles=args.articles,
    )

    table = await enrich_dataset_links(result.table)
    knowledge_base = result.knowledge_base.replace(result.table, table)

    sep = "─" * 62
    print(sep)
    print("RISULTATI FINALI")
    print(sep)

    print("\n📊  TABELLA AGGREGATA\n")
    print(table)

    print("\n✍️  BOZZA RELATED WORKS\n")
    print(result.draft)

    print("\n💡  NOVITÀ PROPOSTE\n")
    print(result.novelties)

    print("\n📚  BIBLIOGRAFIA\n")
    print(result.bibliography)
    print(sep)

    out_dir = Path(__file__).parent / "results" / topic_slug(args.topic)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "aggregated_table.md": table,
        "related_work.md": result.draft,
        "novelties.md": result.novelties,
        "bibliography.md": result.bibliography,
        "knowledge_base.md": knowledge_base,
    }
    for name, content in files.items():
        (out_dir / name).write_text(content.strip() + "\n", encoding="utf-8")

    combined_md = (
        f"# Scientific Research Results\n\n"
        f"**Topic:** {args.topic}\n\n"
        f"## Aggregated Table\n\n{table}\n\n"
        f"## Related Work\n\n{result.draft}\n\n"
        f"## Proposed Novelties\n\n{result.novelties}\n\n"
        f"## References\n\n{result.bibliography}\n"
    )
    combined = out_dir / "results.md"
    combined.write_text(combined_md, encoding="utf-8")

    print(f"\n💾  Risultati salvati in: {out_dir}/")
    for name in files:
        print(f"      {name}")

    try:
        from markdown_pdf import MarkdownPdf, Section

        pdf_path = out_dir / "results.pdf"
        pdf_md = fix_wide_tables(combined_md, max_cols=5)

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
