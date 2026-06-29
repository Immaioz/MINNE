# Scientific Research Results

**Topic:** Test topic

## Aggregated Table

| # | Title | Key Results | Journal | Subtopic |
|---|-------|-------------|---------|----------|
| 1 | Quantifying large language model usage in scientific papers | LLM-modified content in academic papers increased steadily from 2020–2024, with computer science showing up to 22% estimated usage and mathematics up to 9%. | Nature Human Behaviour (Q1) | Test topic |

## Related Work

The advent of large language models (LLMs) has introduced transformative capabilities for text generation, raising important questions about their penetration into academic writing. Liang et al. [1] conducted the first large-scale, population-level analysis to quantify this phenomenon, examining word frequency shifts across over 1.1 million preprints and published papers from arXiv, bioRxiv, and Nature portfolio journals spanning January 2020 to September 2024. Their findings reveal a consistent and steady increase in LLM-modified content over this period, with computer science papers exhibiting the most pronounced estimated usage—up to 22%—while mathematics and Nature portfolio papers showed more modest estimates of up to 9%. The study further identifies that higher LLM modification estimates are associated with authors who post preprints more frequently, research areas with greater crowding, and shorter papers, painting a nuanced picture of how and where LLM-assisted writing is being adopted.

Despite the valuable empirical foundation established by this work, several open questions remain. The population-level framework employed by Liang et al. [1] detects aggregate statistical shifts rather than attributing usage to individual authors or specific sections of a manuscript, leaving finer-grained patterns of LLM integration unexplored. Additionally, the study primarily covers English-language preprints and papers from a subset of repositories and journals; the extent of LLM usage in non-English academic writing, in books and monographs, or in fields such as the humanities and social sciences has yet to be systematically assessed. Future research would benefit from developing methods to distinguish between LLM-assisted editing for language polishing versus deeper contributions to content generation, and from investigating how editorial policies and journal guidelines are evolving in response to this growing trend.

## Proposed Novelties

| # | Novelty | Description | Difficulty | Rationale |
|---|---------|-------------|------------|-----------|
| 1 | **Section-level LLM fingerprinting** | Develop a method that partitions manuscripts into canonical sections (Introduction, Methods, Results, Discussion) and computes per-section vocabulary shift scores, identifying which parts of a paper are most affected by LLM-assisted writing. | ★★ Medium | [1] detects population-level word frequency shifts for entire papers but cannot localize LLM usage within a manuscript; this fills the gap of finer-grained patterns of LLM integration. |
| 2 | **Cross-lingual LLM usage estimation** | Extend the word-frequency-shift framework to non-English academic corpora (e.g., Spanish, Chinese, French) drawn from multilingual preprint servers or regional journals, producing the first estimates of LLM penetration in non-English scientific publishing. | ★★ Medium | [1] is limited to English-language preprints and papers, leaving the global extent of LLM use in non-English academic writing systematically unassessed. |
| 3 | **Polishing vs. generation classifier** | Build a two-stage pipeline that flags LLM-modified sentences via lexical markers and then classifies each flag as either surface-level editing (grammar, phrasing) or substantive content generation based on syntactic complexity and semantic novelty relative to prior author writing. | ★★★ Hard | [1] explicitly identifies the inability to distinguish between language polishing and deeper content generation as an open question; this novelty directly targets that gap. |
| 4 | **Author-level stylometric attribution** | Construct per-author writing profiles from their pre-2020 publication history and measure subsequent stylistic deviation to infer individual-level LLM adoption, moving beyond the aggregate trends reported in [1]. | ★★★ Hard | [1] acknowledges that its population-level framework cannot attribute usage to individual authors; this novelty enables personalized impact studies and accountability. |
| 5 | **Editorial policy impact analysis** | Use interrupted time-series analysis on the corpus of [1] to compare LLM-modified content estimates before and after major journals/publishers announced specific LLM disclosure policies (e.g., Nature, Elsevier), providing the first empirical evidence on whether such policies curb or shape LLM usage. | ★★ Medium | [1] notes that editorial policy responses are still evolving but does not assess their effectiveness; this novelty directly tests whether policies have a measurable impact on LLM adoption in academic writing. |

## References

[1] W. Liang, Y. Zhang, Z. Wu, H. Lepp, W. Ji, X. Zhao, et al., "Quantifying large language model usage in scientific papers," *Nature Human Behaviour*, vol. 9, no. 12, pp. 2599–2609, 2025, doi: 10.1038/s41562-025-02273-8.
