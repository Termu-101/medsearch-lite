# MedSearch Lite

An AI-powered clinical literature research assistant that searches 500 PubMed abstracts on diabetes management and answers evidence-based research questions with inline citations, evidence-level labels, and automated fact-checking.

**This is a research literature tool — not a medical advice tool.** It is explicitly designed to refuse personal health questions and only answer general research queries, always citing its sources.

## Overview

MedSearch Lite is a Retrieval-Augmented Generation (RAG) system built to demonstrate production-grade RAG engineering: hybrid retrieval, domain-specific embeddings, entity-aware reranking, self-verifying generation, and a full evaluation suite with ablation testing.

A user asks a clinical research question (e.g. *"What is the effect of SGLT2 inhibitors on kidney disease progression?"*), and the system:
1. Filters out personal medical advice requests
2. Retrieves relevant evidence from a corpus of PubMed abstracts using multiple search strategies combined
3. Reranks and verifies that evidence for quality and relevance
4. Generates a hedged, cited answer
5. Fact-checks every claim in that answer against the retrieved sources before returning it

## Architecture

```
User Query
  │
  ├─ Step A: Personal advice guard (LLM classifier) → refuse if personal
  │
  ├─ Step B: Medical synonym expansion (for keyword search)
  ├─ Step C: HyDE — generate a hypothetical PubMed abstract excerpt
  │
  ├─ Step D: Hybrid retrieval
  │     ├─ Dense search (BiomedBERT embeddings via Chroma)
  │     └─ Keyword search (BM25) → merged via Reciprocal Rank Fusion
  │
  ├─ Step E: Entity boosting (SciSpacy NER — boosts chunks matching
  │           specific drugs/conditions in the query)
  ├─ Step F: Cross-encoder reranking, weighted by evidence level
  ├─ Step G: Parent abstract lookup + deduplication
  │
  ├─ Step H: CRAG confidence check (LLM grades retrieved evidence)
  ├─ Step I: Answer generation (cited, hedged, evidence-only)
  ├─ Step J: Groundedness verification (fact-checks every claim,
  │           regenerates if unsupported)
  │
  └─ Step K: Mandatory disclaimer appended in code
```

## Key engineering decisions

- **Parent-child chunking with propositions** — each abstract is split into sentence-level child chunks, each rewritten by an LLM into a clean "proposition" for better embedding quality, while the original sentence and full parent abstract are preserved for citation and context.
- **Dual embedding models** — BiomedBERT (domain-specific) and MiniLM (general-purpose) are both indexed, enabling direct ablation comparison of domain-specific vs. general embeddings.
- **Entity-aware boosting with rarity filtering** — common entities (e.g. "diabetes" in a diabetes-focused corpus) are excluded from boosting since they carry no discriminating signal; only specific, rare entities (e.g. a named drug) trigger a score boost.
- **Groundedness loop** — every generated answer is independently fact-checked sentence-by-sentence against the source abstracts, with up to 2 regeneration attempts before falling back to an honest refusal.

## Evaluation

Evaluated on a 30-question golden dataset (24 generated from the corpus itself + 6 external questions from PubMedQA/MedMCQA as an out-of-domain stress test):

| Metric | Score |
|---|---|
| Recall@5 (overall) | 0.733 |
| Recall@5 (in-corpus questions) | 0.917 |
| MRR (overall) | 0.700 |
| Faithfulness | ~0.99 |
| Answer Relevance | 1.00 |
| Hallucination Rate | <2% |

A 5-configuration ablation study (naive dense search → full hybrid + entity boost + reranking pipeline) is included in `evaluation/run_evaluation.py` to quantify the contribution of each pipeline component.

## Tech stack

| Component | Tool |
|---|---|
| Data source | Biopython Entrez (PubMed API) |
| Embeddings | BiomedBERT, all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Medical NER | SciSpacy (en_ner_bc5cdr_md) |
| LLMs | Groq (Llama/GPT-OSS/Qwen models) |
| Frontend | Streamlit |

## Project structure

```
medsearch-lite/
├── data/
│   ├── raw/          # Raw PubMed abstracts (generated, gitignored)
│   ├── processed/     # Chunked + enriched data (generated, gitignored)
│   └── indexes/        # Chroma/BM25 indexes (generated, gitignored)
├── src/
│   ├── data_collection.py   # Phase 1: PubMed fetching, evidence tagging
│   ├── chunking.py          # Phase 2: Parent-child chunking, propositions
│   ├── enrichment.py        # Phase 3: SciSpacy entity extraction
│   ├── indexing.py          # Phase 4: Embedding + BM25 index building
│   ├── retrieval.py         # Phase 5: HyDE, hybrid retrieval, reranking
│   ├── generation.py        # Phase 5: CRAG, generation, groundedness
│   ├── safety.py            # Personal advice guard, disclaimer
│   ├── synonyms.py          # Medical synonym expansion dictionary
│   └── query_pipeline.py    # Orchestrates the full pipeline
├── evaluation/
│   ├── golden_dataset.json  # 30-question evaluation set
│   └── run_evaluation.py    # Retrieval/generation metrics + ablation study
└── app/
    └── streamlit_app.py     # Frontend
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. Install the SciSpacy models (not on PyPI, installed via direct wheel):
   ```
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
   ```

3. Copy `.env.example` to `.env` and add your API keys (NCBI, Groq).

4. Run the pipeline phases in order:
   ```
   python -m src.data_collection
   python -m src.chunking
   python -m src.enrichment
   python -m src.indexing
   ```

5. Launch the app:
   ```
   streamlit run app/streamlit_app.py
   ```

## Known limitations

- **Corpus scope**: the system is built on 499 diabetes-focused abstracts. Questions outside this corpus's coverage (confirmed via the external evaluation subset) are correctly refused rather than hallucinated, but Recall@5 on out-of-domain questions is near 0 — a documented, expected limitation of a small single-topic corpus, not a retrieval bug.
- **HyDE non-determinism**: since HyDE generation has no fixed temperature, identical queries can occasionally retrieve different candidate sets run-to-run.
- **Free-tier LLM rate limits**: evaluation runs are subject to daily token limits on the free Groq tier, which can interrupt large batch evaluations.

## Disclaimer

This tool is intended for literature research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
