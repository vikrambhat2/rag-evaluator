# rag-evaluator

A from-scratch RAG evaluation system — no RAGAS, no OpenAI, fully local using Ollama.

Evaluates your RAG pipeline across four metrics using LLM-as-judge and outputs a structured terminal report with weak spot detection.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Ollama](https://img.shields.io/badge/ollama-llama3.2-black) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Metrics

| Metric | What It Catches |
|---|---|
| **Faithfulness** | LLM hallucinating beyond retrieved context |
| **Answer Relevance** | Response drifting off-topic |
| **Context Precision** | Retriever pulling noisy chunks |
| **Context Recall** | Relevant info existing but not retrieved |

---

## Stack

- **LLM + Judge** — `llama3.2` via Ollama
- **Embeddings** — `nomic-embed-text` via Ollama
- **Vector DB** — ChromaDB (local persistent)
- **Orchestration** — LangChain
- No API keys. No cloud.

---

## Quickstart

```bash
# Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# Install dependencies
pip install -r requirements.txt

# Ingest docs + auto-generate test set
python ingest.py

# Run evaluation
python run_eval.py
```

---

## Output

```
                    RAG Evaluation Report
┌──────────────────────────────────────┬─────────┬──────────┬──────────┬────────┐
│ Query                                │ Faithful│ Relevance│ Precision│ Recall │
├──────────────────────────────────────┼─────────┼──────────┼──────────┼────────┤
│ What are the main chunking strateg...│  0.92   │   0.88   │   0.75   │  0.81  │
│ What is the difference between BM2...│  0.61   │   0.87   │   0.55   │  0.69  │
├──────────────────────────────────────┼─────────┼──────────┼──────────┼────────┤
│ AVERAGE                              │  0.82   │   0.89   │   0.66   │  0.75  │
└──────────────────────────────────────┴─────────┴──────────┴──────────┴────────┘

⚠ Weak Spots Detected:
  → Context Precision average 0.66 is below 0.70
```

Full results saved to `eval_report.json`.

---

## Project Structure

```
rag-evaluator/
├── data/
│   ├── docs/              # Knowledge base (markdown)
│   └── test_set.json      # Auto-generated QA pairs
├── src/
│   ├── models/schemas.py  # Pydantic models
│   ├── rag/
│   │   ├── ingest.py
│   │   └── pipeline.py
│   └── evaluator/
│       ├── judge.py       # LLM-as-judge + metric prompts
│       ├── metrics.py     # 4 scorer functions
│       └── report.py      # Rich table + JSON export
├── ingest.py
└── run_eval.py
```

---

## Part of a Series

This repo accompanies a 3-part blog series on RAG evaluation:

- **Part 1** — Evaluating a RAG Pipeline *(this repo)*
- **Part 2** — Evaluating Agentic RAG *(coming soon)*
- **Part 3** — A/B Testing RAG Pipelines *(coming soon)*

---

*Built by [Vikram](https://medium.com/) · Read the full walkthrough on Medium*