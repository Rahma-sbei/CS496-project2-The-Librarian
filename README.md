# The Librarian

Ask a question about **Streamlit** and get an answer built only from Streamlit's documentation, closed GitHub issues
and release notes. Every claim cites the passage it comes from; when the passages don't contain the answer, it says
"I don't know".

CS496 AI Engineering, Project 2 (retrieval-augmented generation), Mediterranean Institute of Technology, Fall 2026.
Team: Nadia Ben Njima, Ahmed Brini, Rahma Sbei, Mariem Lahmer. Report: [report.pdf](report.pdf).

## Setup (one command)

Python 3.12. From a clean clone:

```bash
python -m venv venv; venv/Scripts/pip install -r requirements.txt
```

(works in PowerShell and bash; on macOS/Linux use `venv/bin/pip`). Then copy `.env.example` to `.env` and paste your `OPENAI_API_KEY`. The corpus
and the index are already in the repository, so nothing needs to be downloaded or rebuilt.

## Run (one command)

```bash
venv/Scripts/python -m uvicorn app:app --port 8000
```

Open <http://localhost:8000>, type a question, and click a `[n]` citation to see the passage it came from.

Command-line version: `venv/Scripts/python generate.py "how do I keep state between reruns?"`.

## Deployment

The service runs on Render's free plan from this repository, configured by [render.yaml](render.yaml): it installs
only [requirements-deploy.txt](requirements-deploy.txt) and embeds questions with `fastembed` (ONNX) instead of
PyTorch (`LIBRARIAN_EMBEDDER=fastembed`), so it needs about 290 MB instead of 640 MB. The vectors match PyTorch's
(cosine ≥ 0.99999) and the retrieval evaluation gives identical recall either way. The free plan sleeps after 15
minutes without visitors; the first visit after that takes about a minute to wake it.

To deploy your own copy: on [render.com](https://render.com), choose **New → Blueprint**, connect this GitHub
repository, and enter `OPENAI_API_KEY` when asked.

## Results

Measured on our evaluation set: 100 answerable questions with exact source passages, plus 12 questions the corpus
cannot answer (`data/eval_dataset.json`).

**Retrieval** (does the right passage come back?):

| Search | Recall@1 | Recall@5 | Recall@20 | Search time (p50) |
|---|---|---|---|---|
| Keyword only (BM25) | 0.29 | 0.69 | 0.89 | 13 ms |
| Vector only | 0.47 | 0.70 | 0.85 | 23 ms |
| **Hybrid (used)** | 0.38 | **0.80** | 0.90 | 41 ms |
| Hybrid + reranker | 0.43 | 0.79 | 0.93 | 2,098 ms |

**Generation** (5 passages, `gpt-4o-mini`, graded by an LLM judge): correct on **86.6%** of all 112 questions;
**98.5%** of claims supported by a retrieved passage; the cited passage states the claim **99.1%** of the time; "I don't
know" on **100%** of unanswerable questions and 5% of answerable ones.

**Service**: $0.000305 per question, 1.3 s median and 2.3 s p95 end to end. **Week 8**: sending 2 passages instead of 5
cuts the cost by 53% but loses 10.3 points of accuracy, so the service keeps 5 (see [optimisation.md](optimisation.md)).

Full numbers and discussion: [report.pdf](report.pdf). Raw per-question results: [results/](results/).

## Reproduce the evaluation

```bash
venv/Scripts/python eval/validate_eval.py                  # the dataset matches the index
venv/Scripts/python eval/run_eval.py --tag final           # retrieval ablation + generation (calls OpenAI, about $0.10)
venv/Scripts/python eval/run_eval.py --skip-generation     # retrieval only, free
venv/Scripts/python eval/generation_eval.py --k 2 --tag opt-k2   # the Week 8 configuration
```

Rebuild everything from scratch (needs a GitHub token in `.env`, 10 to 30 minutes):

```bash
venv/Scripts/python build_corpus.py --max-issues 600 && venv/Scripts/python index_corpus.py
```

## What's in the repository

| Path | What it is |
|---|---|
| `build_corpus.py` | Downloads the docs, closed issues and releases, removes personal data → `corpus/` |
| `index_corpus.py` | Cleans and chunks the corpus, builds vectors and the BM25 keyword index → `index/` |
| `generate.py` | Hybrid search + `gpt-4o-mini` answer with citations; usable from the command line |
| `app.py`, `static/` | The web service (FastAPI) and the chat page |
| `data/eval_dataset.json` | 100 questions with answers and exact source passages, plus 12 unanswerable questions |
| `eval/` | `retrieval_eval.py` (recall@k + ablation), `generation_eval.py` (faithfulness, correctness, citations, latency, cost), `run_eval.py` (both), `validate_eval.py` (dataset check) |
| `results/` | Evaluation outputs used in the report |
| `design.md` | Chunking, index choice, retrieval, generation, known failure cases |
| `optimisation.md` | Week 8: what we changed and what it saved, before/after |
| `postmortem.md` | What went wrong and what we learned |
| `report.pdf` | The project report |
| `render.yaml`, `requirements-deploy.txt` | Deployment on Render (free plan, no PyTorch) |

No API keys or personal data are in the repository: `.env` and `logs/` are ignored by git, and issue threads are
scrubbed of usernames, emails and tokens before they are saved.

## Contributions

| Member | Contribution |
|---|---|
| Rahma Sbei | Project setup; corpus collection (`build_corpus.py`); cleaning, chunking and indexing (`index_corpus.py`); dataset validation and repair; evaluation scripts (`eval/`); design, optimisation and postmortem documents |
| Mariem Lahmer | Answer generation with citations and "I don't know" (`generate.py`) |
| Nadia Ben Njima | Evaluation dataset: questions, answers and source passages (`data/`) |
| Ahmed Brini | UI building (`app.py`, `static/`) and deployment (`render.yaml`, `requirements-deploy.txt`); report creation (`report.pdf`) |

Most scripts were drafted with an AI coding assistant, then run, checked and corrected by the team on our own data
(see [postmortem.md](postmortem.md)). Every member reviewed the code so that each of us can explain every line.
