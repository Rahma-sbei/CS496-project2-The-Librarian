# The Librarian

Ask a question about **Streamlit** and get an answer built only from Streamlit's documentation, closed GitHub issues
and release notes. Every claim cites the passage it comes from; when the passages don't contain the answer, it says
"I don't know".

CS496 AI Engineering, Project 2 (retrieval-augmented generation), Mediterranean Institute of Technology, Fall 2026.
Team: Nadia Ben Njima, Ahmed Brini, Rahma Sbei, Mariem Lahmer. Report: [report.pdf](report.pdf).

**Try it live: <https://the-librarian-6wzd.onrender.com>** (free hosting: after 15 minutes without visitors the first
page load takes about a minute while the service wakes up).

## Setup (one command)

Python 3.12. From a clean clone:

```bash
python -m venv venv; venv/Scripts/pip install -r requirements.txt
```

(works in PowerShell and bash; on macOS/Linux use `venv/bin/pip`). Then copy `.env.example` to `.env` and fill in
`OPENAI_API_KEY`, plus `QDRANT_URL` and `QDRANT_API_KEY` for the team's Qdrant Cloud collection, which already holds
all 3,562 chunks.

No Qdrant credentials? Leave `QDRANT_URL` empty and build a local copy once (embedded Qdrant in `.qdrant/`, about
10 minutes on a laptop CPU; the corpus is already in the repository):

```bash
venv/Scripts/python index_corpus.py --recreate
```

## Run (one command)

```bash
venv/Scripts/python -m uvicorn app:app --port 8000
```

Open <http://localhost:8000>, type a question, and click a `[n]` citation to see the passage it came from.

Command-line version: `venv/Scripts/python generate.py "how do I keep state between reruns?"`.

## Deployment

The service is live at <https://the-librarian-6wzd.onrender.com>. It runs on Render's free plan from this
repository, configured by [render.yaml](render.yaml); the chunks live in Qdrant Cloud, and the embedding models run
through ONNX (`fastembed`, no PyTorch), so the app fits in the free plan's 512 MB. The free plan sleeps after 15
minutes without visitors; the first visit after that takes about a minute to wake it.

To deploy your own copy: on [render.com](https://render.com), choose **New → Blueprint**, connect this GitHub
repository, and enter `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` and `LIBRARIAN_ADMIN_TOKEN` when asked.

## Adding documents

The knowledge base can grow without rebuilding anything: a new document is chunked with the same rules, embedded and
upserted into Qdrant. Sending the same `doc_id` again replaces the earlier version.

From the web service (needs `LIBRARIAN_ADMIN_TOKEN`; without it the endpoint is disabled):

```bash
curl -X POST https://the-librarian-6wzd.onrender.com/api/documents -H "Authorization: Bearer <LIBRARIAN_ADMIN_TOKEN>" -H "Content-Type: application/json" -d '{"title": "Team FAQ", "type": "doc", "url": "https://example.com/faq", "text": "# Deploying
Use render.yaml ..."}'
```

`DELETE /api/documents/<doc_id>` removes a document. From the command line:
`venv/Scripts/python index_corpus.py --add notes.md --title "Team FAQ" --url https://example.com/faq`.

## Results

Measured on our evaluation set: 100 answerable questions with exact source passages, plus 12 questions the corpus
cannot answer (`data/eval_dataset.json`).

**Retrieval** (does the right passage come back?):

| Search              | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Search time (p50) |
| ------------------- | -------- | -------- | --------- | --------- | ----------------- |
| Keyword only (BM25) | 0.32     | 0.72     | 0.83      | 0.89      | 47 ms             |
| Vector only         | 0.47     | 0.70     | 0.82      | 0.85      | 55 ms             |
| **Hybrid (used)**   | 0.44     | 0.77     | **0.87**  | 0.93      | 60 ms             |
| Hybrid + reranker   | 0.44     | 0.80     | 0.88      | 0.93      | 3,526 ms          |

Search times include the round trip to Qdrant Cloud.

**Generation** (10 passages, `gpt-4o-mini`, graded by an independent `gpt-4.1-mini` judge): correct on **89.7%** of all
112 questions; **98.3%** of claims supported by a retrieved passage; the cited passage states the claim **98.8%** of the
time; "I don't know" on **100%** of unanswerable questions and 4% of answerable ones.

**Service**: $0.00057 per question, 1.2 s median end to end (p95 2.5 s). Sending **10 passages instead of 5** is the
deployed default: it raises accuracy +4.4 points (85.3% → 89.7%) for a fraction of a cent more, because the right
passage reaches the model 87% of the time instead of 77%. The Week 8 cost experiment (2 passages) cut cost 53% but lost
about 9 points of accuracy, so it was not adopted (see [optimisation.md](optimisation.md)).

Full numbers and discussion: [report.pdf](report.pdf). Raw per-question results: [results/](results/).

## Reproduce the evaluation

```bash
venv/Scripts/python eval/validate_eval.py                  # the dataset matches the chunks in Qdrant
venv/Scripts/python eval/run_eval.py --tag final           # retrieval ablation + generation (calls OpenAI, about $0.10)
venv/Scripts/python eval/run_eval.py --skip-generation     # retrieval only, free
venv/Scripts/python eval/generation_eval.py --k 2 --tag opt-k2   # the Week 8 configuration
```

Rebuild everything from scratch (needs a GitHub token in `.env`, 20 to 40 minutes):

```bash
venv/Scripts/python build_corpus.py --max-issues 600 && venv/Scripts/python index_corpus.py --recreate
```

## What's in the repository

| Path                                     | What it is                                                                                                                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_corpus.py`                        | Downloads the docs, closed issues and releases, removes personal data → `corpus/`                                                                                               |
| `index_corpus.py`                        | Cleans and chunks documents, embeds them (dense + BM25) and writes them to Qdrant; `add_document()` adds new content                                                              |
| `store.py`                               | The only code that talks to Qdrant: collection setup, upserts, deletes, hybrid search                                                                                          |
| `generate.py`                            | Hybrid search + `gpt-4o-mini` answer with citations; usable from the command line                                                                                               |
| `app.py`, `static/`                      | The web service (FastAPI) and the chat page                                                                                                                                     |
| `data/eval_dataset.json`                 | 100 questions with answers and exact source passages, plus 12 unanswerable questions                                                                                            |
| `eval/`                                  | `retrieval_eval.py` (recall@k + ablation), `generation_eval.py` (faithfulness, correctness, citations, latency, cost), `run_eval.py` (both), `validate_eval.py` (dataset check) |
| `results/`                               | Evaluation outputs used in the report                                                                                                                                           |
| `design.md`                              | Chunking, index choice, retrieval, generation, known failure cases                                                                                                              |
| `optimisation.md`                        | Week 8: what we changed and what it saved, before/after                                                                                                                         |
| `postmortem.md`                          | What went wrong and what we learned                                                                                                                                             |
| `report.pdf`                             | The project report                                                                                                                                                              |
| `render.yaml`                            | Deployment on Render (free plan)                                                                                                                                                |

No API keys or personal data are in the repository: `.env` and `logs/` are ignored by git, and issue threads are
scrubbed of usernames, emails and tokens before they are saved.

## Contributions

| Member          | Contribution                                                                                                                                                                                                          |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rahma Sbei      | Project setup; corpus collection (`build_corpus.py`); cleaning, chunking and indexing (`index_corpus.py`); dataset validation and repair; evaluation scripts (`eval/`); design, optimisation and postmortem documents |
| Mariem Lahmer   | Answer generation with citations and "I don't know" (`generate.py`)                                                                                                                                                   |
| Nadia Ben Njima | Evaluation dataset: questions, answers and source passages (`data/`)                                                                                                                                                  |
| Ahmed Brini     | UI building (`app.py`, `static/`) and deployment (`render.yaml`); report creation (`report.pdf`)                                                                                           |
