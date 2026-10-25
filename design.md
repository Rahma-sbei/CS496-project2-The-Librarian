# Design: The Librarian

A question-answering service over the Streamlit open-source project. It answers only from Streamlit's documentation,
closed GitHub issues and release notes, cites the passage behind every claim, and says "I don't know" when the
passages don't contain the answer.

```
OFFLINE (once)                                           ONLINE (every question)
build_corpus.py  -> corpus/  (docs, issues, releases)    question --(optional rewrite)--> hybrid search
index_corpus.py  -> index/   (chunks, vectors, BM25)        BM25 top 50 + vector top 50, fused (RRF), filtered by type
                                                            -> top 5 chunks -> gpt-4o-mini with numbered sources
                                                            -> answer with [n] citations, or "I don't know"
```

## 1. Corpus

| Source | Where from | Size | Why |
|---|---|---|---|
| Documentation | `git clone github.com/streamlit/docs`, Markdown in `content/` | 404 pages, 1,654 chunks | Authoritative, structured by headings |
| GitHub issues | GitHub API, `streamlit/streamlit`, closed issues with at least one comment | 418 issues (closed 2019-10 to 2026-10), 1,731 chunks | Real user wording, error messages, maintainers' fixes and workarounds |
| Releases | GitHub Releases API, latest 60 | 60 releases (2024-02 to 2026-10), 177 chunks | Version-specific facts a model cannot have memorised |

The corpus is a frozen snapshot taken on 2026-10-25 and committed to the repository (5 MB), so the evaluation's exact
quotes stay valid and graders don't need a GitHub token. `build_corpus.py` rebuilds it. Licence: Apache 2.0 (public).

**Personal data.** Issue threads are scrubbed before saving: emails, `@mentions`, API tokens and home-directory paths
become `[EMAIL]`, `[USER]`, `[TOKEN]`, `/[HOME]`. Author logins are never stored; each comment keeps only the author's
GitHub role (`MEMBER`, `CONTRIBUTOR`, `NONE`), which also tells the reader whether a maintainer wrote it.

## 2. Chunking

Each source is cut differently because the documents have different shapes:

- **Docs**: cut at headings first, then packed paragraph by paragraph up to the target size. Fenced code blocks are
  never split, so code stays runnable even if a chunk runs over. Website markup (MDX components, HTML tables, icons) is
  removed or turned into text (`clean_mdx`), and chunks under 40 words are merged into a neighbour (`merge_small`).
- **Issues**: the description and the discussion are chunked separately; each comment is prefixed with its author's
  role. Issues are not cleaned, because pasted code and logs are the content.
- **Releases**: cut at headings and cleaned like the docs.

Every chunk keeps its metadata: `doc_id`, `type` (doc / issue / release), `title`, `section`, `url`, `date`. The text that
gets embedded is `title > section` followed by the chunk, so a chunk carries its heading's context.

**Chosen setting: 250 words with a 40-word overlap** (3,562 chunks, 147 words on average). Why this size: a
docs section is typically 100 to 300 words, so 250 keeps most sections whole, and with the top 5 chunks the model
receives about 1,700 tokens, cheap for `gpt-4o-mini`. Cleaning and merging small chunks took the index from 4,007
noisy chunks to 3,562. The evaluation scripts can score any rebuilt index (`--index`), so comparing 150 and 400 words
is one command each; we did not finish that comparison (see the report).

## 3. Index: flat files

| Part | File | What it is |
|---|---|---|
| Chunks + metadata | `index/chunks.jsonl` | one JSON line per chunk |
| Vectors | `index/embeddings.npy` | 3,562 × 384 float32, normalised, from `BAAI/bge-small-en-v1.5` |
| Keyword index | `index/bm25.pkl` | `rank_bm25.BM25Okapi` over lower-cased word tokens |
| Settings | `index/config.json` | chunk size, overlap, model, chunk count, build time |

**Why flat files instead of a vector database or Postgres:** 3,562 vectors take 5 MB, so comparing the question with
every chunk (one matrix product) takes a few milliseconds. There is no server to run or pay for, the index ships inside
the app, and one script rebuilds it in about 5 minutes. The trade-offs we accept: filtering is done in Python, there are
no incremental updates (full rebuild), and the BM25 file is a pickle, which must only be loaded from our own repository.
We would switch to a vector database if the corpus grew about a hundred times.

**Embedding model:** `bge-small-en-v1.5` runs on a laptop CPU with no API key or cost, and is a strong small
retrieval model. It reads at most 512 tokens, so the end of very long chunks is not embedded (see failure cases).

## 4. Retrieval

1. *(optional)* **Rewrite** a short or vague question into a full search query with `gpt-4o-mini`.
2. **Keyword search** (BM25) and **vector search** each return their top 50 chunks. BM25 is strong on exact strings
   (error messages, function names, version numbers); vectors are strong on paraphrases ("stop my app from re-running"
   finds the Session State pages). Streamlit questions contain both kinds.
3. **Reciprocal rank fusion** (constant 60) merges the two lists: a chunk ranked high by both wins.
4. **Metadata filter**: the user can restrict the search to docs, issues or release notes.
5. The top 5 chunks go to the language model.

**Reranker: measured, not enabled.** We tested a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that rescores
the top 30 hybrid results. It improves the first result (recall@1 0.38 → 0.43) and recall@20, but not recall@5 (0.80 →
0.79), which is what the model receives, and it adds about 2 s per question on a CPU (41 ms → 2,098 ms). "Skipping the reranker" is also
one of the course's suggested Week 8 savings. The full ablation is in [report.pdf](report.pdf).

## 5. Generation

`gpt-4o-mini` at temperature 0 receives the numbered chunks (`[n] title > section` + text) and these rules: every
factual sentence ends with the number of its source, nothing outside the sources may be used, and if the sources don't
contain the answer it replies exactly "I don't know.". The code then reads the `[n]` markers, turns them into links, and
flags any number that doesn't exist.

**"I don't know" threshold.** `generate.py` can refuse before calling the model when the best vector similarity is
below `--min-score`. On our 12 unanswerable questions vs 100 answerable ones:

| min_score | unanswerable refused | answerable wrongly refused |
|---|---|---|
| 0.70 | 42% | 4% |
| 0.75 | 92% | 15% |

Refusing 15% of good questions is too costly, so the service uses **0** (no pre-filter) and lets the model decide from
the passages; the report shows how well it does that.

## 6. Service

`app.py` (FastAPI) loads the index and model once and serves the page in `static/`. Each request is logged with its
latency, tokens and cost (`logs/requests-*.jsonl`); the page's 👍/👎 buttons are saved to `logs/feedback-*.jsonl`. Because the URL will
be public and every question costs money, each visitor is limited to 10 questions a minute and the whole service to
300 a day. Repeated questions are answered from an in-memory cache at no cost. On a host whose disk is wiped on restart,
the logs can be pushed to a private Hugging Face dataset every 5 minutes (`HF_LOGS_REPO`).

**Deployment.** The service runs on Render's free plan (512 MB), set up by `render.yaml`. With PyTorch loaded only to
embed the question, the service needed about 640 MB, so the deployed version runs the same `bge-small-en-v1.5` model
through `fastembed` (ONNX): about 290 MB. We checked that this changes nothing: question vectors match PyTorch's with
cosine ≥ 0.99999 on all 112 evaluation questions, and the retrieval evaluation gives identical recall at every k.
The index itself is unchanged.

## 7. Known failure cases

- **API reference pages are empty in the corpus.** The docs site fills in each function's description at build time;
  our Markdown only has `<Autofunction function="streamlit.radio" />`. Questions about a function's parameters can only
  be answered if a concept page, tutorial or issue mentions them.
- **Very long chunks.** Code blocks are never split and pasted logs stay whole: 17 chunks exceed 400 words (the longest
  3,923). The embedding model reads only their first 512 tokens.
- **Tables** are flattened to `a | b |` rows, so column meaning can be lost.
- **Cross-references.** Links to other pages are kept as text, but the system cannot follow them; an answer split
  across two pages needs both to be retrieved.
- **Leftover markup.** The cleaner covers the MDX components we saw; others may remain.
- **Issues are noisy.** A thread can contain a wrong first answer corrected later, and issue chunks are not cleaned.
- **Dates.** Docs pages have no date, so "since version X" questions rely on the release notes, not on filtering.
- **Dotted names in keyword search.** BM25 splits `st.cache_data` into `st` and `cache_data`; it still matches, but the
  exact name is not one token.
- **Model knowledge leaking in.** Streamlit is popular, so the model may "know" an answer that isn't in the passages;
  the faithfulness measure in the report catches this.
- **Citations on every sentence are not guaranteed.** The model sometimes cites only some sentences; the report
  measures how often.
- **Scrubbing is regex-based**, a first line of defence rather than a guarantee.
