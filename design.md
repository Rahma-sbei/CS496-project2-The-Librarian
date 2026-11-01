# Design: The Librarian

A question-answering service over the Streamlit open-source project. It answers only from Streamlit's documentation,
closed GitHub issues and release notes, cites the passage behind every claim, and says "I don't know" when the
passages don't contain the answer.

```
OFFLINE (once)                                           ONLINE (every question)
build_corpus.py  -> corpus/  (docs, issues, releases)    question --(optional rewrite)--> hybrid search
index_corpus.py  -> Qdrant   (chunks, dense + BM25 vectors) BM25 top 50 + vector top 50, fused (RRF), filtered by type
                                                            -> top 10 chunks -> gpt-4o-mini with numbered sources
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
docs section is typically 100 to 300 words, so 250 keeps most sections whole, and with the top 10 chunks the model
receives about 3,400 tokens, still cheap for `gpt-4o-mini`. Cleaning and merging small chunks took the index from 4,007
noisy chunks to 3,562. The chunk size is a setting of `index_corpus.py`: building a second collection with other sizes
(`--collection librarian_c150 --chunk-size 150 --recreate`) and scoring it (`--collection` on the eval scripts) is
one command each; we did not finish that comparison (see the report).

## 3. Index: the Qdrant vector database

All chunks live in one Qdrant collection (`librarian`, on Qdrant Cloud). Each chunk is one **point** with two vectors
and its text as payload:

| Part | In Qdrant | What it is |
|---|---|---|
| Meaning vector | named vector `dense` | 384 numbers from `BAAI/bge-small-en-v1.5`, cosine similarity |
| Keyword vector | named sparse vector `bm25` | word weights from `Qdrant/bm25`; Qdrant applies the word-rarity (IDF) part on the server |
| Chunk + metadata | payload | `doc_id`, `chunk` (its number in the document), `type`, `title`, `section`, `url`, `date`, `text` |
| Indexes | payload indexes on `type` and `doc_id` | fast filtering by source type and fast deletion of one document |

`store.py` is the only file that talks to Qdrant: it creates the collection, embeds and writes chunks, deletes a
document, searches, and lists all chunks for the evaluation.

**Why a vector database.** The first version kept everything in flat files loaded into memory: fine for 3,562 chunks,
but every new document meant rebuilding the whole index, the keyword index was a pickle file, and the app had to hold
everything in RAM. With Qdrant:

- **Adding content is incremental.** `add_document()` in `index_corpus.py` (exposed as `POST /api/documents`) chunks a
  new document with the same rules, embeds it and upserts it. Nothing else is touched, and the keyword weights stay
  correct because word rarity is computed by Qdrant at query time, not stored in a file.
- **Replacing is safe.** Point ids are derived from `(doc_id, chunk number)`, so adding a document again overwrites it,
  and its old chunks are deleted first in case the new version is shorter.
- **It scales.** Search happens on the Qdrant server (HNSW index for vectors, inverted index for keywords), so the app
  stays small whatever the corpus size, and several app instances can share one collection.
- **Filtering is native** (payload filter on `type`) instead of filtering rows in Python.

The trade-off: every search is a network call to Qdrant Cloud (both searches go in one batched request). Without
`QDRANT_URL`, `store.py` falls back to an embedded local Qdrant in `.qdrant/`, so a fresh clone can build its own copy
with `python index_corpus.py --recreate` (about 10 minutes on a laptop).

**Embedding models.** `bge-small-en-v1.5` runs on a CPU with no API key or cost and is a strong small retrieval model;
it reads at most 512 tokens, so the end of very long chunks is not embedded (see failure cases). Both models run
through `fastembed` (ONNX runtime), so the project needs no PyTorch.

## 4. Retrieval

1. *(optional)* **Rewrite** a short or vague question into a full search query with `gpt-4o-mini`.
2. **Keyword search** (BM25) and **vector search** each return their top 50 chunks. BM25 is strong on exact strings
   (error messages, function names, version numbers); vectors are strong on paraphrases ("stop my app from re-running"
   finds the Session State pages). Streamlit questions contain both kinds.
3. **Reciprocal rank fusion** (constant 60) merges the two lists: a chunk ranked high by both wins.
4. **Metadata filter**: the user can restrict the search to docs, issues or release notes.
5. The top 10 chunks go to the language model (10 is the accuracy-tuned default; see [optimisation.md](optimisation.md)).

**Reranker: measured, not enabled.** We tested a cross-encoder (`ms-marco-MiniLM-L-6-v2`, run through ONNX) that rescores
the top 30 hybrid results. It lifts recall@3 (0.66 → 0.75) and recall@5 (0.77 → 0.80), but adds about 3.5 s per
question on a CPU (60 ms → 3,526 ms), more than doubling the response time. For an interactive service we skip it;
reranking only when the first search is unsure is the next thing to try. "Skipping the reranker" is also one of the
course's suggested Week 8 savings. The full ablation is in [report.pdf](report.pdf).

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

`app.py` (FastAPI) connects to Qdrant and loads the embedding models once, and serves the page in `static/`. Each request is logged with its
latency, tokens and cost (`logs/requests-*.jsonl`); the page's 👍/👎 buttons are saved to `logs/feedback-*.jsonl`. Because the URL will
be public and every question costs money, each visitor is limited to 10 questions a minute and the whole service to
300 a day. Repeated questions are answered from an in-memory cache at no cost. On a host whose disk is wiped on restart,
the logs can be pushed to a private Hugging Face dataset every 5 minutes (`HF_LOGS_REPO`).

**Adding content.** `POST /api/documents` (title, text, type, url, optional `doc_id` and date) adds or replaces a
document; `DELETE /api/documents/{doc_id}` removes one. Both require the `LIBRARIAN_ADMIN_TOKEN` secret and clear the
answer cache, so the next question sees the new content.

**Deployment.** The service runs on Render's free plan (512 MB), set up by `render.yaml`. The chunks live in Qdrant
Cloud and the embedding models run through ONNX, so the app itself only holds the two models (about 300 MB).

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
- **Dotted names in keyword search.** The BM25 tokenizer splits `st.cache_data` into separate words; it still matches,
  but the exact name is not one token.
- **Model knowledge leaking in.** Streamlit is popular, so the model may "know" an answer that isn't in the passages;
  the faithfulness measure in the report catches this.
- **Citations on every sentence are not guaranteed.** The model sometimes cites only some sentences; the report
  measures how often.
- **Scrubbing is regex-based**, a first line of defence rather than a guarantee.
