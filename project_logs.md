# Project 2, The Librarian: decisions, problems and fixes

Working log for the report, `design.md` and `postmortem.md`. Each section says which deliverable it feeds.
Items marked **[verify]** or **[fill in]** need a real number or a check from the team before they go in the report.

Last updated: see git history.

---

## 1. The project in one paragraph

A retrieval-augmented (RAG) service over the **Streamlit** open-source project: its documentation, closed GitHub issues
with discussion, and release notes. It answers only from those documents, cites its sources, and says "I don't know"
when nothing relevant is found. The course asks for hybrid search (BM25 + vectors), a reranker, retrieval and
generation measured separately, an ablation, and a Week 8 optimisation (50% cheaper or lower p95 latency, at most
2 points of accuracy lost).

---

## 2. Design decisions and why (feeds `design.md`, report section 1 and 2)

### 2.1 Corpus choice: Streamlit docs + issues + releases

| Slide test                | Streamlit                                                         |
| ------------------------- | ----------------------------------------------------------------- |
| Real                      | Documents people use today                                        |
| Real users                | Classmates already used Streamlit in class; feedback is reachable |
| Allowed to use            | Apache 2.0 licence (confirmed on the repo page); public data      |
| Big enough to need search | 405 docs pages, about 300 issues, 60 releases, 3,562 chunks       |
| Checkable                 | Passages can be copied verbatim to build 100+ Q&A pairs           |

Known risk: Streamlit is popular, so an LLM may answer basic questions without retrieval and make the ablation look
weak. Mitigations: version- and error-specific questions, questions answerable only from issues, unanswerable
questions, and a no-retrieval baseline as an extra row in the ablation.

### 2.2 Three sources

| Source   | Where it comes from                                                        | Why                                                        |
| -------- | -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Docs     | `git clone https://github.com/streamlit/docs`, Markdown in `content/`      | Authoritative, structured, many headings                   |
| Issues   | GitHub API, `streamlit/streamlit`, closed issues with at least one comment | Real user phrasing, error messages, maintainer answers     |
| Releases | GitHub Releases API, latest 60                                             | Version-specific questions the model cannot have memorised |

Pull requests are excluded. Issues with no comments are excluded because they contain no checkable answer.
Discussions and the community forum were left out of scope (extra API work); candidates if time allows.

### 2.3 Issue count: 300, not 600

The slides do not ask for issues at all; the 405 docs pages already satisfy "hundreds of pages". More issues add noise
(unresolved threads, wrong first answers, pasted logs), cost one extra API request each, and slow every rebuild.
300 is a balance between coverage and noise. **[verify]** the final count in `corpus/manifest.json`.

### 2.4 Personal data handling

The slides say no personal data, and issues contain usernames, emails and pasted logs.

| What                                                                          | Decision                                                                                                                      |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Emails, `@mentions`, API tokens (`ghp_`, `sk-`, `AKIA`), home-directory paths | Replaced by placeholders (`[EMAIL]`, `[USER]`, `[TOKEN]`, `/[HOME]`)                                                          |
| Author logins                                                                 | **Not stored at all**                                                                                                         |
| Commenter identity                                                            | Replaced by the GitHub role (`MEMBER`, `CONTRIBUTOR`, `NONE`), which is useful because a maintainer reply is more trustworthy |

Trade-off: the slides list "author" among the metadata to keep. We deliberately omit it for issues because a GitHub
login is personal data. The scrubber is regex-based and is a first pass, not a guarantee, so the team spot-checks
about 20 issues by hand. **[fill in]** what the spot-check found.

### 2.5 The corpus and index are committed as a frozen snapshot

Originally `.gitignore` excluded `corpus/` and `index/`. We reversed that:

- Gold passages in the eval set point at exact text. GitHub issues and the docs keep changing, so rebuilding later
  would silently invalidate the eval. A snapshot keeps results reproducible.
- The "runs from a clean clone" rule: graders should not need a GitHub token and a 20-minute download.
- The index is small (`chunks.jsonl` is 5.35 MB, vectors about 5 MB) and ships inside the deployed app.

`.env` (the GitHub token) is never committed; `.env.example` is. The build scripts stay in the repo so everything can
be regenerated. **[fill in]** the snapshot date.

### 2.6 Chunking (feeds `design.md`; slides: "chunk size and overlap are decisions you must test, not defaults")

| Setting      | Current value                                                                                                       | Status                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Unit         | Words (approximation of tokens)                                                                                     | Chosen for simplicity                                                  |
| Target size  | 250 words                                                                                                           | **Untested default**: must be compared with 150 and 400 using recall@k |
| Overlap      | 40 words                                                                                                            | **Untested default**                                                   |
| Minimum size | 40 words (docs and releases): smaller chunks merge into a neighbour of the same document, capped at the target size | Untested; test with 0 / 40 / 80                                        |
| Drop         | Chunks under 3 words after cleaning                                                                                 |                                                                        |

How each source is split:

- **Docs**: split by heading first (chunks never cross a heading unless a small one is merged), then packed up to the
  target size by paragraph. Fenced code blocks are kept whole, even if that exceeds the target. Oversized prose
  paragraphs are hard-split by words with overlap.
- **Issues**: description and discussion are chunked separately; each comment is tagged with the commenter's role.
  Issues are not cleaned (user-pasted code and logs stay as written) and small issue chunks are not merged.
- **Releases**: split by heading, cleaned like docs.

Metadata stored per chunk: `id`, `doc_id`, `type`, `title`, `section`, `url`, `date`.
Docs have no `date` (the script copies Markdown only); this limits time-based filtering to issues and releases.

Embedding text is `title > section` followed by the chunk text, so the heading context travels with the chunk.

### 2.7 MDX/JSX cleaning (added after inspecting real chunks)

The docs are MDX (Markdown mixed with React components). Cleaning is applied to docs and releases only, never inside
code blocks:

| Noise                                             | Treatment               |
| ------------------------------------------------- | ----------------------- | --- | --- |
| `<InlineCallout bold="X">text</InlineCallout>`    | Kept as `X text`        |
| `<Tile title text />`                             | Kept as `title: text`   |
| `<Image ... />`, `![alt](...)`, `<i>` icon tags   | Removed                 |
| HTML tables                                       | Rows become `a          | b   | `   |
| `<Note>`, `<Important>`, `<Tip>`, `<Warning>`     | Become `Note: ...` etc. |
| `<a>`, `<code>`, entities (`&mdash;`, `&dagger;`) | Reduced to their text   |

Effect on the index (see section 4): fewer chunks, longer on average, and no markup in embeddings, BM25 or gold passages.
`--no-clean` exists to reproduce the old behaviour for an ablation. **[fill in]** the recall@k difference between
`index` and `index_raw` once the eval exists.

### 2.8 Embeddings and storage: flat files

| Choice              | Value                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Model               | `BAAI/bge-small-en-v1.5`, 384-dimensional, runs on CPU with no API key                                                   |
| Vectors             | Normalised, stored in `index/embeddings.npy` (float32)                                                                   |
| Chunks and metadata | `index/chunks.jsonl`                                                                                                     |
| Keyword index       | `index/bm25.pkl` (BM25Okapi from `rank_bm25`)                                                                            |
| Reproducibility     | `index/config.json` records chunk size, overlap, min words, cleaning flag, model, chunk count, build time, average words |

Why flat files instead of a vector database or Postgres:

- The corpus is small (about 3,500 chunks, vectors of roughly 5 MB), so brute-force dot product is instant.
- No server or account: one script rebuilds the index from `corpus/` (fits "one setup command, one run command").
- Deployment is simple: the index ships inside the app.
- Cheap and fast, which helps the Week 8 numbers.

Trade-offs accepted: metadata filtering is manual (filter chunk rows in Python), no incremental updates (full rebuild
about 5 minutes), and `bm25.pkl` uses pickle, which should only be loaded from a trusted source (our own repo).
Re-evaluate if the corpus grows by an order of magnitude.

Model limits to remember: the model reads at most 512 tokens, so very long code or log chunks may be truncated at
embedding time. This is a candidate failure case.

### 2.9 Keyword index and hybrid search

- BM25 over `title > section` plus text, tokenised as lowercase `\w+`. This splits `st.cache_data` into `st` and
  `cache_data`; matching still works but exact dotted names are not single tokens. Test a tokenizer that keeps them.
- Hybrid = reciprocal rank fusion (RRF, constant 60) of the top 50 vector results and top 50 BM25 results.
- Why both: BM25 wins on exact strings (error messages, function names, version numbers); vectors win on paraphrase
  ("stop my app from re-running" finds session-state pages). Streamlit questions contain both kinds.
- Reranker (cross-encoder over the top 20 to 50 hybrid results): **not built yet**.

### 2.10 Evaluation dataset design (feeds `data/`, report section 3 and 4)

File: `data/eval.json` (renamed from `eval_v2.json` when final), a JSON array.

| Field                                                    | Purpose                                                                                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `id`, `category`, `question_type`                        | Identification; results are broken down by type (`how-to`, `error-message`, `version`, `conceptual`, `multi-part`, `unanswerable`) |
| `question`, `answer`                                     | Written like a real user would ask; reference answer for correctness                                                               |
| `answerable`                                             | `false` for questions the corpus cannot answer (the "I don't know" test)                                                           |
| `split`                                                  | `dev` or `test`; stratified by category, about 20% `test`, seed 42. Never tune on `test`                                           |
| `doc_id`, `url`, `source_type`, `section`, `source_file` | Where the answer lives                                                                                                             |
| `source_passage`                                         | Tight verbatim span (1 to 4 sentences, up to about 90 words) copied from an index chunk                                            |
| `gold_chunk_ids`                                         | Every chunk of that document containing the passage (neighbouring chunks overlap, so there can be two)                             |
| `match`                                                  | Remap diagnostics (cosine, overlap, answer support, `needs_review`)                                                                |

Hit rule for recall@k (fix it and keep it fixed): a query is a hit if any of the top-k chunks has an id in
`gold_chunk_ids`. Also report a document-level version (same `doc_id`). Keep keys on `doc_id` plus verbatim passage so the
eval survives re-chunking.

Unanswerable questions: about 10 to 15% of the set, about other libraries or nonexistent Streamlit features; each must be
verified absent by searching the index first.

---

## 3. Problems we hit and how we fixed them (feeds `postmortem.md`; "a postmortem with no problems loses points")

| #   | Problem                                                                                                                | Cause                                                                                                                                                                                                                                                                                                                                                                                                                                  | Fix                                                                                                                                                                                                       | Lesson                                                                                               |
| --- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1   | Plan said to clone the docs from `streamlit/streamlit`                                                                 | The docs live in a separate repo, `streamlit/docs`; the main repo only holds the library code                                                                                                                                                                                                                                                                                                                                          | Collector clones `streamlit/docs` and reads `content/`                                                                                                                                                    | Check where the data actually lives before writing the collector                                     |
| 2   | `PermissionError: [WinError 5]` when deleting the temporary `_docs_clone` folder                                       | On Windows, git marks its internal files read-only, so `shutil.rmtree` fails; the docs had already been copied                                                                                                                                                                                                                                                                                                                         | `safe_rmtree` clears the read-only flag and retries; the build also aborts clearly if zero docs pages are found                                                                                           | Test file-system code on Windows; fail loudly on empty results                                       |
| 3   | Only **149 issues** with `--max-issues 600`                                                                            | The first version fetched a fixed batch of 1,200 items, then discarded pull requests and issues without comments                                                                                                                                                                                                                                                                                                                       | Lazy paging until the target is reached; resumable (existing files are skipped); progress printed. The log confirmed it: 300 issues kept out of 1,800 items scanned, so about 17% of closed items qualify | Do not assume a filter's yield; a target needs a loop that stops on the target, not on a fixed batch |
| 4   | Collection runs for a long time                                                                                        | One extra API request per issue for comments, plus rate limits                                                                                                                                                                                                                                                                                                                                                                         | Resumable runs, a smaller target (300), and a one-line fallback to write `manifest.json` after an interrupted run                                                                                         | Interrupting can leave a half-written file; delete it if indexing reports a JSON error               |
| 5   | `.gitignore` excluded the corpus and index                                                                             | Initial advice treated them as build artefacts                                                                                                                                                                                                                                                                                                                                                                                         | Now committed as a frozen snapshot (section 2.5)                                                                                                                                                          | The eval set depends on exact text, so the data is part of the experiment                            |
| 6   | **Initial eval dataset was not usable**                                                                                | 100 entries, 100% docs, only 23 distinct invented file names such as `corpus/docs/widgets.md` (22 questions on one file), which do not exist (real names look like `deploy__community-cloud__...md`); passages looked written from memory, not copied; one error-message question, one version question, no unanswerable questions, no split; 62 of 100 questions started with "How" and 43% of question words appeared in the passage | `remap_dataset.py` re-grounds every entry on real chunks and adds the missing fields; `validate_eval.py` checks it                                                                                        | An eval set written without the corpus open is not an eval set                                       |
| 7   | Chunks full of MDX/JSX markup, many tiny chunks, repeated text from overlap                                            | The docs are MDX and the chunker treated them as plain text                                                                                                                                                                                                                                                                                                                                                                            | `clean_mdx` and `merge_small` (section 2.6, 2.7)                                                                                                                                                          | Read a random sample of chunks before building anything on them                                      |
| 8   | First version of the cleaner deleted every blank line and glued a heading onto a code fence                            | A line filter removed empty lines; trailing newlines were lost                                                                                                                                                                                                                                                                                                                                                                         | Found by testing on real snippets; replaced with targeted regexes that keep paragraph breaks                                                                                                              | Test cleaners on real examples, including code blocks                                                |
| 9   | Validator failed on cleaned docs                                                                                       | It compared passages with raw files, which no longer match after cleaning                                                                                                                                                                                                                                                                                                                                                              | Validator now checks the passage against the chunks of the entry's own `doc_id`                                                                                                                           | Validate against what the system actually indexes                                                    |
| 10  | Passages were whole chunks (up to 250 words) and a correct retrieval of an overlapping neighbour would count as a miss | Chunks are long, and 40 words are repeated between neighbours                                                                                                                                                                                                                                                                                                                                                                          | Passage is now a tight span; `gold_chunk_ids` lists every chunk containing it                                                                                                                             | Overlap changes how recall must be scored                                                            |
| 11  | Validator reported 0 problems, but answers were not supported by their passages                                        | Validator checks that passages exist in the index, not that they support the answer. Remap kept the original answers (written from memory). Example: `eval_002` had answer support 0.33; `eval_003` scored 0.53, just above the 0.4 flag threshold                                                                                                                                                                                     | **Open**: raise the flag threshold (0.4 to 0.6), review every flagged entry, rewrite answers so each claim appears in the passage, spot-check 15 unflagged entries                                        | A structural check is not a quality check                                                            |

### Process notes

- The scripts (collection, indexing, remap, validation) were drafted with an AI assistant and tested by the team on its
  own data. The course rule is that every team member can explain every line, so each member should read the scripts
  before the demo. Some scripts were first tested only on small fake data, and the real runs surfaced problems 2 and 3.
- Time cost: building the index takes about 5 minutes on a laptop CPU (333.5 s with the old chunks, 294.8 s after
  cleaning), the corpus build takes 10 to 30 minutes.

---

## 4. Results so far (feeds report section 2)

| Build        | Issues    | Cleaning            | `n_chunks` | `avg_words`  | Build time   |
| ------------ | --------- | ------------------- | ---------- | ------------ | ------------ |
| v0           | 149       | no                  | 2,937      | not recorded | not recorded |
| v1           | about 300 | no                  | 4,007      | 132.0        | 333.5 s      |
| v2 (current) | about 300 | yes, `min_words` 40 | **3,562**  | **147.0**    | 294.8 s      |

- Cleaning and merging removed about 11% of chunks and raised the average length by about 11%.
- Observed on a first manual search: for `StreamlitAPIException set_page_config` the hybrid top results were the
  `st.set_page_config` issue and its reference page (relevant). One query is an observation, not a result.
- No recall@k, ablation, faithfulness, latency or cost numbers exist yet. **[fill in]** when they do.

---

## 5. Known failure cases and risks (feeds `design.md`, report section 6)

- **Tables and HTML**: flattened to `a | b |` rows; column meaning may be lost.
- **Code blocks**: kept whole, so some chunks exceed the target size and may be truncated at the 512-token embedding limit.
- **Cross-references**: `[text](/path)` links are kept as text, but the system cannot follow a link to another page.
- **Residual MDX**: the cleaner covers the components seen so far; other components may remain. Re-sample chunks after each rebuild.
- **API reference pages**: partly generated, some pages are short stubs that give very small chunks.
- **Issues**: noisy, may contain a wrong first answer later corrected; long pasted logs; issue chunks are not cleaned.
- **Scrubbing**: regex-based, not guaranteed.
- **Time questions on docs**: docs chunks have no date, so "since version X" cannot be filtered for docs.
- **Model knowledge leakage**: a model may answer popular questions from memory; faithfulness and citation checks must catch this.
- **Eval answers**: originals were written from memory and may claim more than the passage supports (problem 11).
- **Tokenizer**: BM25 splits dotted names (`st.cache_data`).

---

## 6. Open work, mapped to the slides

| Item                                                                                                                                                                                                                                         | Slide requirement                         | Status                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Finalise `data/eval.json`: review flagged answers, add issue, release, error-message, version, multi-part and unanswerable questions; aim for 120 to 150 entries                                                                             | 100+ Q&A with source passage              | In progress                                                                                                 |
| Test chunk size (150 / 250 / 400), overlap and `min_words` against recall@k                                                                                                                                                                  | Chunk decisions must be tested            | Not started                                                                                                 |
| `eval/retrieval_eval.py`: recall@k, MRR optional, search latency; modes keyword, vector, hybrid, hybrid + reranker                                                                                                                           | Retrieval measured separately; ablation   | Not started                                                                                                 |
| `retrieval.py` with `search(query, k, mode, filters)` and metadata filtering                                                                                                                                                                 | Retrieve with filters                     | Not started (demo exists in `index_corpus.py --query`)                                                      |
| Reranker                                                                                                                                                                                                                                     | Rerank with cross-encoder or small model  | Not started                                                                                                 |
| Query rewriting                                                                                                                                                                                                                              | Short, vague, multi-part questions        | Not started                                                                                                 |
| Generation with citations, "I don't know"                                                                                                                                                                                                    | Answer from chunks only, each claim cited | Not started; needs an LLM choice and API key in `.env`                                                      |
| `eval/generation_eval.py`: faithfulness, correctness, citation accuracy, "I don't know"                                                                                                                                                      | Generation measured separately            | Not started                                                                                                 |
| Deployed service with URL; user feedback                                                                                                                                                                                                     | Works 25%; real users                     | Not started                                                                                                 |
| Cost model (today and at 100x), p50 / p95 latency logging                                                                                                                                                                                    | Measured 25%                              | Not started; log both cost and latency from the first deployed version so the Week 8 "before" numbers exist |
| Week 8 optimisation (50% cost or p95 cut, at most 2 points of accuracy lost)                                                                                                                                                                 | Efficient 25%                             | Not started; decide the headline accuracy metric and what "2 points" means before Week 8                    |
| `README.md` with Contributions section, `optimisation.md`, `postmortem.md`, 6-page report                                                                                                                                                    | Deliverables                              | README does not exist yet                                                                                   |
| Repo cleanup: delete the original invalid `data/eval_dataset.json` and `remap_dataset.py` (or move to `scripts/`) once the dataset is final; move `validate_eval.py` to `eval/`; keep `eval_review.json` and `eval_problems.json` out of git | Clean repo                                | After the dataset is final                                                                                  |

Questions to settle with the instructor: what the unstated last 25% of the grade covers, what "2 points of accuracy" refers
to, and how long the deployed URL must stay up.

---

## 7. Reproduce the current state

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then paste GITHUB_TOKEN
python build_corpus.py --max-issues 300                # -> corpus/
python index_corpus.py                                 # -> index/ (cleaned, min_words 40)
python index_corpus.py --query "how do I keep state between reruns?"
python remap_dataset.py --input data/eval_dataset.json --out data/eval_v2.json
python validate_eval.py data/eval_v2.json
# for the cleaning ablation:
python index_corpus.py --no-clean --min-words 0 --out index_raw
```

---

## 8. Files in the repo

| File                                             | Role                                                                                               |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `build_corpus.py`                                | Downloads docs, issues and releases, scrubs personal data, writes `corpus/` and `manifest.json`    |
| `index_corpus.py`                                | Cleans, chunks, embeds, builds BM25, writes `index/`; `--query` runs a vector / BM25 / hybrid demo |
| `remap_dataset.py`                               | One-time migration: grounds the original questions on real chunks and adds the missing fields      |
| `validate_eval.py`                               | Checks that every entry's passage exists in the index; read-only, writes `eval_problems.json`      |
| `requirements.txt`, `.env.example`, `.gitignore` | Setup; `.env` is never committed                                                                   |
