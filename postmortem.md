# Postmortem: The Librarian

What went wrong while building the project, why, what we did about it, and what we learned. Problems are in the order
we met them.

## The biggest lessons

1. **Our first evaluation set measured our labelling mistakes, not our search.** With the first dataset, hybrid search
   found the right passage in the top 5 for only 42% of questions. After we re-checked every question against the
   actual index, the same search scored 80%. Most "misses" were questions pointing at the wrong passage.
2. **A check that passes is not a check that matters.** Our validator reported "0 problems" on a dataset where a third
   of the answers were not supported by their passages: it only checked that each passage existed somewhere.
3. **Read the data before building on it.** The scraped API reference pages turned out to be empty template tags, the
   docs were full of website markup, and one "chunk" was a 3,923-word pasted log. Each one was found by opening files,
   not by running code.
4. **Check the hosting plan's current terms before writing the deployment.** Hugging Face Spaces stopped offering free
   Docker hosting, and we found out only when the upload was refused.

## Problems, causes, fixes

### Building the corpus

| # | What went wrong | Why | What we did | Lesson |
|---|---|---|---|---|
| 1 | The plan said to take the docs from the `streamlit/streamlit` repository | The docs live in a separate repository, `streamlit/docs` | The collector clones `streamlit/docs` and reads its `content/` folder | Check where the data actually lives before writing the collector |
| 2 | `PermissionError: [WinError 5]` when deleting the temporary clone | On Windows, git marks its internal files read-only | `safe_rmtree` clears the flag and retries; the build stops loudly if zero pages were copied | Test file-system code on the OS the team uses; fail loudly on empty results |
| 3 | Only 149 issues collected when we asked for 600 | The first version fetched a fixed batch, then threw away pull requests and issues without comments (only about 17% of closed items qualify) | Page through results until the target is reached; resumable runs | Don't assume how many items a filter keeps |
| 4 | Collection took 10 to 30 minutes and could be interrupted mid-file | One extra API call per issue for its comments, plus GitHub rate limits | Resumable collection, a progress counter, and a smaller issue target (418 kept) | Long jobs need to be restartable |
| 5 | `corpus/` and `index/` were ignored by git | They were first treated as build output | Committed as a frozen snapshot (5 MB corpus, 15 MB index) | The evaluation points at exact text, so the data is part of the experiment |
| 6 | Issues contain personal data (usernames, emails, tokens, home paths) | GitHub threads are written by the public | A scrubber replaces them with `[USER]`, `[EMAIL]`, `[TOKEN]`, `/[HOME]`; author logins are never stored, only their role (`MEMBER`, `CONTRIBUTOR`, …) | The scrubber is regex-based: a first line of defence, not a guarantee |

### Building the index

| # | What went wrong | Why | What we did | Lesson |
|---|---|---|---|---|
| 7 | Chunks were full of website markup (`<InlineCallout>`, `<Tile>`, HTML tables) and many were a few words long | The docs are MDX (Markdown mixed with React components) and were chunked as plain text | `clean_mdx` strips the markup without touching code blocks; `merge_small` joins tiny chunks to a neighbour. Chunks went from 4,007 to 3,562 | Read a random sample of chunks before building anything on them |
| 8 | The first cleaner deleted every blank line and glued headings onto code fences | A line filter removed empty lines | Replaced by targeted rules that keep paragraph breaks, tested on real snippets | Test cleaners on real examples, including code |
| 9 | The API reference pages are empty: `st.radio`'s page contains only `<Autofunction function="streamlit.radio" />` | The docs website fills in each function's description at build time from the library's source code; the Markdown we cloned never had it | Not fixed. Questions are grounded on concept pages, the API index, the cheat sheet and the knowledge base instead. Fix for later: add the docstrings from the installed `streamlit` package to the corpus | Check that the scraped text contains what you see on the website |
| 10 | 17 chunks are longer than 400 words, the longest 3,923 | Code blocks are never split (to keep code runnable) and issue comments with pasted logs are kept whole | Accepted; listed as a known failure case. The embedding model only reads the first 512 tokens of these | Every chunking rule has an edge case; write it down |

### Building the evaluation set

| # | What went wrong | Why | What we did | Lesson |
|---|---|---|---|---|
| 11 | The first dataset could not be used: invented file names (`corpus/docs/widgets.md` does not exist), passages written from memory, no unanswerable questions, no train/test split | It was written without the corpus open | A remapping script re-attached every question to the most similar real chunk and added the missing fields | An evaluation set written without the corpus open is not an evaluation set |
| 12 | The validator passed on cleaned docs only after a rewrite | It compared passages with the raw files, which no longer matched after cleaning | It now checks each passage against the chunks the system actually indexes | Validate against what the system really searches |
| 13 | A correct retrieval of a neighbouring chunk counted as a miss | Chunks overlap by 40 words, and passages were whole chunks | Passages became short exact quotes; every chunk containing the quote counts as correct | Overlap changes how recall must be scored |
| 14 | **The remapped dataset was still wrong**: recall@5 was 0.42, and questions sourced from GitHub issues scored 0.03 to 0.23 | The remapping picked the most *similar* text, often an unrelated issue (the `st.stop()` question pointed at a dialog bug, the `st.radio` question at a screen-reader bug). Each question allowed one correct place, though answers appear on several pages. 92 of 100 questions were generic "how do I…" questions that a language model answers from memory | Rebuilt by hand against the index: 69 questions re-attached to the passage that really answers them, several correct documents allowed per question, answers rewritten so every claim is in a passage; 31 trivial questions replaced by error-message, version/release and GitHub-issue questions. Still exactly 100 answerable questions. Recall@5 rose to 0.80 | Before blaming search for a miss, read what it retrieved |
| 15 | The "needs review" flag caught only 17 of 58 bad entries | It was based on word overlap, which a wrong-but-similar passage passes easily | Dropped the flag; every entry was reviewed against the index | Automatic quality flags need to be checked against a hand-checked sample |

### Generation, evaluation runs and the service

| # | What went wrong | Why | What we did | Lesson |
|---|---|---|---|---|
| 16 | `requirements.txt` did not list `openai`, so a clean clone could not run `generate.py` | The package was installed by hand on one laptop | Added it, plus `OPENAI_API_KEY` in `.env.example` | Test the setup from a clean clone, not your own environment |
| 17 | Running two evaluation scripts at once crashed with `OSError: The paging file is too small` | Each script loads PyTorch and the embedding model (about 600 MB); Windows ran out of virtual memory | Evaluations run one after another | Measure the memory of a process before running several |
| 18 | The model cites only some sentences: in one answer only the last sentence carried a `[1]` | The prompt asks for a citation per factual sentence, but `gpt-4o-mini` does not always follow it | Measured instead of guessed: the generation evaluation reports the share of claims carrying a citation (see the report) | Instructions to a model are requests, not guarantees; measure them |
| 19 | The cross-encoder reranker did not help where it mattered | It improves the very first result and recall@20, but not recall@5 (what the model receives), and adds about 1.9 s per question on a laptop CPU | Kept as a measured option in the evaluation, not in the live service | Add a component only when the numbers show it pays for itself |

### Deployment

| # | What went wrong | Why | What we did | Lesson |
|---|---|---|---|---|
| 20 | The planned host refused the upload: `402 Payment Required` | Hugging Face now requires a paid PRO subscription for Docker Spaces on its free hardware | Switched to Render's free plan, which builds straight from the GitHub repository (`render.yaml`) | Check the current pricing page, not a blog post or memory |
| 21 | The service needed about 640 MB of RAM, more than the 512 MB of the remaining free hosts | PyTorch was loaded only to embed the question | The deployed service runs the same embedding model through ONNX (`fastembed`): about 290 MB. Before switching we checked that question vectors match (cosine ≥ 0.99999 on all 112 questions) and that recall is identical at every k, so the evaluation still describes the deployed service | Know your memory footprint before choosing a host, and verify a "drop-in" replacement before trusting it |

## Process

- **We started measuring late.** The evaluation scripts came after the generation code, so for a while we could not tell
  whether a change helped. Next time: build the evaluation set and a recall script in the first week.
- **The dataset was written separately from the corpus.** Problems 11 and 14 cost the most time of anything in the
  project. Next time: write questions while reading the indexed chunks, one source passage at a time.
- **AI assistance.** Most scripts were drafted with an AI coding assistant and then run, checked and corrected on our own
  data; several of the problems above (2, 3, 14, 16) were found exactly that way. The course rule is that every member
  can explain every line, so each member reviewed the scripts before the demo.
