"""Answer generation for The Librarian: find chunks, ask the LLM, return an answer with citations.

Usage:
    python generate.py "how do I keep state between reruns?"
    python generate.py "state?" --rewrite            # rewrite a vague question first
    python generate.py "..." --min-score 0.6         # say "I don't know" if search is weak
    python generate.py "..." --types issue,release   # only search these source types
"""
import argparse, re, time

from dotenv import load_dotenv
from openai import OpenAI

import store

load_dotenv()
LLM_MODEL = "gpt-4o-mini"
IDK = "I don't know."

# USD per 1M tokens (input, output). Check https://openai.com/api/pricing before quoting in the report.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4.1": (2.00, 8.00),
          "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40)}


def cost_usd(model, prompt_tokens, completion_tokens):
    p_in, p_out = PRICES.get(model, (float("nan"), float("nan")))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6

SYSTEM_PROMPT = """You answer questions about Streamlit using ONLY the numbered sources you are given.
Rules:
1. Every sentence that states a fact must end with the number of the source it comes from, like [1] or [2][3].
2. Do not use anything you know that is not in the sources.
3. If the sources do not contain the answer, reply exactly: I don't know.
4. Be short and clear. Show code only if a source shows it."""

REWRITE_PROMPT = """Rewrite the user's question about Streamlit as one clear, complete search query.
Keep the same meaning. Reply with the query only."""


def format_sources(sources):
    """The numbered context the LLM sees (also shown to the judge in eval/generation_eval.py)."""
    return "\n\n".join(f"[{n}] {c['title']} > {c['section']}\n{c['text']}"
                       for n, c in enumerate(sources, 1))


class Librarian:
    def __init__(self, collection=store.COLLECTION):
        # 1. connect to the Qdrant vector store and load the embedding models once, not on every question
        self.collection = collection
        store.ensure_collection(collection)                  # a fresh store starts empty instead of failing
        store.dense_model(), store.sparse_model()
        self._client = None

    @property
    def client(self):
        # 2. connect to OpenAI on first use, so search-only runs (retrieval eval) need no API key
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def _chat(self, system, user, usage):
        r = self.client.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        usage["prompt_tokens"] += r.usage.prompt_tokens        # counted for the cost model
        usage["completion_tokens"] += r.usage.completion_tokens
        return r.choices[0].message.content.strip()

    def rewrite(self, question, usage):
        return self._chat(REWRITE_PROMPT, question, usage)

    def retrieve(self, query, k=5, pool=50, mode="hybrid", types=None):
        """mode: "hybrid", or "vector" / "bm25" alone (used by the ablation in eval/).
        types: only search these source types, e.g. ["issue", "release"] (metadata filter)."""
        return store.search(query, k, pool, mode, types, self.collection)

    # ---------- step 3: ask the LLM to answer from the chunks ----------
    def answer(self, question, k=5, rewrite=False, min_score=0.0, sources=None, types=None):
        """sources: pass chunks here to skip our search (e.g. reranked chunks). types: metadata filter."""
        t0 = time.time()
        usage = dict(prompt_tokens=0, completion_tokens=0)
        query = self.rewrite(question, usage) if rewrite else question
        if sources is None:
            sources, best = self.retrieve(query, k, types=types)
        else:
            best = max((s.get("score") or 0.0) for s in sources) if sources else 0.0

        result = dict(question=question, query=query, sources=sources, best_score=round(best, 3),
                      model=LLM_MODEL)

        # nothing relevant found -> don't even call the LLM
        if not sources or best < min_score:
            return dict(result, answer=IDK, citations=[], bad_citations=[], **usage,
                        tokens=sum(usage.values()), cost_usd=cost_usd(LLM_MODEL, **usage),
                        latency_s=round(time.time() - t0, 2))

        text = self._chat(SYSTEM_PROMPT, f"Sources:\n{format_sources(sources)}\n\nQuestion: {question}", usage)

        # read the [n] numbers the LLM wrote and turn them into links
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        citations = [dict(n=n, title=sources[n - 1]["title"], url=sources[n - 1]["url"])
                     for n in used if 1 <= n <= len(sources)]
        bad = [n for n in used if not 1 <= n <= len(sources)]   # numbers that don't exist

        return dict(result, answer=text, citations=citations, bad_citations=bad, **usage,
                    tokens=sum(usage.values()), cost_usd=cost_usd(LLM_MODEL, **usage),
                    latency_s=round(time.time() - t0, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5, help="how many chunks to send to the LLM")
    ap.add_argument("--rewrite", action="store_true", help="rewrite the question first")
    ap.add_argument("--min-score", type=float, default=0.0, help="below this, answer I don't know")
    ap.add_argument("--types", help="comma list of source types to search: doc, issue, release")
    ap.add_argument("--collection", default=store.COLLECTION)
    a = ap.parse_args()

    res = Librarian(a.collection).answer(a.question, k=a.k, rewrite=a.rewrite, min_score=a.min_score,
                                    types=a.types.split(",") if a.types else None)

    print("\n=== ANSWER ===")
    print(res["answer"])
    print("\n=== SOURCES CITED ===")
    for c in res["citations"]:
        print(f"[{c['n']}] {c['title']}  {c['url']}")
    if res["bad_citations"]:
        print("WARNING: cited sources that do not exist:", res["bad_citations"])
    print(f"\nquery used: {res['query']}")
    print(f"best search score: {res['best_score']}   tokens: {res['tokens']}   time: {res['latency_s']}s")
