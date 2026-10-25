"""Answer generation for The Librarian: find chunks, ask the LLM, return an answer with citations.

Usage:
    python generate.py "how do I keep state between reruns?"
    python generate.py "state?" --rewrite            # rewrite a vague question first
    python generate.py "..." --min-score 0.6         # say "I don't know" if search is weak
"""
import argparse, json, pickle, re, time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from index_corpus import QUERY_PREFIX, tokenize  

load_dotenv()                      
LLM_MODEL = "gpt-4o-mini"
IDK = "I don't know."

SYSTEM_PROMPT = """You answer questions about Streamlit using ONLY the numbered sources you are given.
Rules:
1. Every sentence that states a fact must end with the number of the source it comes from, like [1] or [2][3].
2. Do not use anything you know that is not in the sources.
3. If the sources do not contain the answer, reply exactly: I don't know.
4. Be short and clear. Show code only if a source shows it."""

REWRITE_PROMPT = """Rewrite the user's question about Streamlit as one clear, complete search query.
Keep the same meaning. Reply with the query only."""


class Librarian:
    def __init__(self, index_dir="index"):
        from sentence_transformers import SentenceTransformer
        out = Path(index_dir)
        # 1. load Rahma's index (built once by index_corpus.py)
        with open(out / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.emb = np.load(out / "embeddings.npy")
        with open(out / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        cfg = json.loads((out / "config.json").read_text())
        self.embedder = SentenceTransformer(cfg["model"])
        # 2. connect to OpenAI
        self.client = OpenAI()

  
    def rewrite(self, question):
        r = self.client.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role": "system", "content": REWRITE_PROMPT},
                      {"role": "user", "content": question}])
        return r.choices[0].message.content.strip()

    def retrieve(self, query, k=5, pool=50):
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        sims = self.emb @ q                                   # meaning similarity, one per chunk
        vec_rank = np.argsort(-sims)[:pool]                   # best by meaning
        kw_rank = np.argsort(-self.bm25.get_scores(tokenize(query)))[:pool]  # best by keywords
        rrf = {}                                              # mix the two lists
        for ranks in (vec_rank, kw_rank):
            for r, i in enumerate(ranks):
                rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + r)
        top = sorted(rrf, key=rrf.get, reverse=True)[:k]
        best_score = float(sims[vec_rank[0]])                 # how close the best chunk is
        return [dict(self.chunks[i], score=float(sims[i])) for i in top], best_score

    # ---------- step 3: ask the LLM to answer from the chunks ----------
    def answer(self, question, k=5, rewrite=False, min_score=0.0, sources=None):
        """sources: pass chunks here to skip our search (e.g. Nadia's reranked chunks)."""
        t0 = time.time()
        query = self.rewrite(question) if rewrite else question
        if sources is None:
            sources, best = self.retrieve(query, k)
        else:
            best = max(s.get("score", 1.0) for s in sources) if sources else 0.0

        result = dict(question=question, query=query, sources=sources, best_score=round(best, 3))

        # nothing relevant found -> don't even call the LLM
        if not sources or best < min_score:
            return dict(result, answer=IDK, citations=[], bad_citations=[], tokens=0,
                        latency_s=round(time.time() - t0, 2))

        context = "\n\n".join(f"[{n}] {c['title']} > {c['section']}\n{c['text']}"
                              for n, c in enumerate(sources, 1))
        r = self.client.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}])
        text = r.choices[0].message.content.strip()

        # read the [n] numbers the LLM wrote and turn them into links
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        citations = [dict(n=n, title=sources[n - 1]["title"], url=sources[n - 1]["url"])
                     for n in used if 1 <= n <= len(sources)]
        bad = [n for n in used if not 1 <= n <= len(sources)]   # numbers that don't exist

        return dict(result, answer=text, citations=citations, bad_citations=bad,
                    tokens=r.usage.total_tokens, latency_s=round(time.time() - t0, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5, help="how many chunks to send to the LLM")
    ap.add_argument("--rewrite", action="store_true", help="rewrite the question first")
    ap.add_argument("--min-score", type=float, default=0.0, help="below this, answer I don't know")
    ap.add_argument("--index", default="index")
    a = ap.parse_args()

    res = Librarian(a.index).answer(a.question, k=a.k, rewrite=a.rewrite, min_score=a.min_score)

    print("\n=== ANSWER ===")
    print(res["answer"])
    print("\n=== SOURCES CITED ===")
    for c in res["citations"]:
        print(f"[{c['n']}] {c['title']}  {c['url']}")
    if res["bad_citations"]:
        print("WARNING: cited sources that do not exist:", res["bad_citations"])
    print(f"\nquery used: {res['query']}")
    print(f"best search score: {res['best_score']}   tokens: {res['tokens']}   time: {res['latency_s']}s")
