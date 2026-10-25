"""Shared helpers for the eval scripts: paths, arguments, dataset loading, gold matching, reranker,
cost model, tables."""
import json, re, sys, time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                  # so `import generate` works when run as eval/<script>.py
RESULTS = ROOT / "results"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

from generate import cost_usd as cost          # one price table (generate.PRICES) for service and eval


def add_common_args(ap):
    ap.add_argument("--data", default=str(ROOT / "data" / "eval_dataset.json"))
    ap.add_argument("--index", default=str(ROOT / "index"))
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--k", type=int, default=5, help="chunks sent to the LLM; also the k used in breakdowns")
    ap.add_argument("--rerank-pool", type=int, default=30, help="hybrid candidates the reranker rescores")
    ap.add_argument("--rerank-model", default=RERANK_MODEL)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M"), help="name for this run's result files")


def load_librarian(a):
    from generate import Librarian
    print("loading index and embedding model...", flush=True)
    return Librarian(a.index)


def load_eval(a):
    data = json.load(open(a.data, encoding="utf-8"))
    if a.split != "all":
        data = [e for e in data if e.get("split") == a.split]
    return data


# ---------- gold matching: a chunk is gold if it is in a gold doc and contains a gold passage ----------
def norm(t):
    """Compare text without markdown decoration, issue role tags ([MEMBER], [USER]) or extra spaces."""
    t = re.sub(r"\[[A-Z_]+\] ", "", t or "")
    t = re.sub(r"[`*_>#]", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def chunk_norm(c):
    t = c["text"]
    if c["type"] == "issue":                   # drop the "Issue #N: title / Labels:" header
        t = re.sub(r"^Issue #\d+: [^\n]*\n(Labels:[^\n]*\n)?\n?", "", t)
    return norm(t)


def doc_index(chunks):
    """doc_id -> [(chunk id, normalised text)], built once per index."""
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append((c["id"], chunk_norm(c)))
    return by_doc


def gold_chunks(e, by_doc):
    """Recomputed from the passages, so the eval also works on an index built with other chunk sizes."""
    ids = set()
    for g in e.get("gold_passages", []):
        p = norm(g["passage"])
        ids |= {i for i, t in by_doc.get(g["doc_id"], []) if p in t}
    return sorted(ids)


class Reranker:
    """Cross-encoder: reads (question, chunk) together and rescores the hybrid candidates."""
    def __init__(self, model):
        from sentence_transformers import CrossEncoder
        from index_corpus import embed_text
        self.model, self.embed_text = CrossEncoder(model), embed_text

    def rerank(self, query, chunks, k):
        scores = self.model.predict([(query, self.embed_text(c)) for c in chunks])
        return [dict(chunks[i], rerank_score=float(scores[i])) for i in np.argsort(-scores)[:k]]


@lru_cache
def get_reranker(model):
    print(f"loading reranker {model}...", flush=True)
    return Reranker(model)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else None


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def md_table(headers, rows):
    """Markdown table, padded so it also reads well in the terminal."""
    rows = [[fmt(v) for v in r] for r in rows]
    w = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    line = lambda cells: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
    return "\n".join([line(headers), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
