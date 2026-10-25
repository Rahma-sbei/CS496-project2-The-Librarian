"""Retrieval eval and ablation: recall@k for keyword only, vector only, hybrid, and hybrid + reranker.

No LLM calls, so it is free to run. Usage (from the repo root):
    python eval/retrieval_eval.py                         # all four modes, whole dataset
    python eval/retrieval_eval.py --split test            # held-out questions only
    python eval/retrieval_eval.py --index index_small     # chunk-size experiment on another index

Hit rule (fixed): a question is a hit at k if any of the top-k chunks contains one of its gold passages
(gold chunks are recomputed from gold_passages for the index being tested, so re-chunked indexes work).
Doc recall is the looser version: any top-k chunk comes from one of the gold_doc_ids.
Writes results/retrieval_<tag>.json and prints the summary.
"""
import argparse, json, time

from common import (add_common_args, doc_index, get_reranker, gold_chunks, load_eval, load_librarian,
                    md_table, pct, save_json)

KS = (1, 3, 5, 10, 20)
MODES = {"bm25": "keyword only (BM25)", "vector": "vector only", "hybrid": "hybrid (RRF)",
         "rerank": "hybrid + reranker"}
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def add_args(ap):
    ap.add_argument("--modes", default="bm25,vector,hybrid,rerank", help="comma list of " + ", ".join(MODES))


def search(lib, a, query, mode, k):
    if mode == "rerank":                       # rescore a bigger hybrid pool, keep the best k
        pool, _ = lib.retrieve(query, k=max(a.rerank_pool, k))
        return get_reranker(a.rerank_model).rerank(query, pool, k)
    return lib.retrieve(query, k=k, mode=mode)[0]


def run(a, lib):
    data = load_eval(a)
    by_doc = doc_index(lib.chunks)
    gold = {e["id"]: set(gold_chunks(e, by_doc)) for e in data if e["answerable"]}
    no_gold = sorted(i for i, g in gold.items() if not g)
    if no_gold:                                # a passage split across two chunks of this index
        print(f"  warning: {len(no_gold)} questions have no gold chunk in this index: {no_gold}", flush=True)
    questions = [e for e in data if e["answerable"]]
    modes = a.modes.split(",")
    for m in modes:                            # warm up: first calls load kernels and models
        search(lib, a, "warm up", m, max(KS))

    summary, per_q = {}, []
    for m in modes:
        hits = {k: 0 for k in KS}
        doc_hits = {k: 0 for k in KS}
        rr, lat = 0.0, []
        for e in questions:
            t = time.perf_counter()
            got = search(lib, a, e["question"], m, max(KS))
            lat.append(time.perf_counter() - t)
            ids, docs = [c["id"] for c in got], [c["doc_id"] for c in got]
            first = next((r for r, i in enumerate(ids, 1) if i in gold[e["id"]]), None)
            for k in KS:
                hits[k] += first is not None and first <= k
                doc_hits[k] += any(d in e["gold_doc_ids"] for d in docs[:k])
            rr += 1 / first if first and first <= 10 else 0
            per_q.append(dict(id=e["id"], mode=m, question_type=e["question_type"],
                              source_type=e["source_type"], gold=sorted(gold[e["id"]]),
                              retrieved=ids[:10], first_hit_rank=first))
        n = len(questions)
        summary[m] = {**{f"recall@{k}": hits[k] / n for k in KS},
                      **{f"doc_recall@{k}": doc_hits[k] / n for k in KS},
                      "mrr@10": rr / n,
                      "latency_ms_p50": pct(lat, 50) * 1000, "latency_ms_p95": pct(lat, 95) * 1000}
        print(f"  {MODES[m]:<22} recall@{a.k} = {summary[m][f'recall@{a.k}']:.3f}", flush=True)

    # "I don't know" signal: generate.py refuses when the best vector score is below --min-score
    best = {e["id"]: lib.retrieve(e["question"], k=1, mode="vector")[1] for e in data}
    ans = [best[e["id"]] for e in data if e["answerable"]]
    una = [best[e["id"]] for e in data if not e["answerable"]]
    sweep = [dict(min_score=t, unanswerable_refused=sum(s < t for s in una) / len(una) if una else None,
                  answerable_refused=sum(s < t for s in ans) / len(ans)) for t in THRESHOLDS]

    result = dict(config=dict(json.load(open(f"{a.index}/config.json")), split=a.split,
                              n_questions=len(questions), n_unanswerable=len(una), no_gold_in_index=no_gold,
                              rerank_model=a.rerank_model, rerank_pool=a.rerank_pool, tag=a.tag),
                  summary=summary, idk_threshold_sweep=sweep, per_question=per_q)
    path = save_json(f"retrieval_{a.tag}.json", result)
    return result, report(result, modes, a.k, per_q) + f"\n\nDetails: {path}"


def report(result, modes, k, per_q):
    s, cfg = result["summary"], result["config"]
    out = [f"## Retrieval ({cfg['n_questions']} answerable questions, split={cfg['split']}, "
           f"chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']})", "",
           "Ablation (chunk-level recall unless marked doc):", "",
           md_table(["mode", "R@1", "R@3", "R@5", "R@10", "R@20", "doc R@5", "MRR@10", "p50 ms", "p95 ms"],
                    [[MODES[m]] + [s[m][f"recall@{x}"] for x in KS] +
                     [s[m]["doc_recall@5"], s[m]["mrr@10"], s[m]["latency_ms_p50"], s[m]["latency_ms_p95"]]
                     for m in modes])]
    for field in ("source_type", "question_type"):
        groups = sorted({r[field] for r in per_q})
        rows = []
        for g in groups:
            row = [g, sum(1 for r in per_q if r[field] == g and r["mode"] == modes[0])]
            for m in modes:
                rs = [r for r in per_q if r[field] == g and r["mode"] == m]
                row.append(sum(bool(r["first_hit_rank"] and r["first_hit_rank"] <= k) for r in rs) / len(rs))
            rows.append(row)
        out += ["", f"Recall@{k} by {field}:", "", md_table([field, "n"] + [MODES[m] for m in modes], rows)]
    out += ["", "'I don't know' threshold (generate.py --min-score) vs best vector score:", "",
            md_table(["min_score", "unanswerable refused (want 1)", "answerable refused (want 0)"],
                     [[r["min_score"], r["unanswerable_refused"], r["answerable_refused"]]
                      for r in result["idk_threshold_sweep"]])]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    add_args(ap)
    a = ap.parse_args()
    _, text = run(a, load_librarian(a))
    print("\n" + text)
