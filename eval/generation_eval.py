"""Generation eval: runs the real pipeline (generate.Librarian.answer) on every question, then scores
correctness, faithfulness, citation accuracy and "I don't know", plus latency and cost per request.

Calls OpenAI (needs OPENAI_API_KEY in .env). Usage (from the repo root):
    python eval/generation_eval.py --limit 5 --tag smoke        # quick, costs a fraction of a cent
    python eval/generation_eval.py --tag baseline               # full run
    python eval/generation_eval.py --rerank --tag rerank        # same questions, reranked context
    python eval/generation_eval.py --tag baseline --resume      # continue an interrupted run

How each number is scored:
  correctness     LLM judge vs the reference answer: yes = 1, partial = 0.5, no = 0.
                  Unanswerable questions score 1 only if the answer is "I don't know".
  faithfulness    the judge splits the answer into claims; share of claims stated by any source sent.
  citation acc.   share of cited claims that the cited source itself states.
  latency         wall time of the whole request (search + rerank + LLM), measured here.
  cost            generation tokens x PRICES in generate.py. Judge cost is reported separately.
Writes results/generation_<tag>.jsonl (one line per question, written as it goes) and
results/generation_<tag>.json (summary).
"""
import argparse, json, time

from common import (RESULTS, add_common_args, cost, doc_index, get_reranker, gold_chunks, load_eval,
                    load_librarian, mean, md_table, pct, save_json)
from generate import LLM_MODEL, format_sources

JUDGE_PROMPT = """You grade answers from a question-answering system about Streamlit.
You get the question, a reference answer, the numbered sources the system was given, and the system's answer.

1. Split the system answer into its factual claims (skip filler such as "Here is how").
2. For each claim give:
   - "cited": the source numbers written right after the claim, e.g. [2] -> [2]; [] if none
   - "supported_by_cited": true only if the cited sources themselves state the claim
   - "supported_by_sources": true if any of the numbered sources states the claim
3. "correct": compare the system answer with the reference answer.
   "yes" = gives the key facts of the reference and nothing that contradicts it,
   "partial" = gives some of the key facts, "no" = wrong, contradicting, or missing the point.

Reply with JSON only:
{"claims": [{"claim": "...", "cited": [1], "supported_by_cited": true, "supported_by_sources": true}],
 "correct": "yes", "reason": "one short sentence"}"""

CORRECT = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def add_args(ap):
    ap.add_argument("--limit", type=int, help="only the first N questions (cheap smoke test)")
    ap.add_argument("--rewrite", action="store_true", help="rewrite each question before searching")
    ap.add_argument("--min-score", type=float, default=0.0, help="generate.py 'I don't know' threshold")
    ap.add_argument("--rerank", action="store_true", help="rerank hybrid results before generating")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--daily-requests", type=int, default=100, help="traffic assumed by the cost model")
    ap.add_argument("--resume", action="store_true", help="keep finished questions of this --tag")


def is_idk(text):
    return text.strip().lower().replace("’", "'").startswith("i don't know")


def ask(lib, e, a):
    t0 = time.perf_counter()
    sources = None
    if a.rerank:
        pool, _ = lib.retrieve(e["question"], k=a.rerank_pool)
        sources = get_reranker(a.rerank_model).rerank(e["question"], pool, a.k)
    res = lib.answer(e["question"], k=a.k, rewrite=a.rewrite, min_score=a.min_score, sources=sources)
    res["latency_s"] = time.perf_counter() - t0          # end to end, including search and rerank
    return res


def judge(lib, e, res, a):
    """One judge call per answered question; returns its verdict and token usage."""
    r = lib.client.chat.completions.create(
        model=a.judge_model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": f"Question: {e['question']}\n\nReference answer: {e['answer']}"
                                              f"\n\nSources:\n{format_sources(res['sources'])}"
                                              f"\n\nSystem answer:\n{res['answer']}"}])
    jc = cost(a.judge_model, r.usage.prompt_tokens, r.usage.completion_tokens)
    try:
        v = json.loads(r.choices[0].message.content)
        claims, verdict = v["claims"], v["correct"]
        if verdict not in CORRECT:
            raise ValueError(verdict)
    except (ValueError, KeyError, TypeError) as err:
        return dict(judge_error=repr(err), judge_cost_usd=jc)
    return dict(claims=claims, verdict=verdict, judge_reason=v.get("reason", ""), judge_cost_usd=jc)


def score(e, res, a, lib, gold):
    idk = is_idk(res["answer"])
    row = dict(id=e["id"], question_type=e["question_type"], source_type=e["source_type"],
               answerable=e["answerable"], split=e["split"], question=e["question"],
               reference=e["answer"], answer=res["answer"], idk=idk, query=res["query"],
               sources=[dict(id=c["id"], title=c["title"], url=c["url"]) for c in res["sources"]],
               context_hit=bool(gold & {c["id"] for c in res["sources"]}),
               citations=res["citations"], bad_citations=res["bad_citations"], best_score=res["best_score"],
               prompt_tokens=res["prompt_tokens"], completion_tokens=res["completion_tokens"],
               cost_usd=cost(res["model"], res["prompt_tokens"], res["completion_tokens"]),
               latency_s=round(res["latency_s"], 3), correct=None, faithfulness=None, n_claims=0,
               n_cited=0, n_cited_ok=0, judge_cost_usd=0.0)
    if not e["answerable"]:
        row["correct"] = 1.0 if idk else 0.0
    elif idk:
        row["correct"] = 0.0                      # refused a question the corpus answers
    else:
        row.update(judge(lib, e, res, a))
        if "claims" in row:
            claims = row["claims"]
            cited = [c for c in claims if c.get("cited")]
            row.update(correct=CORRECT[row["verdict"]], n_claims=len(claims), n_cited=len(cited),
                       n_cited_ok=sum(bool(c.get("supported_by_cited")) for c in cited),
                       faithfulness=(sum(bool(c.get("supported_by_sources")) for c in claims) / len(claims)
                                     if claims else None))
    return row


def summarize(rows, a):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if r["faithfulness"] is not None]
    refusals = [r for r in rows if r["idk"]]
    lat = [r["latency_s"] for r in rows]
    c_req = mean([r["cost_usd"] for r in rows])
    claims, cited = sum(r["n_claims"] for r in answered), sum(r["n_cited"] for r in answered)
    return dict(
        n=len(rows), n_answerable=len(ans), n_unanswerable=len(una),
        judge_errors=sum("judge_error" in r for r in rows),
        accuracy_all=mean([r["correct"] for r in rows]),
        correctness_answerable=mean([r["correct"] for r in ans]),
        faithfulness=mean([r["faithfulness"] for r in answered]),
        fully_faithful=mean([float(r["faithfulness"] == 1.0) for r in answered]),
        citation_accuracy=sum(r["n_cited_ok"] for r in answered) / cited if cited else None,
        claims_with_citation=cited / claims if claims else None,
        answers_with_bad_citation=mean([float(bool(r["bad_citations"])) for r in rows]),
        idk_on_unanswerable=mean([float(r["idk"]) for r in una]),
        idk_on_answerable=mean([float(r["idk"]) for r in ans]),
        idk_precision=sum(not r["answerable"] for r in refusals) / len(refusals) if refusals else None,
        context_recall=mean([float(r["context_hit"]) for r in ans]),
        latency_s_p50=pct(lat, 50), latency_s_p95=pct(lat, 95), latency_s_mean=mean(lat),
        prompt_tokens_mean=mean([r["prompt_tokens"] for r in rows]),
        completion_tokens_mean=mean([r["completion_tokens"] for r in rows]),
        cost_per_request_usd=c_req, cost_per_1k_requests_usd=c_req * 1000,
        cost_per_day_usd=c_req * a.daily_requests, cost_per_day_100x_usd=c_req * a.daily_requests * 100,
        judge_cost_usd=sum(r["judge_cost_usd"] for r in rows),
        by_question_type={t: dict(n=len(g), correctness=mean([r["correct"] for r in g]))
                          for t in sorted({r["question_type"] for r in rows})
                          for g in [[r for r in rows if r["question_type"] == t]]})


def run(a, lib):
    if a.rerank and a.rewrite:
        raise SystemExit("--rerank with --rewrite is not supported yet: the rewrite happens after the search")
    data = load_eval(a)[:a.limit]
    by_doc = doc_index(lib.chunks)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"generation_{a.tag}.jsonl"
    done = {}
    if a.resume and out.exists():
        done = {r["id"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
        print(f"resuming: {len(done)} questions already done", flush=True)

    rows = []
    with open(out, "a" if a.resume else "w", encoding="utf-8") as f:
        for n, e in enumerate(data, 1):
            if e["id"] in done:
                rows.append(done[e["id"]])
                continue
            row = score(e, ask(lib, e, a), a, lib, set(gold_chunks(e, by_doc)))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(f"  [{n}/{len(data)}] {e['id']} correct={row['correct']} faithful={row['faithfulness']} "
                  f"idk={row['idk']} {row['latency_s']:.1f}s", flush=True)

    s = summarize(rows, a)
    config = dict(tag=a.tag, model=LLM_MODEL, k=a.k, rewrite=a.rewrite,
                  min_score=a.min_score, rerank=a.rerank, judge_model=a.judge_model, split=a.split,
                  limit=a.limit, daily_requests=a.daily_requests)
    path = save_json(f"generation_{a.tag}.json", dict(config=config, summary=s))
    return s, report(s, config) + f"\n\nDetails: {out}\nSummary: {path}"


def report(s, cfg):
    variant = ", ".join(f"{k}={cfg[k]}" for k in ("model", "k", "rerank", "rewrite", "min_score", "judge_model"))
    return "\n".join([
        f"## Generation ({s['n']} questions: {s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable; "
        f"split={cfg['split']}; {variant})", "",
        md_table(["metric", "value", "want"], [
            ["accuracy, all questions (Week 8 headline)", s["accuracy_all"], "high"],
            ["correctness, answerable (yes 1 / partial 0.5)", s["correctness_answerable"], "high"],
            ["faithfulness (claims stated by a source)", s["faithfulness"], "1.0"],
            ["answers fully faithful", s["fully_faithful"], "1.0"],
            ["citation accuracy (cited source states it)", s["citation_accuracy"], "1.0"],
            ["claims that carry a citation", s["claims_with_citation"], "1.0"],
            ["answers citing a nonexistent source", s["answers_with_bad_citation"], "0.0"],
            ["'I don't know' on unanswerable", s["idk_on_unanswerable"], "1.0"],
            ["'I don't know' on answerable", s["idk_on_answerable"], "0.0"],
            ["'I don't know' precision", s["idk_precision"], "1.0"],
            ["gold passage in the context sent", s["context_recall"], "1.0"],
            ["judge errors (excluded)", s["judge_errors"], "0"]]),
        "", "Service (latency is end to end; cost is generation only, at PRICES in generate.py):", "",
        md_table(["p50 s", "p95 s", "mean s", "prompt tok", "output tok", "$/request", "$/1k req",
                  f"$/day @ {cfg['daily_requests']}", f"$/day @ {cfg['daily_requests'] * 100} (100x)"],
                 [[s["latency_s_p50"], s["latency_s_p95"], s["latency_s_mean"], round(s["prompt_tokens_mean"]),
                   round(s["completion_tokens_mean"]), f"{s['cost_per_request_usd']:.6f}",
                   s["cost_per_1k_requests_usd"], s["cost_per_day_usd"], s["cost_per_day_100x_usd"]]]),
        "", "Correctness by question type:", "",
        md_table(["question_type", "n", "correctness"],
                 [[t, v["n"], v["correctness"]] for t, v in s["by_question_type"].items()]),
        "", f"Judge cost for this run: ${s['judge_cost_usd']:.4f} (not part of the service cost)"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    add_args(ap)
    a = ap.parse_args()
    _, text = run(a, load_librarian(a))
    print("\n" + text)
