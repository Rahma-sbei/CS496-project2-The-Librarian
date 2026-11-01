# Week 8 optimisation: before and after

**Target from the brief:** cut the cost per request or the p95 latency by 50%, losing at most 2 points of accuracy on
our own evaluation, measured before and after on the same test set.

**Result: the cost target was reached, but not within the accuracy limit.** Sending 2 passages instead of 5 cuts the
cost per question by 53%, but accuracy falls by 8.9 points. No configuration we measured met both conditions. Below
are the numbers, why it happened, the changes that did save something at no accuracy cost, and what we would try next.

## How we measured

Same 112 questions (100 answerable, 12 unanswerable), same model (`gpt-4o-mini`, temperature 0), same judge, same
scripts. Only the number of passages sent to the model changes. Latency is end to end (search plus the OpenAI call)
on a laptop over the internet, with the chunks in Qdrant Cloud. Reproduce with:

```bash
venv/Scripts/python eval/generation_eval.py --k 5 --tag final
venv/Scripts/python eval/generation_eval.py --k 3 --tag opt-k3
venv/Scripts/python eval/generation_eval.py --k 2 --tag opt-k2
```

## Where the cost is

A question costs $0.000316 with 5 passages: 1,798 input tokens and 78 output tokens. At $0.15 per million input tokens
and $0.60 per million output tokens, **85% of the cost is the input**, and almost all of the input is the passages.
Search costs nothing per question (the embedding models run locally and the vector store is on a free tier). So the lever is: send fewer or
shorter passages. That is the first saving the brief suggests.

## Change 1: fewer passages (measured)

| Passages sent | Input tokens | Cost / question | Δ cost | p50 latency | p95 latency | Δ p95 | Accuracy (all 112) | Δ accuracy | Faithfulness | Gold passage sent |
|---|---|---|---|---|---|---|---|---|---|---|
| **5 (before)** | 1,798 | $0.000316 | – | 1.31 s | 5.04 s | – | 86.2% | – | 98.4% | 77% |
| 3 | 1,062 | $0.000203 | −36% | 1.27 s | 4.35 s | −14% | 78.1% | −8.0 pts | 99.1% | 66% |
| 2 | 733 | $0.000150 | **−53%** | 1.21 s | 2.51 s | −50% | 77.2% | **−8.9 pts** | 99.6% | 59% |

**Why accuracy fell:** the answer quality per passage did not change (faithfulness stays near 99%). What changed is
whether the right passage was sent at all: it is in the top 5 for 77% of questions but in the top 3 for only 66% and
the top 2 for 59%. Our ranking is good at finding the right passage somewhere in the top 5, not at putting it first
(recall@1 is 0.44). With fewer passages, the model more often has nothing to answer from and says "I don't know" on
answerable questions (6% → 15%).

**Latency:** the median barely moves (1.31 s → 1.21 s), because the time is dominated by the OpenAI call, not by the
prompt length. The p95 column looks like a 50% cut with 2 passages, but it is not a real effect: end-to-end p95 is
driven by a handful of slow network responses (one request took 36 s after a retry, nine took over 3 s in the 5-passage
run), while search itself takes 60 to 68 ms. And the accuracy loss rules that configuration out anyway.

**Decision:** the service keeps 5 passages. Saving $0.00017 per question is not worth 9 points of accuracy.

## Improving accuracy through retrieval (measured)

Week 8 asked for a saving; this is the opposite experiment — can we *raise* accuracy by getting the right
passage in front of the model more often? Retrieval sets the ceiling: the gold passage is in the 5 chunks
sent to the model only 77% of the time, even though it is in the top 20 for 93%. Two cheap changes attack
that, measured on the same 112 questions and scored by the independent judge (see the next section):

| Retrieval | Accuracy (all 112) | Gold passage sent | Cost / question | p50 | p95 |
|---|---|---|---|---|---|
| **5 passages (baseline)** | 85.3% | 77% | $0.000317 | 1.27 s | 2.84 s |
| **10 passages** | **89.7%** (+4.4 pts) | 87% | $0.000569 (+79%) | 1.19 s | 2.45 s |
| Rerank top 30 → best 5 | 83.5% (−1.8 pts) | 80% | $0.000289 | 3.07 s | 10.35 s |

**Sending 10 passages instead of 5 is a clear win: +4.4 points of accuracy at no latency cost and about
$0.00025 more per question** (still a third of a hundredth of a cent). Doubling the context puts the gold
passage in front of the model 87% of the time instead of 77%, and the model uses it.

**Reranking does not pay off end to end.** A cross-encoder re-reads the top 30 and keeps the best 5. It does
raise the share of questions with the gold passage sent (77% → 80%), but accuracy *falls* 1.8 points and the
median response time triples (1.27 s → 3.07 s, p95 over 10 s on a laptop CPU). Re-ordering to put one passage
first sometimes drops a second passage the answer also needed, and the latency is not worth it. This confirms,
end to end, the retrieval-only finding that the reranker helps recall@20 and the very top rank, not recall@5.

**Decision:** 10 passages is the accuracy-improving configuration and is now the **deployed default**. It
reverses the Week 8 "keep 5" choice, which was made under a cost target; with accuracy as the goal it is the
better trade. The default is set in `app.py` (`LIBRARIAN_K` falls back to 10) and in `render.yaml`.

## A more credible grader

The generation metrics come from an LLM judge. At first the judge was `gpt-4o-mini` — the same model that
writes the answers, which invites self-preference bias. We re-graded every run with an independent, stronger
model, `gpt-4.1-mini`, and the headline holds:

| | gpt-4o-mini judge (self) | gpt-4.1-mini judge (independent) |
|---|---|---|
| Baseline accuracy (all 112) | 86.2% | 85.3% |
| Faithfulness | 98.4% | 98.4% |

The independent judge is marginally harsher (−0.9 points) but agrees closely, so the reported accuracy is not
an artefact of a model grading its own answers. `gpt-4.1-mini` is now the default judge in
`eval/generation_eval.py`. The judge runs only during evaluation, never in the live service, so it adds no cost
or latency to answering a question (about $0.11 to grade a full 112-question run).

## Choices that save time or money

| Change | Saving | Accuracy cost |
|---|---|---|
| **Skip the reranker.** A cross-encoder over the top 30 would add about 3.5 s per question on a CPU (p50 search time 60 ms → 3,526 ms) for three points of recall@5 (0.77 → 0.80). | ~3.5 s per question avoided, more than twice the whole current p50 | lowers answer accuracy by 1.8 points end to end (see "Improving accuracy through retrieval") |
| **Cache repeated questions** (`app.py`). An identical question (same wording, options and filter) is answered from memory: no search, no model call. | 100% of the cost and almost all the latency of a repeat | none (same answer) |
| **Local embeddings and a free vector store.** Both embedding models run on our own CPU through ONNX (no embedding API), and the chunks live on Qdrant Cloud's free tier. | $0 per question for search | none |

The cache cannot be measured on the evaluation set (every question is different); its effect depends on how often
real users repeat questions, which we can read from `logs/requests-*.jsonl` once the service is online.

## What we would try next

1. **Better ranking at the top, then fewer passages.** If the right passage were reliably in the top 2, sending 2 would
   keep accuracy. The reranker already lifts recall@3 from 0.66 to 0.75; a faster reranker (or running it only when
   the first search is not confident, as the brief suggests) could make 3 passages viable.
2. **Shorter passages instead of fewer.** Keep all 5 retrieved chunks but send only their paragraphs most similar to
   the question. This keeps the 77% recall of the top 5 while cutting the input roughly in half.
3. **A cheaper model.** `gpt-4.1-nano` costs a third less per token than `gpt-4o-mini`; it needs its own accuracy run.
