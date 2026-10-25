# Week 8 optimisation: before and after

**Target from the brief:** cut the cost per request or the p95 latency by 50%, losing at most 2 points of accuracy on
our own evaluation, measured before and after on the same test set.

**Result: the cost target was reached, but not within the accuracy limit.** Sending 2 passages instead of 5 cuts the
cost per question by 53%, but accuracy falls by 10.3 points. No configuration we measured met both conditions. Below
are the numbers, why it happened, the changes that did save something at no accuracy cost, and what we would try next.

## How we measured

Same 112 questions (100 answerable, 12 unanswerable), same model (`gpt-4o-mini`, temperature 0), same judge, same
scripts. Only the number of passages sent to the model changes. Latency is end to end (search plus the OpenAI call)
on a laptop over the internet. Reproduce with:

```bash
venv/Scripts/python eval/generation_eval.py --k 5 --tag final
venv/Scripts/python eval/generation_eval.py --k 3 --tag opt-k3
venv/Scripts/python eval/generation_eval.py --k 2 --tag opt-k2
```

## Where the cost is

A question costs $0.000305 with 5 passages: 1,733 input tokens and 76 output tokens. At $0.15 per million input tokens
and $0.60 per million output tokens, **85% of the cost is the input**, and almost all of the input is the passages.
Search costs nothing per question (the embedding model and both indexes run locally). So the lever is: send fewer or
shorter passages. That is the first saving the brief suggests.

## Change 1: fewer passages (measured)

| Passages sent | Input tokens | Cost / question | Δ cost | p50 latency | p95 latency | Δ p95 | Accuracy (all 112) | Δ accuracy | Faithfulness | Gold passage sent |
|---|---|---|---|---|---|---|---|---|---|---|
| **5 (before)** | 1,733 | $0.000305 | – | 1.34 s | 2.34 s | – | 86.6% | – | 98.5% | 80% |
| 3 | 1,038 | $0.000197 | −35% | 1.19 s | 2.42 s | +3% | 80.4% | −6.2 pts | 99.0% | 69% |
| 2 | 708 | $0.000144 | **−53%** | 1.21 s | 2.12 s | −9% | 76.3% | **−10.3 pts** | 98.8% | 59% |

**Why accuracy fell:** the answer quality per passage did not change (faithfulness stays near 99%). What changed is
whether the right passage was sent at all: it is in the top 5 for 80% of questions but in the top 3 for only 69% and
the top 2 for 59%. Our ranking is good at finding the right passage somewhere in the top 5, not at putting it first
(recall@1 is 0.38). With fewer passages, the model more often has nothing to answer from and says "I don't know" on
answerable questions (5% → 16%).

**Latency:** it barely moves (p95 −9% at best), because the time is dominated by the OpenAI call, not by the prompt
length. Cutting p95 latency by 50% is not reachable this way.

**Decision:** the service keeps 5 passages. Saving $0.00016 per question is not worth 10 points of accuracy.

## Changes that saved something without losing accuracy

| Change | Saving | Accuracy cost |
|---|---|---|
| **Skip the reranker.** A cross-encoder over the top 30 would add about 2.1 s per question on a CPU (p50 search time 41 ms → 2,098 ms) and does not raise recall@5 (0.80 → 0.79). | ~2 s per question avoided, more than the whole current p50 | none (measured on the retrieval evaluation) |
| **Cache repeated questions** (`app.py`). An identical question (same wording, options and filter) is answered from memory: no search, no model call. | 100% of the cost and almost all the latency of a repeat | none (same answer) |
| **Local embeddings and flat-file index.** No embedding API and no database server. | $0 per question for search | none |

The cache cannot be measured on the evaluation set (every question is different); its effect depends on how often
real users repeat questions, which we can read from `logs/requests-*.jsonl` once the service is online.

## What we would try next

1. **Better ranking at the top, then fewer passages.** If the right passage were reliably in the top 2, sending 2 would
   keep accuracy. The reranker already lifts recall@3 from 0.69 to 0.74; a faster reranker (or running it only when
   the first search is not confident, as the brief suggests) could make 3 passages viable.
2. **Shorter passages instead of fewer.** Keep all 5 retrieved chunks but send only their paragraphs most similar to
   the question. This keeps the 80% recall of the top 5 while cutting the input roughly in half.
3. **A cheaper model.** `gpt-4.1-nano` costs a third less per token than `gpt-4o-mini`; it needs its own accuracy run.
