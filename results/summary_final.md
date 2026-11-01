# Evaluation summary: final

## Retrieval (100 answerable questions, split=all, chunk_size=250, overlap=40)

Ablation (chunk-level recall unless marked doc):

| mode                | R@1   | R@3   | R@5   | R@10  | R@20  | doc R@5 | MRR@10 | p50 ms | p95 ms |
|---------------------|-------|-------|-------|-------|-------|---------|--------|--------|--------|
| keyword only (BM25) | 0.320 | 0.560 | 0.720 | 0.830 | 0.890 | 0.820   | 0.475  | 47.146 | 52.158 |
| vector only         | 0.470 | 0.670 | 0.700 | 0.820 | 0.850 | 0.790   | 0.580  | 55.474 | 60.509 |
| hybrid (RRF)        | 0.440 | 0.660 | 0.770 | 0.870 | 0.930 | 0.830   | 0.578  | 60.058 | 68.053 |
| hybrid + reranker   | 0.440 | 0.750 | 0.800 | 0.880 | 0.930 | 0.850   | 0.605  | 3,526  | 3,625  |

Recall@5 by source_type:

| source_type | n  | keyword only (BM25) | vector only | hybrid (RRF) | hybrid + reranker |
|-------------|----|---------------------|-------------|--------------|-------------------|
| doc         | 86 | 0.698               | 0.686       | 0.744        | 0.791             |
| issue       | 11 | 1.000               | 0.909       | 1.000        | 1.000             |
| release     | 3  | 0.333               | 0.333       | 0.667        | 0.333             |

Recall@5 by question_type:

| question_type | n  | keyword only (BM25) | vector only | hybrid (RRF) | hybrid + reranker |
|---------------|----|---------------------|-------------|--------------|-------------------|
| conceptual    | 17 | 0.647               | 0.765       | 0.765        | 0.765             |
| error-message | 13 | 0.923               | 0.692       | 0.846        | 0.846             |
| how-to        | 47 | 0.681               | 0.660       | 0.723        | 0.766             |
| multi-part    | 8  | 0.750               | 1.000       | 1.000        | 1.000             |
| version       | 15 | 0.733               | 0.600       | 0.733        | 0.800             |

'I don't know' threshold (generate.py --min-score) vs best vector score:

| min_score | unanswerable refused (want 1) | answerable refused (want 0) |
|-----------|-------------------------------|-----------------------------|
| 0.550     | 0.000                         | 0.000                       |
| 0.600     | 0.000                         | 0.000                       |
| 0.650     | 0.167                         | 0.000                       |
| 0.700     | 0.417                         | 0.040                       |
| 0.750     | 0.917                         | 0.150                       |
| 0.800     | 1.000                         | 0.420                       |

Details: C:\Users\Rahma\Desktop\CS496-project2-The-Librarian\results\retrieval_final.json

## Generation (112 questions: 100 answerable, 12 unanswerable; split=all; model=gpt-4o-mini, k=5, rerank=False, rewrite=False, min_score=0.0, judge_model=gpt-4o-mini)

| metric                                        | value | want |
|-----------------------------------------------|-------|------|
| accuracy, all questions (Week 8 headline)     | 0.862 | high |
| correctness, answerable (yes 1 / partial 0.5) | 0.845 | high |
| faithfulness (claims stated by a source)      | 0.984 | 1.0  |
| answers fully faithful                        | 0.957 | 1.0  |
| citation accuracy (cited source states it)    | 0.996 | 1.0  |
| claims that carry a citation                  | 0.889 | 1.0  |
| answers citing a nonexistent source           | 0.000 | 0.0  |
| 'I don't know' on unanswerable                | 1.000 | 1.0  |
| 'I don't know' on answerable                  | 0.060 | 0.0  |
| 'I don't know' precision                      | 0.667 | 1.0  |
| gold passage in the context sent              | 0.770 | 1.0  |
| judge errors (excluded)                       | 0     | 0    |

Service (latency is end to end; cost is generation only, at PRICES in generate.py):

| p50 s | p95 s | mean s | prompt tok | output tok | $/request | $/1k req | $/day @ 100 | $/day @ 10000 (100x) |
|-------|-------|--------|------------|------------|-----------|----------|-------------|----------------------|
| 1.313 | 5.044 | 1.995  | 1798       | 78         | 0.000316  | 0.316    | 0.032       | 3.162                |

Correctness by question type:

| question_type | n  | correctness |
|---------------|----|-------------|
| conceptual    | 17 | 0.941       |
| error-message | 13 | 0.923       |
| how-to        | 47 | 0.777       |
| multi-part    | 8  | 1.000       |
| unanswerable  | 12 | 1.000       |
| version       | 15 | 0.800       |

Judge cost for this run: $0.0392 (not part of the service cost)

Details: C:\Users\Rahma\Desktop\CS496-project2-The-Librarian\results\generation_final.jsonl
Summary: C:\Users\Rahma\Desktop\CS496-project2-The-Librarian\results\generation_final.json
