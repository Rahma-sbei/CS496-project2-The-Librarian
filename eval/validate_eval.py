"""Validate the eval dataset against the chunks stored in the vector store.

Usage (from the repo root):
    python eval/validate_eval.py                                   # data/eval_dataset.json against the collection
    python eval/validate_eval.py data/eval_dataset.json --collection librarian_c150

Checks, per entry:
  1. required fields present and non-empty, unique ids, known question_type and split
  2. answerable entries: every gold passage appears verbatim (ignoring markdown decoration and whitespace) inside a
     single chunk of its own gold document, and doc_id is the first gold document
  3. unanswerable entries have no gold passages
Structural only: it cannot tell whether a passage really supports the answer, so review by hand.
Prints a summary and writes eval_problems.json listing every failing entry.
"""
import argparse, json
from collections import Counter
from pathlib import Path

from common import ROOT, doc_index, gold_chunks
import store

REQUIRED = ["id", "category", "question_type", "question", "answer", "split"]
ANSWERABLE = ["doc_id", "source_type", "url", "source_file", "source_passage", "gold_doc_ids", "gold_passages"]
TYPES = {"how-to", "conceptual", "multi-part", "error-message", "version", "unanswerable"}


def check(e, by_doc, seen):
    errs = [f"missing or empty field: {k}" for k in REQUIRED if not str(e.get(k) or "").strip()]
    if e.get("id") in seen:
        errs.append("duplicate id")
    seen.add(e.get("id"))
    if e.get("question_type") not in TYPES:
        errs.append(f"unknown question_type: {e.get('question_type')}")
    if e.get("split") not in ("dev", "test"):
        errs.append(f"split must be dev or test: {e.get('split')}")

    if e.get("answerable") is False:
        if e.get("gold_passages") or str(e.get("source_passage") or "").strip():
            errs.append("unanswerable entry should have no gold passages")
        return errs
    errs += [f"missing or empty field: {k}" for k in ANSWERABLE if not e.get(k)]
    if errs:
        return errs
    for g in e["gold_passages"]:
        if g["doc_id"] not in by_doc:
            errs.append(f"doc_id not in the vector store: {g['doc_id']}")
        elif not gold_chunks(dict(gold_passages=[g]), by_doc):
            errs.append(f"passage not found in any chunk of {g['doc_id']}: {g['passage'][:60]!r}")
    if e["gold_doc_ids"][0] != e["doc_id"] or {g["doc_id"] for g in e["gold_passages"]} != set(e["gold_doc_ids"]):
        errs.append("doc_id / gold_doc_ids / gold_passages disagree")
    return errs


def main(path, collection):
    data = json.load(open(path, encoding="utf-8"))
    by_doc, seen, problems = doc_index(store.all_chunks(collection)), set(), []
    for e in data:
        errs = check(e, by_doc, seen)
        if errs:
            problems.append({"id": e.get("id"), "problems": errs})

    n, ans = len(data), sum(e.get("answerable") is not False for e in data)
    print(f"{n} entries ({ans} answerable, {n - ans} unanswerable), {n - len(problems)} OK, "
          f"{len(problems)} with problems (collection {collection}, {store.count(collection)} chunks)")
    for field in ("question_type", "source_type", "split"):
        print(f"  {field}: {dict(Counter(str(e.get(field)) for e in data))}")
    kinds = Counter(m.split(":")[0] for p in problems for m in p["problems"])
    for k, v in kinds.most_common():
        print(f"  {v:4d}  {k}")
    Path("eval_problems.json").write_text(json.dumps(problems, indent=2))
    print("details written to eval_problems.json")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(ROOT / "data" / "eval_dataset.json"))
    ap.add_argument("--collection", default=store.COLLECTION)
    a = ap.parse_args()
    main(a.path, a.collection)
