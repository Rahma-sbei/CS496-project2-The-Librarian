"""Validate the eval dataset against the real corpus and index.

Usage:
    python validate_eval.py data/eval_dataset.json

Checks, per entry:
  1. required fields present and non-empty
  2. unique ids
  3. if the entry has a doc_id (new format): the doc_id exists in the index and the
     source_passage appears verbatim inside a chunk of that document
  4. legacy entries (no doc_id): source_file exists and contains the passage, and the
     passage sits inside some indexed chunk
  5. answerable=false entries must have no source passage
Prints a summary and writes eval_problems.json listing every failing entry.
"""
import json, re, sys
from pathlib import Path

REQUIRED = ["id", "question", "answer", "source_file", "source_passage"]


def norm(t):
    t = re.sub(r"[`*_>#]", "", t or "")           # ignore markdown decoration
    return re.sub(r"\s+", " ", t).strip().lower()


def load_text(corpus_root, source_file):
    p = Path(source_file)
    if not p.exists():
        p = corpus_root.parent / source_file
    if not p.exists():
        return None
    raw = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix == ".json":                        # issues / releases
        try:
            d = json.loads(raw)
            parts = [d.get("title", ""), d.get("body", "")]
            parts += [c.get("body", "") for c in d.get("comments", [])]
            raw = "\n".join(parts)
        except json.JSONDecodeError:
            pass
    return raw


def main(path, corpus="corpus", index="index"):
    data = json.load(open(path, encoding="utf-8"))
    corpus_root = Path(corpus)
    chunks_file = Path(index) / "chunks.jsonl"
    chunk_texts, by_doc = [], {}
    if chunks_file.exists():
        for l in open(chunks_file, encoding="utf-8"):
            c = json.loads(l)
            t = c["text"]
            if c["type"] == "issue":
                t = re.sub(r"^Issue #\d+: [^\n]*\n(Labels:[^\n]*\n)?\n?", "", t)
                t = re.sub(r"\[[A-Z_]+\] ", "", t)
            chunk_texts.append(norm(t))
            by_doc.setdefault(c["doc_id"], []).append(norm(t))
    else:
        print(f"[warn] {chunks_file} not found, skipping the chunk check")

    problems, seen = [], set()
    for e in data:
        errs = []
        unanswerable = e.get("answerable") is False
        for k in (["id", "question", "answer"] if unanswerable else REQUIRED):
            if k not in e or not str(e[k]).strip():
                errs.append(f"missing or empty field: {k}")
        if e.get("id") in seen:
            errs.append("duplicate id")
        seen.add(e.get("id"))

        if unanswerable:
            if str(e.get("source_passage", "")).strip():
                errs.append("unanswerable entry should have an empty source_passage")
        elif not errs and e.get("doc_id"):
            docs = by_doc.get(e["doc_id"])
            if docs is None:
                errs.append(f"doc_id not in index: {e['doc_id']}")
            elif not any(norm(e["source_passage"]) in c for c in docs):
                errs.append("source_passage not found in any chunk of that doc_id")
        elif not errs:
            text = load_text(corpus_root, e["source_file"])
            if text is None:
                errs.append(f"source_file not found: {e['source_file']}")
            elif norm(e["source_passage"]) not in norm(text):
                errs.append("source_passage NOT found verbatim in the source file")
            if chunk_texts and not any(norm(e["source_passage"]) in c for c in chunk_texts):
                errs.append("source_passage not contained in any single indexed chunk")
        if errs:
            problems.append({"id": e.get("id"), "problems": errs})

    n = len(data)
    print(f"{n} entries checked, {n - len(problems)} OK, {len(problems)} with problems")
    kinds = {}
    for p in problems:
        for m in p["problems"]:
            key = m.split(":")[0]
            kinds[key] = kinds.get(key, 0) + 1
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")
    Path("eval_problems.json").write_text(json.dumps(problems, indent=2))
    print("details written to eval_problems.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/eval_dataset.json")
