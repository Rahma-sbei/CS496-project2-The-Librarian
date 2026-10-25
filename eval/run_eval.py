"""Run the whole evaluation and print one summary: retrieval + ablation, then generation.

Usage (from the repo root):
    python eval/run_eval.py --tag baseline               # everything (generation calls OpenAI)
    python eval/run_eval.py --skip-generation            # retrieval and ablation only, free
    python eval/run_eval.py --limit 10 --tag smoke       # cheap end-to-end check
    python eval/run_eval.py --split test --tag final     # held-out questions only

All options of retrieval_eval.py and generation_eval.py are accepted (see --help).
Writes results/summary_<tag>.md (paste it into the report) next to the per-script result files.
"""
import argparse

import generation_eval, retrieval_eval
from common import RESULTS, add_common_args, load_librarian

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    retrieval_eval.add_args(ap)
    generation_eval.add_args(ap)
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    a = ap.parse_args()

    lib = load_librarian(a)
    parts = [f"# Evaluation summary: {a.tag}"]
    if not a.skip_retrieval:
        print("\nretrieval eval...", flush=True)
        parts.append(retrieval_eval.run(a, lib)[1])
    if not a.skip_generation:
        print("\ngeneration eval...", flush=True)
        parts.append(generation_eval.run(a, lib)[1])

    text = "\n\n".join(parts)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"summary_{a.tag}.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text + f"\n\nSaved: {RESULTS / f'summary_{a.tag}.md'}")
