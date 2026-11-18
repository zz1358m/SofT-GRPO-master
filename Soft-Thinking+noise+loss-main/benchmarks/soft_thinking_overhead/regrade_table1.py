"""Re-grade saved outputs with the paper's Table 1 grader (matheval.evaluator_map[dataset].rule_judge, math-verify).

Usage: python regrade_table1.py <repo Soft-Thinking+noise+loss-main dir> <out22 dir>
Adds "correct_table1" and "extracted_table1" to every row of out22/<model>/<dataset>/seed*/<run>.jsonl (in place).
"""
import glob
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
import matheval  # noqa: E402

DATASET_KEY = {"aime": "aime2024", "math": "math500"}
root = sys.argv[2]
for p in sorted(glob.glob(os.path.join(root, "*", "*", "seed*", "*.jsonl"))):
    dataset = p.split(os.sep)[-3]
    ev = matheval.evaluator_map[DATASET_KEY[dataset]]
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if rows and all("correct_table1" in r for r in rows):
        continue
    n_ok = 0
    for r in rows:
        try:
            ok, ext = ev.rule_judge(r["text"], str(r["gold"]), True)
        except Exception as e:  # math-verify timeouts etc. count as wrong, as in the paper's loop
            ok, ext = False, f"error: {type(e).__name__}"
        r["correct_table1"] = bool(ok)
        r["extracted_table1"] = str(ext)
        n_ok += bool(ok)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{p}: {n_ok}/{len(rows)} = {n_ok / max(len(rows), 1):.3f}", flush=True)
