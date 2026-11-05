"""Summarise evaluation results.
Usage: uv run python analyze_results.py file1.json [file2.json ...]
(With no file names it reads every eval_results_*.json in the folder.)"""
import glob
import json
import sys
from collections import defaultdict

paths = sys.argv[1:] or sorted(glob.glob("eval_results_*.json"))
if not paths:
    raise SystemExit("No result files found. Pass the file name, e.g. eval_results_xxx.json")

data = []
for p in paths:
    data += json.load(open(p, encoding="utf-8"))

scores = defaultdict(list)
passed = defaultdict(int)
for case in data:
    for m in case.get("metrics", []):
        if m.get("score") is not None:
            scores[m["name"]].append(m["score"])
            passed[m["name"]] += 1 if m.get("passed") else 0

total = len(data)
complete = sum(1 for c in data if len(c.get("metrics", [])) >= 5)
fully_passed = sum(1 for c in data if c.get("success") is True)

print(f"\nFile(s): {', '.join(paths)}")
print(f"Questions in file: {total}   Fully scored (all 5 metrics): {complete}   "
      f"Passed all metrics: {fully_passed}\n")
print(f"{'Metric':<24}{'Mean':>7}{'Passed':>12}{'Scored':>9}")
order = ["Contextual Precision", "Contextual Recall", "Contextual Relevancy",
         "Answer Relevancy", "Faithfulness"]
for name in order + [n for n in scores if n not in order]:
    vals = scores.get(name)
    if not vals:
        continue
    print(f"{name:<24}{sum(vals)/len(vals):>7.2f}{passed[name]:>8}/{len(vals):<3}{len(vals):>6}")

if complete < total:
    print(f"\nNote: {total - complete} question(s) were only partly scored; "
          "each metric's mean uses only the cases scored for that metric.")

print("\n--- Failed metrics and the judge's reasons ---")
for i, case in enumerate(data, 1):
    for m in case.get("metrics", []):
        if not m.get("passed"):
            print(f"\nQ{i}: {case['input'][:90]}")
            print(f"  {m['name']} = {m.get('score')}")
            print(f"  Reason: {(m.get('reason') or '')[:300]}")