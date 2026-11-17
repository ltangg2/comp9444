# UPDATE COMMENTS BELOW

"""

Collect every results/*.json written by evaluate_inference.py (--output ...) into
one comparison table, and save it as results/summary.csv for the report.

  python summarise_results.py
  python summarise_results.py results/hop2_test_20k.json results/hop2_test_full.json

Each row = one evaluation run (file + retrieval setting). Compare rows only when
they use the same split (test) and the same hop.
"""
import csv
import glob
import json
import sys
from pathlib import Path

COLUMNS = ["hits@1", "hits@5", "hits@10", "mrr", "recall@1", "recall@5",
           "recall@10", "answer_coverage", "avg_entities", "avg_facts"]

paths = sys.argv[1:] or sorted(glob.glob("results/*.json"))
if not paths:
    sys.exit("No results/*.json files found. Run evaluate_inference.py with --output first.")

rows = []
for path in paths:
    data = json.load(open(path))
    for name, result in data["results"].items():
        s, m = result["setting"], result["metrics"]
        rows.append({
            "file": Path(path).name, "hop": data["hop"], "split": data["split"],
            "questions": data["n"], "max_facts": s["max_facts"], "top_k": s["top_k"],
            **{c: m[c] for c in COLUMNS},
        })

print(f"{'file':<26}{'hop':>4}{'split':>6}{'n':>7}{'facts':>6}"
      f"{'Hits@1':>8}{'Hits@5':>8}{'MRR':>7}{'Rec@5':>7}{'Cover':>7}")
print("-" * 86)
for r in rows:
    print(f"{r['file']:<26}{r['hop']:>4}{r['split']:>6}{r['questions']:>7}{r['max_facts']:>6}"
          f"{r['hits@1']:>8.3f}{r['hits@5']:>8.3f}{r['mrr']:>7.3f}"
          f"{r['recall@5']:>7.3f}{r['answer_coverage']:>7.3f}")

Path("results").mkdir(exist_ok=True)
with open("results/summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("\nSaved results/summary.csv")