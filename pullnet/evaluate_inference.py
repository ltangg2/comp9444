# TO UPDATE COMMENTS

"""
EVALUATION (the one to report): end-to-end inference with the model's OWN retrieval.

No BFS teacher is used here. For each question the model:
  encodes the question -> scores relations -> retrieves facts around the topic
  -> picks entities to expand -> retrieves again -> ranks everything it retrieved.

Metrics (see metrics.py): Hits@1/5/10, MRR, Recall@1/5/10, plus
  answer_coverage  fraction of gold answers that were retrieved AT ALL. Because an
                   answer that was never retrieved can never be ranked, this is an
                   upper bound on Hits@K and tells you whether failures come from
                   retrieval or from ranking.

Examples
--------
  python evaluate_inference.py --split dev  --num-examples 500    # tune settings
  python evaluate_inference.py --split test                       # final number
  python evaluate_inference.py --split dev --num-examples 500 --sweep

Use dev while choosing retrieval settings; touch test only for the final result.
"""

import argparse
import json
import random
from pathlib import Path

import torch

import config
from data import load_knowledge_graph, load_all_datasets
from graph import KnowledgeGraph
from model_utils import load_checkpoint
from pipeline import run_retrieval_pipeline
from metrics import ranking_metrics, average_metrics, METRIC_NAMES

parser = argparse.ArgumentParser()
parser.add_argument("--hop", type=int, default=config.HOP)
parser.add_argument("--split", default="dev", choices=["dev", "test"])
parser.add_argument("--num-examples", type=int, default=0, help="0 = whole split")
parser.add_argument("--max-facts", type=int, default=config.MAX_FACTS_PER_ENTITY)
parser.add_argument("--top-k", type=int, default=config.TOP_K_PULL_NODES)
parser.add_argument("--sweep", action="store_true",
                    help="Compare several retrieval settings instead of one.")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output", default=None, help="Optional JSON file for the results.")
args = parser.parse_args()

HOP = args.hop
DEVICE = torch.device(args.device)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Load KB, model, data
# ---------------------------------------------------------------------------
kg = KnowledgeGraph(load_knowledge_graph(config.KB_PATH))
components, vocab, checkpoint = load_checkpoint(config.checkpoint_path(HOP), kg, DEVICE)
print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, dev loss {checkpoint['dev_loss']:.4f})")

examples = list(load_all_datasets(config.DATA_DIR, hops=(HOP,))[HOP][args.split])
random.Random(0).shuffle(examples)           # unbiased subset if --num-examples is used
if args.num_examples:
    examples = examples[:args.num_examples]
print(f"Evaluating {len(examples)} {args.split} questions ({HOP}-hop)")

# ---------------------------------------------------------------------------
# Encode each question once (identical for every retrieval setting)
# ---------------------------------------------------------------------------
prepared, skipped, unmatched_answers = [], 0, 0

with torch.no_grad():
    for example in examples:
        try:
            topic_id = kg.entity_id(example["topic_entity"])
        except KeyError:
            skipped += 1
            continue

        gold_ids = set()
        for name in example["answers"]:
            try:
                gold_ids.add(kg.entity_id(name))
            except KeyError:
                unmatched_answers += 1
        if not gold_ids:
            skipped += 1
            continue

        tokens, lengths = vocab.encode_batch([example["question"]])
        question_vector = components["question_encoder"](tokens.to(DEVICE), lengths)
        _, relation_scores = components["relation_scorer"](question_vector)

        prepared.append({
            "question": example["question"],
            "topic_id": topic_id,
            "gold_ids": gold_ids,
            "question_vector": question_vector,
            "relation_scores": relation_scores.squeeze(0),
        })

print(f"Prepared {len(prepared)} | skipped {skipped} | unmatched gold answers {unmatched_answers}")
if not prepared:
    raise RuntimeError("No questions could be prepared.")


# ---------------------------------------------------------------------------
# Evaluate one retrieval setting
# ---------------------------------------------------------------------------
def evaluate(setting):
    per_question, coverage, entities, facts = [], [], [], []

    with torch.no_grad():
        for i, ex in enumerate(prepared, start=1):
            out = run_retrieval_pipeline(
                kg=kg,
                topic_entity_id=ex["topic_id"],
                question_vector=ex["question_vector"],
                relation_scores=ex["relation_scores"],
                graph_encoder=components["graph_encoder"],
                pull_scorer=components["pull_scorer"],
                answer_scorer=components["answer_scorer"],
                max_hops=HOP,
                max_facts_per_entity=setting["max_facts"],
                top_k_pull_nodes=setting["top_k"],
            )

            retrieved = out["entity_ids"]
            order = torch.argsort(out["answer_scores"].reshape(-1), descending=True)
            ranked = [retrieved[p] for p in order.tolist()]

            per_question.append(ranking_metrics(ranked, ex["gold_ids"]))
            coverage.append(len(ex["gold_ids"] & set(retrieved)) / len(ex["gold_ids"]))
            entities.append(len(retrieved))
            facts.append(len(out["subgraph"].fact_ids))

            if i % 200 == 0:
                print(f"  {i}/{len(prepared)} questions", flush=True)

    averages = average_metrics(per_question)
    averages["answer_coverage"] = sum(coverage) / len(coverage)
    averages["avg_entities"] = sum(entities) / len(entities)
    averages["avg_facts"] = sum(facts) / len(facts)
    return averages


settings = (
    [
        {"name": "default",          "max_facts": 10, "top_k": 10},
        {"name": "more facts",       "max_facts": 20, "top_k": 10},
        {"name": "many facts",       "max_facts": 50, "top_k": 10},
        {"name": "fewer expansions", "max_facts": 10, "top_k": 5},
    ]
    if args.sweep
    else [{"name": "run", "max_facts": args.max_facts, "top_k": args.top_k}]
)

all_results = {}
for setting in settings:
    print(f"\n=== {setting['name']}: max_facts={setting['max_facts']}, top_k={setting['top_k']} ===")
    all_results[setting["name"]] = {"setting": setting, "metrics": evaluate(setting)}

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print(f"END-TO-END EVALUATION  |  {HOP}-hop {args.split}  |  {len(prepared)} questions")
print("=" * 92)
print(f"{'setting':<18}{'Hits@1':>8}{'Hits@5':>8}{'Hits@10':>9}{'MRR':>8}"
      f"{'Cover':>8}{'Ents':>8}{'Facts':>8}")
print("-" * 92)
for name, r in all_results.items():
    m = r["metrics"]
    print(f"{name:<18}{m['hits@1']:>8.3f}{m['hits@5']:>8.3f}{m['hits@10']:>9.3f}"
          f"{m['mrr']:>8.3f}{m['answer_coverage']:>8.3f}"
          f"{m['avg_entities']:>8.0f}{m['avg_facts']:>8.0f}")
print("=" * 92)
print("Cover = answer coverage: the ceiling for Hits@K set by retrieval alone.")

if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"hop": HOP, "split": args.split, "n": len(prepared),
                   "results": all_results}, f, indent=2)
    print("Saved results to", args.output)
