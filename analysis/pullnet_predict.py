"""Save PullNet's ranked answers for every question in a split, in file order.

Same steps as pullnet/evaluate_inference.py (Lily's script), but instead of keeping
only the averages it writes one line per question, so wrong answers can be inspected.
Line i of the output matches line i of MetaQA/<hop>-hop/vanilla/qa_<split>.txt and of
qa_<split>_qtype.txt.

Run from analysis/pullnet_run/ (it holds the kb.txt / hop-folder links and checkpoints/):
    python ../pullnet_predict.py --hop 3 --split test
Writes predictions/hop3_test.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

PULLNET = Path(__file__).resolve().parents[1] / "pullnet"
sys.path.insert(0, str(PULLNET))

import config  # noqa: E402
from data import get_qa_path, load_knowledge_graph, load_qa_dataset  # noqa: E402
from graph import KnowledgeGraph  # noqa: E402
from metrics import average_metrics, ranking_metrics  # noqa: E402
from model_utils import load_checkpoint  # noqa: E402
from pipeline import run_retrieval_pipeline  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--hop", type=int, required=True)
parser.add_argument("--split", default="test", choices=["dev", "test"])
parser.add_argument("--limit", type=int, default=0, help="only the first N questions (for testing)")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

kg = KnowledgeGraph(load_knowledge_graph(config.KB_PATH))
components, vocab, checkpoint = load_checkpoint(config.checkpoint_path(args.hop), kg, args.device)
examples = load_qa_dataset(get_qa_path(config.DATA_DIR, args.hop, args.split))
qtypes = (Path(config.DATA_DIR) / f"{args.hop}-hop" / f"qa_{args.split}_qtype.txt").read_text().splitlines()
assert len(qtypes) == len(examples), "question file and qtype file do not line up"
if args.limit:
    examples, qtypes = examples[:args.limit], qtypes[:args.limit]

out_path = Path("predictions") / f"hop{args.hop}_{args.split}.jsonl"
out_path.parent.mkdir(exist_ok=True)
per_question, started = [], time.time()

with torch.no_grad(), out_path.open("w") as out:
    for line_no, (example, qtype) in enumerate(zip(examples, qtypes), start=1):
        topic_id = kg.entity_id(example["topic_entity"])
        gold_ids = {kg.entity_id(a) for a in example["answers"] if a in kg.entity_to_id}

        tokens, lengths = vocab.encode_batch([example["question"]])
        question_vector = components["question_encoder"](tokens.to(args.device), lengths)
        _, relation_scores = components["relation_scorer"](question_vector)
        result = run_retrieval_pipeline(
            kg=kg,
            topic_entity_id=topic_id,
            question_vector=question_vector,
            relation_scores=relation_scores.squeeze(0),
            graph_encoder=components["graph_encoder"],
            pull_scorer=components["pull_scorer"],
            answer_scorer=components["answer_scorer"],
            max_hops=args.hop,
            max_facts_per_entity=config.MAX_FACTS_PER_ENTITY,
            top_k_pull_nodes=config.TOP_K_PULL_NODES,
        )

        retrieved = result["entity_ids"]
        scores = result["answer_scores"].reshape(-1)
        order = torch.argsort(scores, descending=True).tolist()
        ranked = [retrieved[p] for p in order]
        metrics = ranking_metrics(ranked, gold_ids)
        per_question.append(metrics)

        first_correct = next((rank for rank, e in enumerate(ranked, 1) if e in gold_ids), None)
        out.write(json.dumps({
            "line": line_no,
            "qtype": qtype,
            "question": example["question"],
            "topic": example["topic_entity"],
            "gold": example["answers"],
            "top10": [[kg.entity_name(ranked[i]), round(float(scores[order[i]]), 4)]
                      for i in range(min(10, len(ranked)))],
            "first_correct_rank": first_correct,
            "gold_retrieved": len(gold_ids & set(retrieved)),
            "num_entities": len(retrieved),
            "num_facts": len(result["subgraph"].fact_ids),
            **metrics,
        }) + "\n")

        if line_no % 2000 == 0:
            print(f"  {line_no}/{len(examples)} questions, {time.time() - started:.0f}s", flush=True)

averages = average_metrics(per_question)
print(f"Saved {len(per_question)} questions to {out_path} in {time.time() - started:.0f}s")
print("Averages (should match evaluate_inference.py):",
      {k: round(v, 6) for k, v in averages.items()})
