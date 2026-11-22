"""Re-run PullNet on a few chosen test questions and record what it collected at each step.

predictions/hop3_test.jsonl stores each question's ranked answers, but not *which* entities were
explored along the way. For the worked examples in the notebook we also want to see whether a
particular entity (e.g. a film on the path to the answer) was ever reached.

Run from analysis/pullnet_run/ (CPU only, a few seconds):
    python ../pullnet_trace_examples.py --hop 3 --lines 1711 3230 10544 \
        --check "Robert Zemeckis" "Thomas F. Wilson" "Tony Gilroy" 2002 2007 "Steven Spielberg" "The Goonies" "Spanish"
Writes results/hop3_example_traces.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pullnet"))

import config  # noqa: E402
from data import get_qa_path, load_knowledge_graph, load_qa_dataset  # noqa: E402
from graph import KnowledgeGraph  # noqa: E402
from model_utils import load_checkpoint  # noqa: E402
from pipeline import run_retrieval_pipeline  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--hop", type=int, required=True)
parser.add_argument("--lines", type=int, nargs="+", required=True, help="1-based lines of qa_test.txt")
parser.add_argument("--check", nargs="*", default=[], help="entity names to look for in what was collected")
args = parser.parse_args()

kg = KnowledgeGraph(load_knowledge_graph(config.KB_PATH))
components, vocab, _ = load_checkpoint(config.checkpoint_path(args.hop), kg, "cpu")
examples = load_qa_dataset(get_qa_path(config.DATA_DIR, args.hop, "test"))

traces = []
for line in args.lines:
    example = examples[line - 1]
    with torch.no_grad():
        question_vector = components["question_encoder"](*vocab.encode_batch([example["question"]]))
        _, relation_scores = components["relation_scorer"](question_vector)
        result = run_retrieval_pipeline(
            kg=kg,
            topic_entity_id=kg.entity_id(example["topic_entity"]),
            question_vector=question_vector,
            relation_scores=relation_scores.squeeze(0),
            graph_encoder=components["graph_encoder"],
            pull_scorer=components["pull_scorer"],
            answer_scorer=components["answer_scorer"],
            max_hops=args.hop,
            max_facts_per_entity=config.MAX_FACTS_PER_ENTITY,
            top_k_pull_nodes=config.TOP_K_PULL_NODES,
        )
    collected = {kg.entity_name(i) for i in result["entity_ids"]}
    traces.append({
        "line": line,
        "question": example["question"],
        "entities_collected": len(collected),
        # iteration i+1: how many newly found entities could be explored next, and how many were
        "explore_steps": result["retrieval_trace"],
        "collected": {name: name in collected for name in args.check},
    })
    print(traces[-1])

out = Path("results") / f"hop{args.hop}_example_traces.json"
out.write_text(json.dumps(traces, indent=2))
print("Saved", out)
