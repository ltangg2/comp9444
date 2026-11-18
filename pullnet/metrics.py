"""
ranking metrics used by both evaluation scripts

given the model's ranked list of entities for one question and the set of gold
answers:
    - Hits@K = 1 if at least one gold answer is in the top K, else 0
        - Hits@1 = headline MetaQA number (the fairest one to compare
            against the baseline)
    - MRR = 1 / (rank of the best-ranked gold answer)
        - 0 if none was ranked
    - Recall@K = fraction of ALL gold answers that appear in the top K
        - Questions with many answers cannot reach Recall@1 = 1
"""

METRIC_NAMES = [
    "hits@1", "hits@5", "hits@10", "mrr",
    "recall@1", "recall@5", "recall@10",
]


def ranking_metrics(ranked_entity_ids, gold_entity_ids):
    gold = set(gold_entity_ids)

    # 1-based ranks of every gold answer that appears in the ranking
    gold_ranks = [
        rank for rank, entity_id in enumerate(ranked_entity_ids, start=1)
        if entity_id in gold
    ]
    best_rank = min(gold_ranks) if gold_ranks else None

    metrics = {"mrr": 1.0 / best_rank if best_rank else 0.0}
    for k in (1, 5, 10):
        metrics[f"hits@{k}"] = float(best_rank is not None and best_rank <= k)
        metrics[f"recall@{k}"] = len(gold & set(ranked_entity_ids[:k])) / len(gold)
    return metrics


def average_metrics(per_question_metrics, names=METRIC_NAMES):
    n = len(per_question_metrics)
    return {name: sum(m[name] for m in per_question_metrics) / n for name in names}
