
"""
TRAINING-DATA PREPARATION (part 4): turn BFS supervision into labelled examples

- BFS (supervision.py) identifies entities/relations that lie on answer paths
- this file converts that into the binary-classification examples the three scorers are trained on and SAMPLES negatives
    - relation targets = multi-hot vector over the 18 relations (1 = on an answer path)
    - pull candidates = entities at distance d: on-path = 1, others = 0 (sampled)
    - answer candidates = gold answers = 1, other entities within the hop limit = 0
    - samples negatives as there can be a large number of reachable entities within a given hop boundary, with only a small proportion being answers
        - using all reachable entities = high runtime
        - negative_ratio=1 keeps about one negative per positive
        - limitation: the model only sees a subset of negatives in training but must rank every retrieved entity at inference
"""
import random
import torch


def build_training_targets(
    supervision,
    max_hops,
    num_relations,
    negative_ratio=1,
    random_seed=42
):
    """
    convert one question's BFS supervision into training targets

    - creates targets for:
        - relation prediction
        - pull-node prediction at each retrieval iteration
        - answer prediction
    - negative examples are sampled to keep training data manageable

    - parameters:
        - supervision: one result from build_supervision() or build_supervision_dataset()
        - negative_ratio: number of negative examples to sample per positive example
        - random_seed: seed for reproducible negative sampling

    - returns: dictionary containing candidate IDs and binary labels
    """

    rng = random.Random(random_seed)

    distances = supervision["distances"]
    pull_node_labels = supervision["pull_node_labels"]
    relation_labels = supervision["relation_labels"]

    reachable_answers = supervision["reachable_answer_ids"]
    all_answers = supervision["answer_entity_ids"]

    # relation prediction targets

    # current RelationScorer predicts relation relevance from the question, rather than from a specific entity
    # combine the positive relations found across all positively labelled pull nodes and iterations
    positive_relation_ids = set()

    for iteration_labels in relation_labels:
        for relation_ids in iteration_labels.values():
            positive_relation_ids.update(relation_ids)

    relation_targets = torch.zeros(num_relations)

    for relation_id in positive_relation_ids:
        relation_targets[relation_id] = 1.0

    # every relation not in the positive set = treated as a negative relation for this Q
    # weak question-level approximation as a relation may be useful from an entity that is not on
    # one of the shortest paths retained by BFS

    # pull-node prediction targets
    pull_candidates = []

    for iteration_index in range(max_hops):

        positive_ids = set(
            pull_node_labels[iteration_index]
        )

        # candidate entities must be at distance corresponding to the retrieval iteration
        # E.g. iteration 0 expands distance-0 entities (the topic)
        entities_at_distance = {
            entity_id
            for entity_id, distance in distances.items()
            if distance == iteration_index
        }

        # positive entities = entities on a shortest path (i.e. should be expanded at this iteration)
        positives = sorted(
            positive_ids.intersection(entities_at_distance)
        )

        # other entities at the same distance = possible negative candidates
        negative_pool = sorted(
            entities_at_distance - positive_ids
        )

        # topic entity is always expanded initially as there is no useful pull-node classification decision to learn at iteration 0
        # keep topic entity's label but do not sample negatives for this iteration
        if iteration_index == 0:
            negatives = []

        else:
            # sample a limited number of negatives relative to the number of positives
            number_of_negatives = min(
                len(negative_pool),
                negative_ratio * len(positives)
            )

            negatives = rng.sample(
                negative_pool,
                number_of_negatives
            )

        for entity_id in positives:
            pull_candidates.append({
                "entity_id": entity_id,
                "iteration": iteration_index,
                "label": 1.0
            })

        for entity_id in negatives:
            pull_candidates.append({
                "entity_id": entity_id,
                "iteration": iteration_index,
                "label": 0.0
            })

    # answer prediction targets

    # all reachable gold answers = positive answer candidates
    positive_answer_ids = sorted(reachable_answers)

    # candidate negatives = from entities reached by BFS within the hop limit, excluding all gold answers
    # exclude even unreachable gold answers to avoid accidentally labelling a known answer as negative
    negative_answer_pool = sorted(
        set(distances.keys()) - set(all_answers)
    )

    number_of_answer_negatives = min(
        len(negative_answer_pool),
        negative_ratio * len(positive_answer_ids)
    )

    negative_answer_ids = rng.sample(
        negative_answer_pool,
        number_of_answer_negatives
    )

    answer_candidates = []

    for entity_id in positive_answer_ids:
        answer_candidates.append({
            "entity_id": entity_id,
            "label": 1.0
        })

    for entity_id in negative_answer_ids:
        answer_candidates.append({
            "entity_id": entity_id,
            "label": 0.0
        })

    return {
        "topic_entity_id": supervision["topic_entity_id"],

        "relation_targets": relation_targets,
        "positive_relation_ids": positive_relation_ids,

        "pull_candidates": pull_candidates,

        "answer_candidates": answer_candidates,
        "reachable_answer_ids": reachable_answers,

        "unreachable_answers": supervision["unreachable_answers"]
    }


def build_training_targets_dataset(
    supervision_results,
    max_hops,
    num_relations,
    negative_ratio=1,
    random_seed=42
):
    """
    convert a list of supervision results into training targets

    - each question gets a different deterministic random seed
        - negative sampling is reproducible but not identical across all Qs
    """

    training_examples = []

    for index, supervision in enumerate(supervision_results):

        targets = build_training_targets(
            supervision=supervision,
            max_hops=max_hops,
            num_relations=num_relations,
            negative_ratio=negative_ratio,
            random_seed=random_seed + index
        )

        training_examples.append(targets)

    return training_examples