# TO UPDATE COMMENTS

"""
TRAINING (graph construction): build the "teacher-forced" graph for one question.

Why this exists
---------------
At inference the model builds its own subgraph by retrieving facts. But early in
training the model is random, so its own retrieval would be garbage and the
scorers would have nothing sensible to learn from.

"Teacher forcing" solves this: during training we build the subgraph FROM THE
BFS SUPERVISION instead (a graph guaranteed to contain the answer paths) and
train the scorers on it. At test time the model has to cope without the teacher,
which is why evaluate_inference.py is the realistic evaluation.

What changed from the first version (and why it matters for run time)
---------------------------------------------------------------------
The first version kept EVERY shortest-path edge in the whole 2-hop neighbourhood
of the topic. Because MetaQA has hub entities (genre "Drama" links to 4,300
movies), that was ~2,000 entities / 2,000 facts per question (measured), and each
fact was processed in a Python loop. This version keeps:

  * every edge that touches an entity we need to score or that lies on an
    answer path ("required" edges), plus
  * at most `max_edges_per_parent` randomly chosen "filler" edges per expanded
    entity (mirrors the max_facts_per_entity cap used at inference).

Negatives are NOT disconnected: every candidate keeps its edge(s) to its parent
entities. That matters: if only positives were connected, the model could cheat
by learning "connected => answer" instead of learning from the question.
"""

import random

from retrieval import QuestionSubgraph
import config


def build_training_subgraph(
    kg,
    supervision,
    training_targets,
    max_edges_per_parent=config.TRAIN_MAX_EDGES_PER_PARENT,
):
    """
    Args:
        supervision: one result of build_supervision()/build_supervision_dataset()
            (BFS distances, answer-path entities, ...).
        training_targets: one result of build_training_targets() (the sampled
            pull-node and answer candidates with labels).
        max_edges_per_parent: cap on filler edges kept per expanded entity.

    Returns:
        A QuestionSubgraph containing the topic entity, all answer-path entities,
        all candidates, and shortest-path edges (distance d -> d+1) between them.
    """

    subgraph = QuestionSubgraph()

    distances = supervision["distances"]
    topic_id = supervision["topic_entity_id"]
    max_hops = len(supervision["pull_node_labels"])

    # Entities we must be able to score or that sit on answer paths.
    required = {topic_id}
    required.update(supervision["path_entity_ids"])
    required.update(c["entity_id"] for c in training_targets["pull_candidates"])
    required.update(c["entity_id"] for c in training_targets["answer_candidates"])

    for entity_id in required:
        subgraph.add_entity(entity_id)

    # Deterministic filler sampling: the same question gets the same graph in
    # every epoch (the topic ID seeds the generator).
    rng = random.Random(topic_id)

    for parent_id, parent_distance in distances.items():

        # Entities at the hop limit have no further children to expand.
        if parent_distance >= max_hops:
            continue

        required_edges, filler_edges = [], []

        for traversal in kg.get_facts_for_entity(parent_id):
            child_id = traversal.neighbour_id

            # Keep only edges that move one step further from the topic
            # (a shortest-path edge: distance d -> d + 1).
            if distances.get(child_id) != parent_distance + 1:
                continue

            if child_id in required:
                required_edges.append(traversal)
            else:
                filler_edges.append(traversal)

        if len(filler_edges) > max_edges_per_parent:
            filler_edges = rng.sample(filler_edges, max_edges_per_parent)

        for traversal in required_edges + filler_edges:
            subgraph.add_fact(traversal.fact_id)
            subgraph.add_traversal(parent_id, traversal)
            subgraph.add_entity(traversal.neighbour_id)

    return subgraph
