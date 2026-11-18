"""
TRAINING (graph construction): build the "teacher-forced" graph for one question

- at inference, the model builds a separate subgraph by retrieving facts
- during training, we build the subgraph from the BFS supervision (graph guaranteed to contain the answer paths)
    - scorers are trained on subgraph from BFS supervision
    - evaluate_inference.py is the realistic evaluation without the "teacher" at test time


- training keeps:
    - every edge that touches an entity to be scored / lies on an answer path ("required" edges)
    - at most `max_edges_per_parent` randomly chosen "filler" edges per expanded
    entity (mirrors the max_facts_per_entity cap used at inference)
- negatives are not disconnected - every candidate keeps its edge(s) to its parent
entities
    - if only positives were connected, the model could cheat by learning "connected => answer" instead of learning from the question
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
    parameters:
        - supervision: one result of build_supervision()/build_supervision_dataset()
            - i.e. BFS distances, answer-path entities, ...
        - training_targets: one result of build_training_targets()
            - i.e. the sampled pull-node and answer candidates with labels
        - max_edges_per_parent: cap on filler edges kept per expanded entity

    returns:
        - QuestionSubgraph containing
            - the topic entity
            - all answer-path entities
            - all candidates
            - shortest-path edges (distance d -> d+1) between them
    """

    subgraph = QuestionSubgraph()

    distances = supervision["distances"]
    topic_id = supervision["topic_entity_id"]
    max_hops = len(supervision["pull_node_labels"])

    # entities to be scored / sit on answer paths
    required = {topic_id}
    required.update(supervision["path_entity_ids"])
    required.update(c["entity_id"] for c in training_targets["pull_candidates"])
    required.update(c["entity_id"] for c in training_targets["answer_candidates"])

    for entity_id in required:
        subgraph.add_entity(entity_id)

    # deterministic filler sampling: the same question gets the same graph in
    # every epoch (the topic ID seeds the generator)
    rng = random.Random(topic_id)

    for parent_id, parent_distance in distances.items():

        # entities at the hop limit have no further children to expand
        if parent_distance >= max_hops:
            continue

        required_edges, filler_edges = [], []

        for traversal in kg.get_facts_for_entity(parent_id):
            child_id = traversal.neighbour_id

            # keep only edges that move one step further from the topic
            # a shortest-path edge: distance d -> d + 1
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