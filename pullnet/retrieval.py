# TO UPDATE COMMENTS

"""
RETRIEVAL: the question-specific mini-graph, and how facts get added to it.

Why this exists
---------------
The full KB has ~135k facts; we cannot run a neural network over all of it for
every question. PullNet's key idea is to start at the topic entity and PULL IN
only a small, relevant slice of the KB. QuestionSubgraph is the container for
that slice, and retrieve_facts() is the step that grows it.

Used at inference time (pipeline.py). Training uses a teacher-forced version
of the same container (training_utils.py).
"""

from dataclasses import dataclass, field

import torch


@dataclass
class QuestionSubgraph:
    """
    The part of the KB retrieved for ONE question.

    The full KnowledgeGraph is never modified; this object only records which
    entity IDs and fact IDs have been selected.
    """

    entity_ids: set = field(default_factory=set)
    fact_ids: set = field(default_factory=set)

    # Each traversal is (source_entity_id, fact_id, relation_id, neighbour_id).
    # relation_id is direction-aware (it may be an __inv relation).
    traversals: list = field(default_factory=list)

    # Set of the keys already stored in `traversals`, so duplicate checks are
    # O(1). (The earlier version rebuilt a set from the whole list on every
    # call, which became quadratic for large graphs.)
    _traversal_keys: set = field(default_factory=set, repr=False)

    def add_entity(self, entity_id):
        self.entity_ids.add(entity_id)

    def add_fact(self, fact_id):
        self.fact_ids.add(fact_id)

    def add_traversal(self, source_entity_id, traversal):
        """Record that `traversal` was followed away from `source_entity_id`."""
        key = (
            source_entity_id,
            traversal.fact_id,
            traversal.relation_id,
            traversal.neighbour_id,
        )
        if key not in self._traversal_keys:
            self._traversal_keys.add(key)
            self.traversals.append(key)


def retrieve_facts(
    kg,
    subgraph,
    entity_ids,
    relation_scores,
    max_facts_per_entity=10,
):
    """
    Expand each entity in `entity_ids`: fetch its highest-scoring facts and add
    them (and the entities they lead to) to the subgraph.

    "Highest-scoring" uses the RelationScorer's output: each fact is scored by
    the question-relevance of the relation used to traverse it. So for
    "who directed [X]?" the `directed_by` facts are preferred over `has_tags`.

    Args:
        relation_scores: 1-D tensor, one score per direction-aware relation.
        max_facts_per_entity: cap on facts taken per expanded entity. This is
            what stops hub entities (e.g. the genre "Drama", 4,300 facts) from
            flooding the subgraph. It is also a recall limit: if the answer is
            not among the top-k facts, it is never retrieved.

    Returns:
        (subgraph, new_entity_ids)
    """
    new_entity_ids = set()
    relation_scores = relation_scores.detach()  # retrieval is a discrete choice

    for entity_id in entity_ids:

        candidates = kg.get_facts_for_entity(entity_id)
        if not candidates:
            continue

        # Score every candidate traversal in ONE tensor operation.
        relation_ids = torch.tensor(
            [t.relation_id for t in candidates],
            dtype=torch.long,
            device=relation_scores.device,
        )
        scores = relation_scores[relation_ids]

        k = min(max_facts_per_entity, len(candidates))
        top_positions = torch.topk(scores, k=k).indices.tolist()

        for position in top_positions:
            traversal = candidates[position]

            subgraph.add_fact(traversal.fact_id)
            subgraph.add_traversal(entity_id, traversal)

            neighbour_id = traversal.neighbour_id
            if neighbour_id not in subgraph.entity_ids:
                new_entity_ids.add(neighbour_id)
            subgraph.add_entity(neighbour_id)

    return subgraph, new_entity_ids
