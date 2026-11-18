"""
RETRIEVAL: the question-specific mini-graph, and how facts get added to it

- ~135k facts in full KB (cannot run a NN over all of it for every Q)
    - start at topic entity and pull the relevant slice of the KB
    - QuestionSubgraph = container for the KB segment
    - retrieve_facts = grows the selected KB segment

- this file is used in pipeline.py
    - training uses a teacher-forced version of the same container (training_utils.py)
"""

import torch


class QuestionSubgraph:
    """
    selects the part of the KB retrieved for one Q

    - records which entity IDs and fact IDs have been selected
    - full KG is not modified
    """


    def __init__(self):
        self.entity_ids = set()
        self.fact_ids = set()

        # each traversal is (source_entity_id, fact_id, relation_id, neighbour_id)
        self.traversals = []

        # the same traversals in a set to quickly check if a key has already been stored
        self.traversal_keys = set()


    def add_entity(self, entity_id):
        self.entity_ids.add(entity_id)


    def add_fact(self, fact_id):
        self.fact_ids.add(fact_id)


    def add_traversal(self, source_entity_id, traversal):
        """record that `traversal` was followed away from `source_entity_id`"""
        key = (
            source_entity_id,
            traversal.fact_id,
            traversal.relation_id,
            traversal.neighbour_id,
        )
        if key not in self.traversal_keys:
            self.traversal_keys.add(key)
            self.traversals.append(key)


def retrieve_facts(
    kg,
    subgraph,
    entity_ids,
    relation_scores,
    max_facts_per_entity=10,
):
    """
    expand each entity in `entity_ids`
    - fetch its highest-scoring facts and add them (and the entities they lead to) to the subgraph
    - "highest-scoring" uses the RelationScorer's output:
        - each fact is scored by the question-relevance of the relation used to traverse it
        - E.g. for "who directed [X]?" the `directed_by` facts are preferred over `has_tags`

    parameters:
        - relation_scores:
            - 1-D tensor, one score per direction-aware relation
        - max_facts_per_entity:
            - cap on facts taken per expanded entity
            - this stops hub entities (e.g. the genre "Drama", 4,300 facts) from
            flooding the subgraph
            - is also a recall limit (e.g. answer is not retrieved if it is not among the top-k facts)

    returns:
        - (subgraph, new_entity_ids)
    """
    new_entity_ids = set()
    relation_scores = relation_scores.detach()

    for entity_id in entity_ids:

        candidates = kg.get_facts_for_entity(entity_id)
        if not candidates:
            continue

        # score every candidate traversal in one tensor operation
        relation_ids = torch.tensor(
            [t.relation_id for t in candidates],
            dtype=torch.long,
            device=relation_scores.device,
        )
        scores = relation_scores[relation_ids]

        # keep the top-k highest scoring facts
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