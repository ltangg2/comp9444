"""
DATA PREPARATION (part 2): convert KB triples into a traversable graph

- Neural networks work on integers and BFS runs multiple lookups to find
  reachable nodes from a given entity
    - convert strings (E.g. entities, relations and facts) into IDs

- KB stores triples in a specific direction (E.g. (Kismet, directed_by, William Dieterle))
  but q can ask about the inverse relation (E.g. which movies did the director of [Kismet] direct?)
    - convert 9 original relations into 18 directed relations (9 original + 9 inverse)
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    fact_id: int
    subject_id: int
    relation_id: int
    object_id: int


@dataclass(frozen=True)
class FactTraversal:
    fact_id: int
    relation_id: int
    neighbour_id: int


class KnowledgeGraph:
    """
    Each original KB triple is one fact node.
    Entity IDs: [0, num_entities)
    Fact graph-node IDs: [num_entities, num_entities + num_facts)
    Incidence role 0=subject, 1=object.
    Inverse relation labels are traversal labels, not duplicate facts.
    """
    def __init__(self, triples):
        self.triples = list(triples)

        # collect all entity and relation names
        entity_names, original_relations = set(), set()
        for subject, relation, obj in self.triples:
            entity_names.update((subject, obj))
            original_relations.add(relation)

        # assign ID to each entity (sort for IDs to be reproducible)
        self.entity_to_id = {
            name: idx for idx, name in enumerate(sorted(entity_names))
        }
        self.id_to_entity = {
            idx: name for name, idx in self.entity_to_id.items()
        }

        # assign ID to each relation and inverse relation (sort for IDs to be reproducible)
        relation_names = set(original_relations)
        relation_names.update(r + "__inv" for r in original_relations)
        self.relation_to_id = {
            name: idx for idx, name in enumerate(sorted(relation_names))
        }
        self.id_to_relation = {
            idx: name for name, idx in self.relation_to_id.items()
        }

        self.num_entities = len(self.entity_to_id)
        self.num_relations = len(self.relation_to_id)
        self.num_facts = len(self.triples)
        self.num_graph_nodes = self.num_entities + self.num_facts

        # build facts and adjacency lists
        self.facts = []

        # form a list of FactTraversal using entity ID
        self.entity_to_facts = defaultdict(list)
        self.incidence_edges = []
        self._fact_tensors = None  # built lazily by fact_tensors()

        for fact_id, (subject, relation, obj) in enumerate(self.triples):
            s_id = self.entity_to_id[subject]
            o_id = self.entity_to_id[obj]
            r_id = self.relation_to_id[relation]
            r_inv_id = self.relation_to_id[relation + "__inv"]

            self.facts.append(Fact(fact_id, s_id, r_id, o_id))
            self.incidence_edges.append((s_id, fact_id, 0))
            self.incidence_edges.append((o_id, fact_id, 1))

            # forward direction relation - from subject, follow fact to object
            self.entity_to_facts[s_id].append(
                FactTraversal(fact_id, r_id, o_id)
            )

            # inverse direction relation - from object, follow fact to subject
            self.entity_to_facts[o_id].append(
                FactTraversal(fact_id, r_inv_id, s_id)
            )

    # @property
    # def num_graph_nodes(self):
    #     return self.num_entities + self.num_facts

    def entity_id(self, name):
        try:
            return self.entity_to_id[name]
        except KeyError as exc:
            raise KeyError(f"Entity not found in KB: {name!r}") from exc

    def entity_name(self, entity_id):
        return self.id_to_entity[entity_id]

    def relation_name(self, relation_id):
        return self.id_to_relation[relation_id]

    def fact_node_id(self, fact_id):
        if not 0 <= fact_id < self.num_facts:
            raise IndexError(f"Invalid fact ID: {fact_id}")
        return self.num_entities + fact_id

    def get_facts_for_entity(self, entity_id):
        return self.entity_to_facts.get(entity_id, [])

    def fact_tensors(self):
        """
        Return three LongTensors (subject_ids, relation_ids, object_ids), each of
        length num_facts, where position i describes fact i.

        Built once and cached. The graph encoder uses them to look up thousands of
        facts in one vectorised operation instead of one Python call per fact.
        (torch is imported here so the rest of this file stays pure Python.)
        """
        if self._fact_tensors is None:
            import torch
            self._fact_tensors = (
                torch.tensor([f.subject_id for f in self.facts], dtype=torch.long),
                torch.tensor([f.relation_id for f in self.facts], dtype=torch.long),
                torch.tensor([f.object_id for f in self.facts], dtype=torch.long),
            )
        return self._fact_tensors

    def get_fact(self, fact_id):
        return self.facts[fact_id]

    def describe_fact(self, fact_id):
        return self.triples[fact_id]

    def to_adjacency_list(self):
        """entity ID -> list of (relation ID, neighbour ID) """
        adjacency = defaultdict(list)
        for entity_id, traversals in self.entity_to_facts.items():
            adjacency[entity_id].extend(
                (item.relation_id, item.neighbour_id)
                for item in traversals
            )
        return adjacency

    def validate(self):
        expected = 2 * self.num_facts
        return {
            "num_entities": self.num_entities,
            "num_relations_including_inverse": self.num_relations,
            "num_facts": self.num_facts,
            "num_graph_nodes": self.num_graph_nodes,
            "num_incidence_edges": len(self.incidence_edges),
            "expected_incidence_edges": expected,
            "incidence_count_correct": len(self.incidence_edges) == expected,
        }

# QUESTIONS
# i took my initial file and added in some of your changes
    # are the initial imports still needed?
    # am i able to just change class Fact and class FactTraversal to what you provided (i.e. no more @dataClass and have the actual def init)? i dont want to change knowledgeGraph
    # commented out @property and fact_tensor() will this work? after removing them what i have is basically what you provided?
# how does this work? it allows us to specify what to run in command line? pass hops=(2,) to avoid loading unnecessary 1-hop and 3-hop files
# remove sanity check?