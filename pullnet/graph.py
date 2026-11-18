"""
DATA PREPARATION (part 2): convert KB triples into a traversable graph

- neural networks work on integers and BFS runs multiple lookups to find
  reachable nodes from a given entity
    - convert strings (E.g. entities, relations and facts) into IDs

- KB stores triples in a specific direction (E.g. (Kismet, directed_by, William Dieterle))
  but q can ask about the inverse relation (E.g. which movies did the director of [Kismet] direct?)
    - convert 9 original relations into 18 directed relations (9 original + 9 inverse)
"""
from collections import defaultdict


class Fact:
    """
    one KB triple stored as integer IDs

    - fact_id: position of the triple in kb.txt
    - subject_id, relation_id, object_id: IDs of the three parts of the triple
    """

    def __init__(self, fact_id, subject_id, relation_id, object_id):
        self.fact_id = fact_id
        self.subject_id = subject_id
        self.relation_id = relation_id
        self.object_id = object_id


class FactTraversal:
    """
    one way of stepping along a fact, starting from one of its two entities

    - fact_id: the fact being followed
    - relation_id: direction-aware relation (original relation or its __inv twin)
    - neighbour_id: the entity reached after following the fact
    """


    def __init__(self, fact_id, relation_id, neighbour_id):
        self.fact_id = fact_id
        self.relation_id = relation_id
        self.neighbour_id = neighbour_id


class KnowledgeGraph:
    """
    each original KB triple is one fact

    - entity IDs: [0, num_entities)
    - fact graph-node IDs: [num_entities, num_entities + num_facts)
    - incidence role: 0 = subject, 1 = object
    - inverse relation labels are traversal labels, not duplicate facts
    """


    def __init__(self, triples):
        self.triples = list(triples)

        # collect all entity and relation names
        entity_names = set()
        original_relations = set()
        for subject, relation, obj in self.triples:
            entity_names.add(subject)
            entity_names.add(obj)
            original_relations.add(relation)

        # assign ID to each entity (sort for IDs to be reproducible)
        self.entity_to_id = {}
        for idx, name in enumerate(sorted(entity_names)):
            self.entity_to_id[name] = idx
        self.id_to_entity = {}
        for name, idx in self.entity_to_id.items():
            self.id_to_entity[idx] = name

        # assign ID to each relation and inverse relation (sort for IDs to be reproducible)
        relation_names = set(original_relations)
        for relation in original_relations:
            relation_names.add(relation + "__inv")

        self.relation_to_id = {}
        for idx, name in enumerate(sorted(relation_names)):
            self.relation_to_id[name] = idx
        self.id_to_relation = {}
        for name, idx in self.relation_to_id.items():
            self.id_to_relation[idx] = name

        self.num_entities = len(self.entity_to_id)
        self.num_relations = len(self.relation_to_id)
        self.num_facts = len(self.triples)
        self.num_graph_nodes = self.num_entities + self.num_facts

        self.facts = []

        # entity ID -> list of FactTraversal (everything we can step along from that entity)
        self.entity_to_facts = defaultdict(list)

        # (entity ID, fact ID, role) where role 0 = subject and 1 = object
        self.incidence_edges = []

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


    def entity_id(self, name):
        if name not in self.entity_to_id:
            raise KeyError(f"Entity not found in KB: {name!r}")
        return self.entity_to_id[name]


    def entity_name(self, entity_id):
        return self.id_to_entity[entity_id]


    def relation_name(self, relation_id):
        return self.id_to_relation[relation_id]


    def fact_node_id(self, fact_id):
        if not 0 <= fact_id < self.num_facts:
            raise IndexError(f"Invalid fact ID: {fact_id}")
        return self.num_entities + fact_id


    def get_facts_for_entity(self, entity_id):
        """all traversals leaving this entity (empty list if it has none)"""
        return self.entity_to_facts.get(entity_id, [])


    def get_fact(self, fact_id):
        return self.facts[fact_id]


    def describe_fact(self, fact_id):
        return self.triples[fact_id]


    def to_adjacency_list(self):
        """entity ID -> list of (relation ID, neighbour ID)"""
        adjacency = defaultdict(list)
        for entity_id, traversals in self.entity_to_facts.items():
            for item in traversals:
                adjacency[entity_id].append((item.relation_id, item.neighbour_id))
        return adjacency


    def validate(self):
        """sanity check - every fact should give exactly two incidence edges"""
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