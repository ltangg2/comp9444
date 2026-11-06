from collections import defaultdict

class Graph:
    def __init__(self):
        self.relation_node_type_mapping = {
            "directed_by": ("movie", "director"),
            "starred_actors": ("movie", "actor"),
            "has_genre": ("movie", "genre"),
            "in_language": ("movie", "language"),
            "release_year": ("movie", "year"),
            "has_tags": ("movie", "tag"),
            "has_imdb_rating": ("movie", "imdb_rating"),
            "written_by": ("movie", "writer"),
            "has_imdb_votes": ("movie", "imdb_votes"),
        }

        node_types = ['movie', 'director', 'actor', 'genre', 'writer',
                    'language', 'year', 'tag', 'imdb_rating']

        self.adj_list = defaultdict(list)
        self.node_types = {t: set() for t in node_types}

    def add_entity(self, u, node_type=None):
        # Add an entity to the graph if it doesn't already exist.
        # If given the type of entity, store it in the node_types dictionary.
        if u not in self.adj_list:
            self.adj_list[u] = []

        if node_type and node_type in self.node_types:
            self.node_types[node_type].add(u)

    def add_relation(self, u, v, relation):
        # Add a relation between two entities in the graph.
        u_type, v_type = self.relation_node_type_mapping[relation]
        u_node = (u_type, u)
        v_node = (v_type, v)

        self.add_entity(u_node, u_type)
        self.add_entity(v_node, v_type)
        self.adj_list[u_node].append((v_node, relation))
        self.adj_list[v_node].append((u_node, relation))

    def get_relation(self, u, v):
        # Get the relation between two entities if it exists.
        for neighbor, relation in self.get_neighbors(u):
            if neighbor == v:
                return relation
        return None

    def get_related_entities(self, node, relation):
        # Get all entities related to u by the specified relation.
        entities = []
        for nbr, rel in self.adj_list.get(node, []):
            if rel == relation:
                entities.append(nbr)

        return entities

    def get_neighbors(self, u):
        return self.adj_list.get(u, [])