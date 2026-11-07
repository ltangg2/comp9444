from collections import defaultdict

class Graph:
    def __init__(self):
        self.relation_node_type_mapping = {
            "directed_by": ("movie", "person"),
            "starred_actors": ("movie", "person"),
            "has_genre": ("movie", "genre"),
            "in_language": ("movie", "language"),
            "release_year": ("movie", "year"),
            "has_tags": ("movie", "tag"),
            "has_imdb_rating": ("movie", "imdbrating"),
            "written_by": ("movie", "person"),
            "has_imdb_votes": ("movie", "imdbvotes"),
        }

        node_types = [
            'movie', 'person', 'genre', 'language',
            'year', 'tag', 'imdbrating', 'imdbvotes'
        ]

        self.adj_list = defaultdict(list)
        self.node_types = {t: set() for t in node_types}

    def add_entity(self, node_u, node_type=None):
        # Add an entity to the graph if it doesn't already exist.
        # If given the type of entity, store it in the node_types dictionary.
        if u not in self.adj_list:
            self.adj_list[node_u] = []

        if node_type and node_type in self.node_types:
            self.node_types[node_type].add(node_u)

    def add_relation(self, u, v, relation):
        # Add a relation between two entities in the graph.
        u_type, v_type = self.relation_node_type_mapping[relation]
        u_node = (u_type, u)
        v_node = (v_type, v)

        self.add_entity(u_node, u_type)
        self.add_entity(v_node, v_type)
        self.adj_list[u_node].append((v_node, relation))
        self.adj_list[v_node].append((u_node, relation))

    def get_relation(self, node_u, node_v):
        # Get the relation between two entities if it exists.
        for neighbour, relation in self.get_neighbours(node_u):
            if neighbour == node_v:
                return relation
        return None

    def get_related_entities(self, node, relation):
        # Get all entities related to node by the specified relation.
        entities = []
        for nbr, rel in self.adj_list.get(node, []):
            if rel == relation:
                entities.append(nbr)

        return entities

    def get_neighbours(self, node):
        return self.adj_list.get(node, [])