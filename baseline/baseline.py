from baseline.graph import Graph
from baseline.baseline_question_classifer import Question_Classifer
import re

class Baseline:
    """
    Baseline model for COMP9444 project.
    """

    ROLE_TO_RELATION = {
        "director": "directed_by",
        "writer": "written_by",
        "actor": "starred_actors",
        "genre": "has_genre",
        "language": "in_language",
        "year": "release_year",
        "tag": "has_tags",
        "imdbrating": "has_imdb_rating",
        "imdbvotes": "has_imdb_votes",
    }

    PERSON_ROLES = {"director", "writer", "actor"}

    def __init__(self, graph, question_classifier):
        self.graph = graph
        self.question_classifier = question_classifier

    def evaluate(self, questions, answers):
        # Evaluate the model's performance on a set of questions and their
        # corresponding answers, returning the exact-match accuracy.
        if not questions:
            return 0.0

        n_correct = 0
        for question, true_answers in zip(questions, answers):
            if set(self.predict(question)) == set(true_answers):
                n_correct += 1

        return n_correct / len(questions)

    def predict(self, question):
        # Predict the answer to a given question by classifying it and
        # traversing the graph.
        entities, node_path, relation_path = self.classify_question(question)
        return self.answer_question(entities, node_path, relation_path)

    def classify_question(self, question):
        # Identify the type of question and extract the starting entities and
        # relation path.
        entities = self.extract_entities(question)
        question_type = self.question_classifier.predict(question)
        relation_path, node_path = self.get_relation_path(question_type)

        return entities, node_path, relation_path

    def extract_entities(self, question):
        # Extract entities from the question using regex.
        return re.findall(r'\[([^\]]+)\]', question)

    def get_relation_path(self, question_type):
        # Given the question type, remove the "_to_" from the string and obtain
        # the relation path.
        role_path = [
            "tag" if role == "tags" else role
            for role in question_type.split("_to_")
        ]

        node_path = [
            "person" if role in self.PERSON_ROLES else role
            for role in role_path
        ]

        relation_path = []

        for source_role, target_role in zip(role_path, role_path[1:]):
            if source_role == "movie":
                role = target_role
            elif target_role == "movie":
                role = source_role

            relation_path.append(self.ROLE_TO_RELATION[role])

        return relation_path, node_path

    def answer_question(self, entities, node_path, relation_path):
        # Given the starting entities and relation path, traverse the graph to
        # find the answer entities.
        paths = [
            (None, None, (node_path[0], entity))
            for entity in entities
        ]

        for rel, target in zip(relation_path, node_path[1:]):
            next_paths = []

            for prev, prev_rel, curr in paths:
                nbrs = self.valid_nbrs(curr, prev, prev_rel, rel, target)
                next_paths.extend((curr, rel, nbr) for nbr in nbrs)

            paths = next_paths

        return {curr[1] for _, _, curr in paths}

    def valid_nbrs(self, curr, prev, prev_rel, relation, target):
        # Fetches all related entities and filter them to ensure the entities
        # have not been visited in previous path.
        nbrs = self.graph.get_related_entities(curr, relation)

        valid_nbrs = []
        for nbr in nbrs:
            if nbr[0] == target and not (nbr == prev and relation == prev_rel):
                valid_nbrs.append(nbr)

        return valid_nbrs
