from graph import Graph
from baseline_preprocess import Preprocess
from baseline_question_classifer import Question_Classifer
import re
import sys
from collections import Counter

class Baseline:
    """
    Baseline model for COMP9444 project.
    """
    def __init__(self, graph, question_classifier):
        self.graph = graph
        self.question_classifier = question_classifier

    def evaluate(self, questions, answers):
        # Evaluate the model's performance on a set of questions and their
        # corresponding answers, returning the exact-match accuracy.
        if len(questions) == 0:
            return 0

        correct = 0

        for question, true_answers in zip(questions, answers):
            predicted_answers = self.predict(question)

            if set(predicted_answers) == set(true_answers):
                correct += 1

        accuracy = correct / len(questions)

        return accuracy

    def predict(self, question):
        # Predict the answer to a given question by classifying it and
        # traversing the graph.
        entities, node_path, relation_path = self.classify_question(question)
        answers = self.answer_question(entities, node_path, relation_path)
        return answers

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
            "tag" if t == "tags" else t
            for t in question_type.split("_to_")
        ]

        role_to_relation = {
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

        relation_path = []

        for source_type, target_type in zip(role_path, role_path[1:]):
            role = target_type if source_type == "movie" else source_type
            relation_path.append(role_to_relation[role])

        node_path = [
            "person" if t in {"actor", "director", "writer"} else t
            for t in role_path
        ]

        return relation_path, node_path

    def answer_question(self, entities, node_path, relation_path):
        # Given the starting entities and relation path, traverse the graph to
        # find the answer entities.
        paths = [
            ((node_path[0], entity), None)
            for entity in entities
        ]

        for relation, target_type in zip(relation_path, node_path[1:]):
            next_paths = []

            for current_node, previous_node in paths:
                nbrs = self.graph.get_related_entities(
                    current_node, relation
                )

                for nbr in nbrs:
                    if nbr[0] == target_type and nbr != previous_node:
                        next_paths.append((nbr, current_node))

            paths = next_paths

            if not paths:
                break

        return {node[1] for node, previous_node in paths}

if __name__ == "__main__":
    print("\n=== Preprocessing ===")
    preprocess = Preprocess()
    preprocess.preprocess()

    graph = preprocess.get_graph()
    train_qs, train_qs_types, train_ans = preprocess.get_data("train")
    test_qs, test_qs_types, test_ans = preprocess.get_data("test")
    valid_qs, valid_qs_types, valid_ans = preprocess.get_data("validate")
    questions = preprocess.get_questions()
    answers = preprocess.get_answers()

    print("\n=== Question Classifier ===")
    question_classifier = Question_Classifer()
    question_classifier.train(train_qs, train_qs_types)
    print("=== Validate Data ===")
    question_classifier.evaluate(valid_qs, valid_qs_types)
    print("=== Test Data ===")
    question_classifier.evaluate(test_qs, test_qs_types)

    print("\n=== Baseline ===")
    baseline = Baseline(graph, question_classifier)

    for hop in range(1, 4):
        print(f"\t=== {hop}-hop ===")
        for folder in ["vanilla", "ntm"]:
            hop_qs = questions["test"][hop][folder]
            hop_ans = answers["test"][hop][folder]
            accuracy = baseline.evaluate(hop_qs, hop_ans)
            print(f"\t{folder} data's accuracy: {accuracy:.2%}")

