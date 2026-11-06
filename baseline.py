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

    def test(self, questions, answers):
        # Evaluate the model's performance on a set of questions and their
        # corresponding answers, returning the exact-match accuracy.
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
        node_path = question_type.split("_to_")
        relation_path = []

        for source_type, target_type in zip(node_path, node_path[1:]):
            for relation, (u_type, v_type) in self.graph.relation_node_type_mapping.items():
                if ((source_type, target_type) == (u_type, v_type) or
                    (source_type, target_type) == (v_type, u_type)):
                    relation_path.append(relation)
                    break

        return relation_path, node_path

    def answer_question(self, entities, node_path, relation_path):
        # Given the starting entities and relation path, traverse the graph to
        # find the answer entities.
        curr_entities = set()
        for entity in entities:
            curr_entities.add((node_path[0], entity))

        for relation, target_type in zip(relation_path, node_path[1:]):
            next_entities = set()

            for entity in curr_entities:
                nbrs = self.graph.get_related_entities(entity, relation)

                for nbr in nbrs:
                    if nbr[0] == target_type:
                        next_entities.add(nbr)

            curr_entities = next_entities

            if not curr_entities:
                break

        return {entity_name for _, entity_name in curr_entities}

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
    question_classifier.test(test_qs, test_qs_types)

    print("\n=== Baseline ===")
    baseline = Baseline(graph, question_classifier)
    print("Training Data:")
    for hop in range(1, 4):
        accuracies = []
        for folder in ["vanilla", "ntm", "audio"]:
            q_set = questions["train"][hop][folder]
            a_set = answers["train"][hop][folder]
            accuracy = baseline.test(q_set, a_set)
            accuracies.append(accuracy)

        print(f"{hop}-hop:")
        print(f"\tvanilla: {accuracies[0]} | ntm: {accuracies[1]} | audio: {accuracies[2]}")
