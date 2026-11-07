import re
import numpy as np
import base64

from baseline.graph import Graph

class Preprocess:
    N_HOPS = 3

    def __init__(self):
        self.graph = Graph()
        self.splits = ["train", "dev", "test"]
        self.folders = ["vanilla", "ntm"]
        self.relations = set()

        self.qs_types = {
            data_type: {hop: [] for hop in range(1, self.N_HOPS + 1)}
            for data_type in self.splits
        }

        self.questions = {
            split: {
                hop: {folder: [] for folder in self.folders}
                for hop in range(1, self.N_HOPS + 1)
            }
            for split in self.splits
        }

        self.answers = {
            split: {
                hop: {folder: [] for folder in self.folders}
                for hop in range(1, self.N_HOPS + 1)
            }
            for split in self.splits
        }

    def get_graph(self):
        return self.graph

    def set_graph(self, graph):
        self.graph = graph

    def get_questions(self):
        return self.questions

    def set_questions(self, questions):
        self.questions = questions

    def get_answers(self):
        return self.answers

    def set_answers(self, answers):
        self.answers = answers

    def get_question_types(self):
        return self.qs_types

    def get_relations(self):
        return self.relations

    def get_data(self, data_type):
        # Retrieve the specified data type (training, validation, or test) from
        # the preprocessed data.
        questions = []
        qs_types = []
        answers = []

        for hop in range(1, self.N_HOPS + 1):
            hop_types = self.qs_types[data_type][hop]

            for folder in self.folders:
                hop_questions = self.questions[data_type][hop][folder]
                hop_answers = self.answers[data_type][hop][folder]

                questions.extend(hop_questions)
                answers.extend(hop_answers)

                if not hop_questions:
                    continue

                if len(hop_questions) != len(hop_types):
                    raise ValueError(
                        f"Question type mismatch: {data_type}, "
                        f"{hop}-hop, {folder}: "
                        f"{len(hop_questions)} questions vs "
                        f"{len(hop_types)} types"
                    )

                qs_types.extend(hop_types)

        return questions,  qs_types, answers

    def preprocess(self):
        # Preprocess the knowledge base data and add it to the graph.
        self.add_entities()
        self.add_relations()
        self.add_question_types()
        self.add_questions()

    def add_entities(self):
        # Add the knowledge base data into the graph as entities.
        with open('MetaQA/entity/kb_entity_dict.txt', 'r') as f:
            for line in f:
                u = re.sub(r'^[0-9 ]+', '', line.strip())
                self.graph.add_entity(u)

    def add_relations(self):
        # Add the knowledge base data into the graph as entities and relations.
        with open('MetaQA/kb.txt', 'r') as f:
            for line in f:
                u, relation, v = line.strip().split('|')
                self.graph.add_relation(u, v, relation)
                self.relations.add(relation)

    def add_question_types(self):
        # Store the question types from each hop's question_types.txt file into
        # a list.
        for split in self.splits:
            for i in range(1, self.N_HOPS + 1):
                filename = f"MetaQA/{i}-hop/qa_{split}_qtype.txt"
                with open(filename, "r") as f:
                    self.qs_types[split][i] = [line.strip() for line in f]

    def add_questions(self):
        # Load the question-answer pairs from each hop's text file and store
        # them in the questions and answers dictionaries.
        for data_type in self.splits:
            for i in range(1, self.N_HOPS + 1):
                for folder in self.folders:
                    qs, ans = self.add_qs_to_hop(i, folder, data_type)
                    self.questions[data_type][i][folder] = qs
                    self.answers[data_type][i][folder] = ans

    def add_qs_to_hop(self, hop, folder, split):
        # Load the question-answer pairs from the specified hop's text file.
        questions = []
        answers = []
        filename = f'MetaQA/{hop}-hop/{folder}/qa_{split}.txt'
        with open(filename, 'r') as f:
            rows = [line.strip() for line in f if line.strip()]
            for line in rows:
                question, answer = line.split('\t')
                answer_list = answer.split('|')
                questions.append(question)
                answers.append(answer_list)

        return questions, answers