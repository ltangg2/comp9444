import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class Question_Classifer:
    def __init__(self):
        self.model = LogisticRegression(
            solver="saga",
            random_state=42,
            max_iter=300
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
            min_df=5,
            max_features=40000
        )

    def train(self, questions, question_types):
        # Fit the question classifier model using the provided questions and
        # their corresponding types.
        normalised_questions = self.normalise_questions(questions)
        X_train = self.vectorizer.fit_transform(normalised_questions)
        self.model.fit(X_train, question_types)

    def normalise_questions(self, questions):
        # Replaces the entities in every question with ENTITY.
        results = []
        for question in questions:
            results.append(self.normalise_question(question))

        return results

    def normalise_question(self, question):
        # Replace the specific entity in the question with ENTITY, such that
        # during training, validation or testing, the classifier can focus on
        # the words that indicate the relation path, rather than the
        # associations with particular entity names.
        return re.sub(
            r"\[[^\]]+\]",
            "ENTITY",
            question
        )

    def predict(self, question):
        # Predict the type of a given question using the trained model.
        normalised_question = self.normalise_question(question)
        features = self.vectorizer.transform([normalised_question])
        return self.model.predict(features)[0]

    def evaluate(self, questions, labels):
        # Evaluate the model on a test dataset and print the classification
        # report and accuracy.
        normalised_questions = self.normalise_questions(questions)
        features = self.vectorizer.transform(normalised_questions)
        predictions = self.model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, zero_division=0)

        return accuracy, report, predictions