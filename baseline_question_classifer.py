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
            stop_words="english",
            min_df=5,
            max_features=40000
        )

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

    def train(self, questions, question_types):
        # Fit the question classifier model using the provided questions and
        # their corresponding types.
        print("Start question classifier training...")

        normalised_questions = [
            self.normalise_question(q)
            for q in questions
        ]

        X_train = self.vectorizer.fit_transform(normalised_questions)
        self.model.fit(X_train, question_types)
        print("Question classifier training completed.")

    def predict(self, question):
        # Predict the type of a given question using the trained model.
        normalised_question = self.normalize_question(question)
        features = self.vectorizer.transform([normalised_question])
        return self.model.predict(features)[0]

    def evaluate(self, questions, labels):
        # Evaluate the model on a test dataset and print the classification
        # report and accuracy.
        normalised_questions = [
            self.normalise_question(q)
            for q in questions
        ]

        features = self.vectorizer.transform(normalised_questions)
        predictions = self.model.predict(features)
        accuracy = accuracy_score(labels, predictions)

        print(f"Classification accuracy: {accuracy:.2%}")
        print(classification_report(labels, predictions, zero_division=0))
        return accuracy