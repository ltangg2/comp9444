import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class Question_Classifer:
    def __init__(self):
        self.model = LogisticRegression(solver="saga", random_state=42,
                                        max_iter=300)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                        stop_words="english",
                                        min_df=5,
                                        max_features=40000
                                    )

    def train(self, questions, question_types):
        # Fit the question classifier model using the provided questions and
        # their corresponding types.
        print("Start question classifier training...")
        X_train_tfidf = self.vectorizer.fit_transform(questions)
        self.model.fit(X_train_tfidf, question_types)
        print("Question classifier training completed.")

    def predict(self, question):
        # Predict the type of a given question using the trained model.
        features = self.vectorizer.transform([question])
        prediction = self.model.predict(features)[0]

        return prediction

    def validate(self, validate_questions, validate_labels):
        # Validate the model on a validation dataset and print the classification
        # report and accuracy.
        print("Start question classifier validation...")
        X_validate_tfidf = self.vectorizer.transform(validate_questions)
        y_pred = self.model.predict(X_validate_tfidf)
        acc = accuracy_score(validate_labels, y_pred)

        print(f"Overall Accuracy: {acc:.2%}")
        print("Detailed Classification Report:")
        print(classification_report(validate_labels, y_pred))
        print("End of question classifier validation.")

    def test(self, test_questions, test_labels):
        # Evaluate the model on a test dataset and print the classification
        # report and accuracy.
        print("Start question classifier testing...")
        X_test_tfidf = self.vectorizer.transform(test_questions)
        y_pred = self.model.predict(X_test_tfidf)
        acc = accuracy_score(test_labels, y_pred)

        print(f"Overall Accuracy: {acc:.2%}")
        print("Detailed Classification Report:")
        print(classification_report(test_labels, y_pred))
        print("End of question classifier testing.")