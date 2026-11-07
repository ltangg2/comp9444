from baseline.graph import Graph
from baseline.baseline_preprocess import Preprocess
from baseline.baseline_question_classifer import Question_Classifer
from baseline.baseline import Baseline

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

QS = 0
TYPES = 1
ANS = 2

def plot_training_validation_curve(
    graph,
    train_qs,
    train_types,
    train_ans,
    dev_qs,
    dev_ans
):
    # Plots training and validation exact-match accuracy
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_accs = []
    val_accs = []

    print("\n=== Generating Training and Validation Curves ===")
    for frac in fractions:
        subset_size = int(len(train_qs) * frac)
        sub_qs = train_qs[:subset_size]
        sub_types = train_types[:subset_size]
        sub_ans = train_ans[:subset_size]

        qs_class = Question_Classifer()
        qs_class.train(sub_qs, sub_types)

        baseline = Baseline(graph, qs_class)

        train_acc = baseline.evaluate(sub_qs, sub_ans)
        val_acc = baseline.evaluate(dev_qs, dev_ans)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

    plt.figure(figsize=(8, 5))

    plt.plot(
        fractions,
        train_accs,
        marker='o',
        label='Training Accuracy',
        color='blue',
        linewidth=2
    )

    plt.plot(
        fractions,
        val_accs,
        marker='s',
        label='Validation Accuracy',
        color='orange',
        linewidth=2
    )

    plt.xlabel("Fraction of Training Data", fontsize=11)
    plt.ylabel("Exact-Match Accuracy", fontsize=11)
    plt.title("Baseline Model Training and Validation Curve", fontsize=13)
    plt.xticks(fractions, [f"{int(f*100)}%" for f in fractions])
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(labels, predicted):
    # Creating a confusion matrix to evaluate the performance of the
    # question classifier.
    classes = sorted(set(labels) | set(predicted))
    matrix = confusion_matrix(labels, predicted, labels=classes)
    num_classes = len(classes)
    fig_size = max(12, num_classes * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    matrix_normalized = matrix.astype('float') / matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.nan_to_num(matrix_normalized)

    sns.heatmap(
        matrix_normalized,
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=classes,
        yticklabels=classes,
        annot=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
    )

    ax.set_xlabel("Predicted Question Type")
    ax.set_ylabel("True Question Type")
    ax.set_title(f"Confusion Matrix ({num_classes} Question Types)")

    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n=== Preprocessing ===")
    preprocess = Preprocess()
    print("Start preprocessing...")
    preprocess.preprocess()
    print("Preprocessing completed.")

    graph = preprocess.get_graph()
    data = {
        "train": preprocess.get_data("train"),
        "dev": preprocess.get_data("dev"),
        "test": preprocess.get_data("test")
    }

    print("\n=== Question Classifier ===")
    question_classifier = Question_Classifer()
    print("Start question classifier training...")
    question_classifier.train(data["train"][QS], data["train"][TYPES])
    print("Question classifier training completed.")

    predictions = []
    for split in ["dev", "test"]:
        accuracy, report, pred = question_classifier.evaluate(
            data[split][QS],
            data[split][TYPES]
        )

        if split == "dev":
            predictions = pred

        print(f"Classification accuracy for {split} data: {accuracy:.2%}")
        print(report)

    print("\n=== Confusion Matrix ===")
    plot_confusion_matrix(data["dev"][TYPES], predictions)

    print("\n=== Baseline ===")
    baseline = Baseline(graph, question_classifier)

    print("\n### Training and validation curve ###")
    plot_training_validation_curve(
        graph,
        data["train"][QS],
        data["train"][TYPES],
        data["train"][ANS],
        data["dev"][QS],
        data["dev"][ANS]
    )

    print("\n### Final Test Evaluation ###")
    accuracy = baseline.evaluate(data["test"][QS], data["test"][ANS])
    print(f"Test accuracy: {accuracy:.2%}")