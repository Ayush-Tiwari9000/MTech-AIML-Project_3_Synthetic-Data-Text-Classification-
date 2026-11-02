import time
from typing import Any, Dict
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)


def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate a trained classification model on a test dataset.

    Computes:
        - Accuracy
        - Macro-F1
        - Matthews Correlation Coefficient (MCC)
        - Per-class metrics (HAM & SPAM Precision, Recall, F1)
        - Confusion Matrix
        - Text classification report
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, predictions)

    # Detailed per-class metrics
    ham_f1 = f1_score(y_test, predictions, pos_label="ham", zero_division=0)
    spam_f1 = f1_score(y_test, predictions, pos_label="spam", zero_division=0)
    ham_recall = recall_score(y_test, predictions, pos_label="ham", zero_division=0)
    spam_recall = recall_score(y_test, predictions, pos_label="spam", zero_division=0)
    ham_prec = precision_score(y_test, predictions, pos_label="ham", zero_division=0)
    spam_prec = precision_score(y_test, predictions, pos_label="spam", zero_division=0)

    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    report = classification_report(y_test, predictions, zero_division=0)

    return {
        "Accuracy": float(accuracy),
        "Macro-F1": float(macro_f1),
        "MCC": float(mcc),
        "HAM Precision": float(ham_prec),
        "HAM Recall": float(ham_recall),
        "HAM F1": float(ham_f1),
        "SPAM Precision": float(spam_prec),
        "SPAM Recall": float(spam_recall),
        "SPAM F1": float(spam_f1),
        "Confusion Matrix": cm,
        "Classification Report": report
    }


def train_and_evaluate(model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Train a model pipeline, measure precise training time via time.perf_counter(),
    and evaluate its performance on the test set.
    """
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    metrics = evaluate_model(model, X_test, y_test)
    metrics["Training Time (sec)"] = float(training_time)

    return metrics


if __name__ == "__main__":
    print("Evaluation functions loaded successfully.")
