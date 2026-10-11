import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model on a test dataset.

    Returns Accuracy, Macro-F1, MCC, and a classification report.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )
    mcc = matthews_corrcoef(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Macro-F1": macro_f1,
        "MCC": mcc,
        "Classification Report": report
    }


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train a model, measure training time, and evaluate its performance.
    """
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    metrics = evaluate_model(model, X_test, y_test)
    metrics["Training Time (sec)"] = training_time

    return metrics


if __name__ == "__main__":
    print("Evaluation functions loaded successfully.")