from typing import Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def create_model(
    ngram_range: Tuple[int, int] = (1, 2),
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: Optional[Union[str, dict]] = None,
    sublinear_tf: bool = False,
    random_state: int = 42
) -> Pipeline:
    """
    Create the SMS spam classification pipeline.

    Architecture:
        1. TF-IDF Vectorizer: converts raw SMS text into numerical n-gram features
           (unigrams and bigrams by default, English stop-words removed).
        2. Logistic Regression: linear classifier modeling the probability
           of a message being spam vs. ham.

    Parameters:
        ngram_range: Lower and upper boundary of range of n-values for n-grams.
        max_iter: Maximum number of iterations for solver convergence.
        C: Inverse regularization strength (smaller values specify stronger regularization).
        class_weight: Weights associated with classes ('balanced' or None).
        sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf)).
        random_state: Seed for random number generator reproducibility.

    Returns:
        Pipeline: Scikit-learn Pipeline instance ready for training.
    """
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=ngram_range,
                sublinear_tf=sublinear_tf
            )
        ),
        (
            "classifier",
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state
            )
        )
    ])

    return model


if __name__ == "__main__":
    default_model = create_model()
    balanced_model = create_model(class_weight="balanced")

    print("Model pipeline created successfully.")
    print("\nDefault pipeline:")
    print(default_model)
    print("\nBalanced pipeline:")
    print(balanced_model)
