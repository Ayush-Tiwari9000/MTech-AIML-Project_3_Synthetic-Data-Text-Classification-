from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def create_model():
    """
    Create the SMS spam classification pipeline.

    TF-IDF converts SMS text into numerical features.
    Logistic Regression predicts whether a message is ham or spam.
    """
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    return model


if __name__ == "__main__":
    model = create_model()
    print("Model pipeline created successfully.")
    print(model)