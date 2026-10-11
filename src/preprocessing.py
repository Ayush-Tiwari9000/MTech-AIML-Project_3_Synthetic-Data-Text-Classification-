from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sms_spam.csv"


def load_dataset(file_path=DEFAULT_DATA_PATH):
    """
    Load and validate the SMS Spam Collection dataset.

    Expected columns:
        text  - SMS message
        label - ham or spam
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Standardize column names.
    df.columns = df.columns.astype(str).str.strip().str.lower()

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[["text", "label"]].copy()

    # Remove incomplete rows and normalize labels.
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]

    if not df["label"].isin(["ham", "spam"]).all():
        raise ValueError(
            "Labels must contain only 'ham' and 'spam'."
        )

    return df.reset_index(drop=True)


def prepare_data(
    df,
    test_size=0.20,
    real_fraction=0.20,
    random_state=42
):
    """
    Split the dataset and retain only a restricted portion
    of the training data as real training data.

    Returns:
        X_real, X_test, y_real, y_test
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < real_fraction < 1:
        raise ValueError("real_fraction must be between 0 and 1.")

    X = df["text"]
    y = df["label"]

    # Reserve the real test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Restrict the available training portion.
    X_real, _, y_real, _ = train_test_split(
        X_train,
        y_train,
        train_size=real_fraction,
        stratify=y_train,
        random_state=random_state
    )

    return (
        X_real.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_real.reset_index(drop=True),
        y_test.reset_index(drop=True)
    )


if __name__ == "__main__":
    dataset = load_dataset()

    X_real, X_test, y_real, y_test = prepare_data(dataset)

    print("Dataset loaded successfully.")
    print("Total messages:", len(dataset))
    print("\nOriginal class distribution:")
    print(dataset["label"].value_counts())

    print("\nLimited real training samples:", len(X_real))
    print(y_real.value_counts())

    print("\nTest samples:", len(X_test))
    print(y_test.value_counts())