import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

DATA_PATHS = ["data/train.csv", "train.csv"]
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "preference_model.joblib")
TEST_SIZE = 0.2
RANDOM_STATE = 42

def find_data():
    for path in DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find train.csv. Put the Kaggle file at data/train.csv "
        "or in the same folder as train.py."
    )

def to_text(value):
    if pd.isna(value):
        return ""
    return str(value)

def main():
    data_path = find_data()
    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    required = {"prompt", "response_a", "response_b",
                "winner_model_a", "winner_model_b", "winner_tie"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}\n"
            f"Columns found: {list(df.columns)}\n"
            "Check that you downloaded the competition's train.csv file."
        )

    for col in ["prompt", "response_a", "response_b"]:
        df[col] = df[col].apply(to_text)

    # Convert the competition's three one-hot winner columns to one target label.
    winner_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    for col in winner_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df[winner_cols].sum(axis=1) > 0].copy()
    df["label"] = df[winner_cols].idxmax(axis=1).map({
        "winner_model_a": "Response A preferred",
        "winner_model_b": "Response B preferred",
        "winner_tie": "Tie"
    })

    # Combine all three text fields. Clear separators help distinguish each part.
    df["text"] = (
        "PROMPT: " + df["prompt"] +
        " RESPONSE A: " + df["response_a"] +
        " RESPONSE B: " + df["response_b"]
    )
    df = df[df["text"].str.strip().ne("")].copy()

    # Optional safety cap for laptops. Set MAX_ROWS=0 to use the full dataset.
    max_rows = int(os.environ.get("MAX_ROWS", "0"))
    if max_rows > 0 and len(df) > max_rows:
        df = resample(df, n_samples=max_rows, random_state=RANDOM_STATE,
                      stratify=df["label"])

    print(f"Rows used: {len(df):,}")
    print("Label counts:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    print("Next run: python evaluate.py")

if __name__ == "__main__":
    main()
