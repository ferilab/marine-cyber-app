
# training/train_phishing_model.py
"""
Train a lightweight phishing email detector (text-based).
Assumes CSV with a text column (text_combined, email_text, body, etc.)
and a 'label' column ('phish' or 'legit').

Saves a fitted sklearn Pipeline (TF-IDF + LogisticRegression) to:
    models/phishing_model.joblib
"""

import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Ensure correct import paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

DATA_PATH = "data/phishing-email-dataset/phishing_emails.csv"
OUT_MODEL = "models/phishing_model.joblib"

def train():
    df = pd.read_csv(DATA_PATH)

    # -----------------------------
    # 1. Find the text column
    # -----------------------------
    text_col = None
    for c in df.columns:
        if c.lower() in ["text_combined", "text", "email_text", "email", "body", "message"]:
            text_col = c
            break

    if text_col is None:
        raise ValueError("Could not find a text column. Expected one of: text_combined, text, email_text, body")

    df = df.rename(columns={text_col: "text_combined"})

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")

    # Drop missing rows
    df = df.dropna(subset=["text_combined", "label"])

    X = df["text_combined"].astype(str)
    y = df["label"].astype(str)

    # -----------------------------
    # 2. Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------
    # 3. Pipeline: TF-IDF + LR
    # -----------------------------
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    tfidf = pipeline.named_steps["tfidf"]
    print("Has idf_?:", hasattr(tfidf, "idf_"))
    print("Vocabulary size:", len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else 0)

    # -----------------------------
    # 4. Evaluation
    # -----------------------------
    preds = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # -----------------------------
    # 5. Save the fitted pipeline
    # -----------------------------
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(pipeline, OUT_MODEL)
    print("\nSaved phishing model to:", OUT_MODEL)

if __name__ == "__main__":
    train()
