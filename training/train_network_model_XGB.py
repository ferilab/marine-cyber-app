
# training/train_network_model.py
"""
Training script specialized for UNSW-NB15.
XGBoost version (no hyperparameter tuning).
"""

import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from xgboost import XGBClassifier

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
print("********** ", os.getcwd())

TRAIN_PATH = "data/unsw-nb15/UNSW_NB15_training-set.csv"
TEST_PATH  = "data/unsw-nb15/UNSW_NB15_testing-set.csv"

OUT_MODEL = "models/network_model_unsw_xgb.joblib"

TARGET = "label"
# TARGET = "attack_cat"

DROP_COLS = ["id", "attack_cat"]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_unsw():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    for df in (train, test):
        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    if TARGET not in train.columns:
        raise ValueError(f"Target '{TARGET}' not found in dataset")

    return train, test


# ---------------------------------------------------------
# PREPARE FEATURES
# ---------------------------------------------------------

def split_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)
    return X, y


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train():
    print("Loading UNSW-NB15 datasets...")
    train_df, test_df = load_unsw()

    X_train, y_train = split_xy(train_df)
    X_test,  y_test  = split_xy(test_df)

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols     = X_train.select_dtypes(include=[np.number]).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc  = label_encoder.transform(y_test)

    # -------------------------------
    # Preprocessing + XGBoost
    # -------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",     # Fastest on CPU
        random_state=42
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", xgb_model)
    ])

    print("Training XGBoost model...")
    pipe.fit(X_train, y_train_enc)

    # -------------------------------
    # Evaluation
    # -------------------------------
    preds = pipe.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test_enc, preds, target_names=label_encoder.classes_))

    print("Confusion matrix:")
    print(confusion_matrix(y_test_enc, preds))

    # -------------------------------
    # Save model
    # -------------------------------
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "label_encoder": label_encoder,
        "target": TARGET,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }, OUT_MODEL)

    print(f"\nModel saved to: {OUT_MODEL}")


# ---------------------------------------------------------
if __name__ == "__main__":
    train()
