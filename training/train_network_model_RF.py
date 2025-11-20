
# training/train_network_model.py
"""
Training script specialized for UNSW-NB15.
Uses the original training/testing CSVs:
    data/UNSW-NB15/UNSW_NB15_training-set.csv
    data/UNSW-NB15/UNSW_NB15_testing-set.csv

You can choose which target to model:
    TARGET = "label"        # binary (0 = normal, 1 = attack)
    TARGET = "attack_cat"   # 10-category multiclass
"""

import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

# Setting the working directory in scripts so that the project's package works properly wherever it is copied
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
print("********** ", os.getcwd())

# The testing-set is much bigger than the training-set (175341 vs 82332)!
TRAIN_PATH = "data/unsw-nb15/UNSW_NB15_training-set.csv"
TEST_PATH  = "data/unsw-nb15/UNSW_NB15_testing-set.csv"

OUT_MODEL = "models/network_model_unsw_rf.joblib"

# Choose one: "label" (binary) or "attack_cat" (multiclass)
TARGET = "label"
# TARGET = "attack_cat"

# Columns we will drop because they are identifiers or redundant like id and a source of leakage like attack_cat
DROP_COLS = ["id", "attack_cat"]   # if we choose attack_cat as the target then label should be droped instead

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_unsw():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    # Drop non-predictive columns if present
    for df in (train, test):
        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Ensure target exists
    if TARGET not in train.columns:
        raise ValueError(f"Target '{TARGET}' not found in dataset")

    return train, test


# ---------------------------------------------------------
# PREPARE FEATURES
# ---------------------------------------------------------

def split_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Convert y to string labels for consistent encoding later
    y = y.astype(str)

    return X, y


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train():
    print("Loading UNSW-NB15 datasets...")
    train_df, test_df = load_unsw()

    X_train, y_train = split_xy(train_df)
    X_test,  y_test  = split_xy(test_df)

    # Detect categorical vs numeric columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols     = X_train.select_dtypes(include=[np.number]).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # Label-encode the target
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc  = label_encoder.transform(y_test)

    # -------------------------------
    # Preprocessing + Base Model
    # -------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    base_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", base_rf)
    ])

    best_model = pipe.fit(X_train, y_train_enc)

    # # -------------------------------
    # # Hyperparameter Tuning
    # # -------------------------------
    # from sklearn.model_selection import RandomizedSearchCV

    # param_grid = {
    #     "clf__n_estimators": [200, 400],
    #     "clf__max_depth": [10, 20],
    #     "clf__min_samples_split": [5, 10],
    #     "clf__min_samples_leaf": [1, 2],
    #     "clf__max_features": ["sqrt", "log2"],
    #     "clf__bootstrap": [True]
    # }

    # print("\n🔍 Running RandomizedSearchCV for RandomForest (30 iterations)...")

    # tuner = RandomizedSearchCV(
    #     estimator=pipe,
    #     param_distributions=param_grid,
    #     n_iter=30,                # you can increase to 50–100 for better results
    #     scoring="f1_macro",       # works well for imbalance + multiclass
    #     refit=True,
    #     cv=3,
    #     verbose=2,
    #     random_state=42,
    #     n_jobs=-1
    # )

    # tuner.fit(X_train, y_train_enc)

    # print("\nBest parameters found:")
    # print(tuner.best_params_)

    # best_model = tuner.best_estimator_

    # -------------------------------
    # Evaluation
    # -------------------------------
    preds = best_model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test_enc, preds, target_names=label_encoder.classes_))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test_enc, preds))

    # -------------------------------
    # Save everything
    # -------------------------------
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({
        "pipeline": best_model,
        "label_encoder": label_encoder,
        "target": TARGET,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }, OUT_MODEL)

    print(f"\n💾 Tuned model saved to: {OUT_MODEL}")


# ---------------------------------------------------------
if __name__ == "__main__":
    train()
