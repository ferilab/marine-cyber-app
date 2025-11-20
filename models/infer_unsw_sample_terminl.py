
import pandas as pd
import joblib
import os, sys

# Setting the working directory in scripts so that the project's package works properly wherever it is copied
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
print("\n", "********** ", os.getcwd(), "\n")

# Paths
MODEL_PATH = "network_model_unsw.joblib"
CSV_PATH = "data/sample_intrusion.csv"  # your downloaded CSV

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

# Load model pipeline + label encoder
saved = joblib.load(MODEL_PATH)
pipe = saved["pipeline"]
label_encoder = saved["label_encoder"]
cat_cols = saved["categorical_cols"]
num_cols = saved["numeric_cols"]
target = saved["target"]

# Read sample CSV
df = pd.read_csv(CSV_PATH)

# Make sure we pass exactly the columns used during training
X = df[cat_cols + num_cols]

# Predict
preds_enc = pipe.predict(X)
preds = label_encoder.inverse_transform(preds_enc)

# Output
df["predicted_" + target] = preds
print("Predictions:")
print(df)
