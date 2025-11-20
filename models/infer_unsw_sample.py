
import pandas as pd
import joblib
import sys, os

# Setting the working directory in scripts so that the project's package works properly wherever it is copied
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("###### ", os.getcwd())

# Use CSV path from command line argument, fallback to default sample
csv_file = sys.argv[1] if len(sys.argv) > 1 else "sample_intrusion.csv"

# Load CSV
df = pd.read_csv(csv_file)

# Load model pipeline + metadata
saved = joblib.load("../models/network_model_unsw.joblib")
pipe = saved["pipeline"]
label_encoder = saved["label_encoder"]
cat_cols = saved["categorical_cols"]
num_cols = saved["numeric_cols"]
target = saved["target"]

# Ensure dtypes match training (optional, can leave as-is)
for c in cat_cols:
    df[c] = df[c].astype(str)
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Select columns in correct order
X = df[cat_cols + num_cols]

# Predict
preds_enc = pipe.predict(X)
preds = label_encoder.inverse_transform(preds_enc)

# Add predictions to DataFrame
df["predicted_" + target] = preds

# Print predictions to stdout for Streamlit to capture
print(df)
