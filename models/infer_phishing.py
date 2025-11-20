
# models/infer_phishing.py
import sys
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "phishing_model.joblib")

def read_input_text(arg_path):
    # If arg_path == "-" read from stdin, else read file
    if arg_path == "-":
        return sys.stdin.read()
    with open(arg_path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    # usage: python infer_phishing.py /path/to/textfile.txt
    arg = sys.argv[1] if len(sys.argv) > 1 else "-"
    text = read_input_text(arg)
    text = text.strip()
    if text == "":
        print("ERROR: empty input", file=sys.stderr)
        sys.exit(2)

    # Load pipeline (should be a fitted sklearn Pipeline)
    model = joblib.load(MODEL_PATH)

    # If someone saved a dict accidentally, try to extract pipeline
    if isinstance(model, dict):
        # common keys: 'pipeline', 'model', 'vectorizer'
        if "pipeline" in model:
            model = model["pipeline"]
        elif "model" in model:
            model = model["model"]
        else:
            print("ERROR: loaded object is a dict but no pipeline found.", file=sys.stderr)
            sys.exit(3)

    # Predict
    try:
        pred = model.predict([text])[0]
    except Exception as e:
        print(f"ERROR: prediction failed: {e}", file=sys.stderr)
        sys.exit(4)

    # Try probabilities if available
    probs_str = ""
    try:
        probs = model.predict_proba([text])[0]
        probs_str = " | probs (legit, fishing): " + ", ".join([f"{p:.4f}" for p in probs])
    except Exception:
        probs_str = ""

    # Print simple, parseable single-line result
    print("prediction:", "Fishing" if int(pred) == 1 else "Legit", probs_str)

if __name__ == "__main__":
    main()
