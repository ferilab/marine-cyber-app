
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, sys

# Setting the working directory in scripts so that the project's package works properly wherever it is copied
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("********** ", os.getcwd())

st.set_page_config(page_title="Marine Cybersecurity Demo", layout="wide")

st.title("🚢 Marine Cybersecurity Demo")
st.write("Network intrusion detection and phishing email classifier")

# ----------------------------------------------------------
# Model locations
# ----------------------------------------------------------
# UNSW_MODEL_PATH = "../models/network_model_unsw_rf.joblib"
UNSW_MODEL_PATH = "../models/network_model_unsw_xgb.joblib"
PHISH_MODEL_PATH = "../models/phishing_model.joblib"

# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------
col1, col2 = st.columns(2)

# ==========================================================
# NETWORK INTRUSION DETECTOR (UNSW-NB15)
# ==========================================================
with col1:
    st.header("🔐 Network Intrusion Detection")

    st.write("Upload a CSV containing flow records (in sample-like format). The app will predict if it is a likely intrusion.")

    uploaded_file = st.file_uploader("Upload your data", type=["csv"], key="unsw")

    if st.button("Run intrusion detector"):
        if uploaded_file is None:
            st.error("Please upload a CSV first.")
        else:
            import tempfile
            import subprocess

            # Save uploaded file to a temporary CSV on disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Path to your working inference script
            infer_script = os.path.join("../models", "infer_unsw_sample.py")

            if not os.path.exists(infer_script):
                st.error(f"Inference script not found: {infer_script}")
            else:
                try:
                    # Call the script via subprocess
                    result = subprocess.run(
                        ["python", infer_script, tmp_path],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    # Display the stdout of the script
                    st.text("Predictions output (0: Normal, 1: Attack):\n" + result.stdout)

                except subprocess.CalledProcessError as e:
                    st.error(f"Error running inference script:\n{e.stderr}")


# ==========================================================
# PHISHING EMAIL CLASSIFIER
# ==========================================================
with col2:
    st.header("✉️ Phishing Email Detector")

    st.write("Paste email content. The classifier will detect phishing vs legit.")

    email_text = st.text_area("Enter email text here", height=200)

    if st.button("Check email"):
        if not email_text:
            st.error("Paste some text first.")
        else:
            import tempfile, subprocess, os, shlex

            # write text to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
                tmp.write(email_text)
                tmp_path = tmp.name

            infer_script = os.path.join("../models", "infer_phishing.py")
            if not os.path.exists(infer_script):
                st.error(f"Inference script not found: {infer_script}")
            else:
                try:
                    # call the script; pass the temp file path as single arg
                    res = subprocess.run(
                        ["python", infer_script, tmp_path],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    # show stdout from script
                    st.text("Result:\n" + res.stdout.strip())

                except subprocess.CalledProcessError as e:
                    # show stderr if script failed
                    st.error("Inference script failed:\n" + (e.stderr or e.stdout))

                finally:
                    # cleanup temp file
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass


st.markdown("---")
st.write("⚙️ **Technical notes:** The Network Intrusion and Fishing models use real-world verified datasets for traning and testing.")
# st.write("⚙️ **Technical notes:** The UNSW model uses a full scikit-learn pipeline (OneHotEncoder + StandardScaler + RandomForest) and auto-detects features. `attack_cat` is intentionally excluded to avoid leakage when predicting `label`.")
