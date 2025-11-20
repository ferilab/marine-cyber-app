# marine-cyber-app
A maritime cyber application to protect the marine facilities and ships against ransomware, phishing, fraudulent communications and intrusion.

INTRODUCTION:

🔐 Marine Cybersecurity Threat Detection Platform

This project provides an integrated machine-learning platform for detecting cyber threats in maritime digital systems. It includes two complementary components:

Network Intrusion Detection System (NIDS) based on the UNSW-NB15 dataset

Phishing Email Classification System using TF-IDF and logistic regression

The platform is delivered through a unified Streamlit web interface, supporting:

CSV-based batch inference for network traffic

Real-time email text analysis

Model training scripts

Reproducible inference modules

Clear documentation of all required features

The goal is to serve as a lightweight, fully transparent, and easily deployable reference system for research, education, and prototyping in marine cyber defense scenarios. All features used by each model, file structures, expected inputs, and reproducible workflows are documented below.



# A. Intrusion Detection (using UNSW-NB15 data)

About this Dataset (https://www.kaggle.com/code/joshuaaideloje/intrusion-detection-system)

The raw network packets of the UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) for generating a hybrid of real modern normal activities and synthetic contemporary attack behaviours.

Tcpdump tool is utilised to capture 100 GB of the raw traffic (e.g., Pcap files). This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. The Argus, Bro-IDS tools are used and twelve algorithms are developed to generate totally 49 features with the class label.

These features are described in UNSW-NB15_features.csv file.

The total number of records is two million and 540,044 which are stored in the four CSV files, namely, UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv and UNSW-NB15_4.csv.

The ground truth table is named UNSW-NB15_GT.csv and the list of event file is called UNSW-NB15_LIST_EVENTS.csv.

A partition from this dataset is configured as a training set and testing set, namely, UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv respectively.

The number of records in the training set is 175,341 records and the testing set is 82,332 records from the different types, attack and normal.Figure 1 and 2 show the testbed configuration dataset and the method of the feature creation of the UNSW-NB15, respectively.


🛡️ 1. Intrusion Detection Model – UNSW-NB15
✔️ Input Format

The model expects a CSV file with 42 numerical features, matching the original UNSW-NB15 numeric-only feature subset.
The target columns removed before training were:

id

attack_cat

label

✔️ Expected Columns (42 Features)

Below is the full list of variables used by the model.
These match the official UNSW-NB15 feature definitions.

Feature	Description
dur	Duration of the flow (seconds)
spkts	Source-to-destination packet count
dpkts	Destination-to-source packet count
sbytes	Bytes sent from source to destination
dbytes	Bytes sent from destination to source
rate	Packet rate (pkts/sec)
sttl	Source-to-destination time-to-live
dttl	Destination-to-source time-to-live
sload	Source bits per second
dload	Destination bits per second
sloss	Source packet loss
dloss	Destination packet loss
sinpkt	Source inter-packet arrival time
dinpkt	Destination inter-packet arrival time
sjit	Source jitter
djit	Destination jitter
swin	Source TCP window size
dwin	Destination TCP window size
stcpb	Source TCP base sequence number
dtcpb	Destination TCP base sequence number
tcprtt	TCP round-trip time
synack	SYN–ACK time
ackdat	ACK–DATA time
smean	Mean packet size from source
dmean	Mean packet size from destination
trans_depth	Pipelined requests
response_body_len	Response body length
ct_srv_src	Connections to same service from same source IP
ct_state_ttl	Count of specific state and TTL combination
ct_dst_ltm	Connections from same destination IP in last X time
ct_src_dport_ltm	Connections from same src IP to same dst port (recent)
ct_dst_sport_ltm	Connections to same dst IP from same src port (recent)
ct_dst_src_ltm	Connections between same src–dst pair (recent)
is_ftp_login	FTP login success indicator
ct_ftp_cmd	Number of FTP commands
ct_flw_http_mthd	Count of HTTP methods
ct_src_ltm	Connections from same source IP (recent)
ct_srv_dst	Connections to same service from same destination IP
is_sm_ips_ports	Indicator of (src IP == dst IP & src port == dst port)


✔️ Model Output

predicted_label

0 = "normal"

1 = "attack"

There are two models avaialble for training:

1. RandomForest (use train_network_model_rf.py and it will save the trained model as network_model_unsw_rf.joblib in models folder)

The hyperparameter tuning block is commented out after finding the best paramaters as follows

The Tuned Randomforest:
Best parameters found:
{'clf__n_estimators': 200, 'clf__min_samples_split': 5, 'clf__min_samples_leaf': 2, 'clf__max_features': 'sqrt', 'clf__max_depth': 20, 'clf__bootstrap': True}
   
2. XGBoost (use train_network_model_xgb.py and it will save the trained model as network_model_unsw_xgb.joblib in models folder)



# ✉️ B. Phishing Email Classifier (Text-Based)

About Dataset 

PHISHING EMAIL DATASET  (https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=phishing_email.csv)

This dataset was compiled by researchers to study phishing email tactics. It combines emails from a variety of sources to create a comprehensive resource for analysis.

Initial Datasets:
Enron and Ling Datasets: These datasets focus on the core content of phishing emails, containing subject lines, email body text, and labels indicating whether the email is spam (phishing) or legitimate.

CEAS, Nazario, Nigerian Fraud, and SpamAssassin Datasets: These datasets provide broader context for the emails, including sender information, recipient information, date, and labels for spam/legitimate classification.

Final Dataset:
The final dataset combines the information from the initial datasets into a single resource for analysis. This dataset contains:

Approximately 82,500 emails
42,891 spam emails
39,595 legitimate emails
This dataset allows researchers to study the content of phishing emails and the context in which they are sent to improve detection methods.



✔️ Input Format

A single email body as free-text input (string).
The model processes:

entire email body

header + body if combined in dataset

The training script uses:

df['text_combined']

✔️ Text Vectorization

The model uses:

TfidfVectorizer

min_df = 2

max_df = 0.9

ngram_range = (1, 2) → unigram + bigram features

This produces a sparse feature vector of ~1.2M features, depending on the dataset.

✔️ Model Output

1 = phish → detected phishing email

0 = legit → normal/non-phishing email

If enabled, the model returns:

P(phish)
P(legit)


# UI

Streamlit Frontend

The web interface provides:

1. File upload for network traffic (.csv)

2. Email text analysis panel

3. Real-time prediction and probability outputs



# 📊 File Locations

marine-cyber-app
    - app
      - streamlit_app.py
    - data
      - phishing-email-dataset
        - phishing_emails.csv
      - sample_intrusion_20.csv
      - sample_intrusion.csv
    - unsw-nb15
      - UNSW_NB15_testing-set.csv
      - UNSW_NB15_training-set.csv
      - UNSW-NB15_features.csv
      - UNSW-NB15_LIST_EVENTS.csv
    - models
      - infer_phishing.py
      - infer_unsw_sample_terminl.py
      - infer_unsw_sample.py
      - network_model_unsw_xgb.joblib
      - network_model_unsw.joblib
      - phishing_model.joblib
    - README.md
    - requirements.txt
    - train
      - dev.ipynb
      - train_network_model_RF.py
      - train_network_model_XGB.py
      - train_phishing_model.py


🚀 4. Usage Summary
Intrusion Detection

Upload a CSV with the 42 features → model outputs "attack" or "normal" per row.

Phishing Detector

Paste an email → model outputs "phish" or "legit" along with probability scores.