# AI-Driven-Fraud-Detection-using-Blockchain-and-Twitter-Sentiment-Data
📌 Overview
This project presents an AI-powered anomaly detection system that combines blockchain transaction data and real-time Twitter sentiment to identify potentially fraudulent activity. By leveraging unsupervised learning (Isolation Forest) and deep learning (Autoencoder), this system enables scalable fraud detection using public, trustless data sources.

💡 Problem Statement
With the explosive growth of decentralized finance (DeFi) and cryptocurrencies, fraudulent transactions and market manipulation are becoming increasingly common. Traditional systems often fail to detect subtle patterns across diverse data sources like user sentiment and on-chain metrics.

This project aims to:

Detect anomalies in blockchain transactions that may indicate fraud or manipulation.

Correlate unusual activity with shifts in public sentiment from Twitter data.

Build an automated, real-time fraud detection pipeline using publicly available data.

🎯 Objectives
Build a full pipeline to collect, clean, engineer, and merge Twitter + blockchain data.

Train anomaly detection models (Isolation Forest, Autoencoder) using engineered features.

Visualize anomaly scores and flag high-risk time windows.

Support real-time extension for future deployment.

📁 Project Structure
bash
Copy
Edit
fraud_detection_project/
│
├── data/                # Raw and processed data
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/            # Scripts for data collection and preprocessing
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Training and evaluation scripts
│   └── utils/           # Helpers, constants, logging, etc.
│
├── notebooks/           # Jupyter notebooks for exploration (optional)
├── models/              # Saved models and scalers
├── main.py              # Orchestration script for entire pipeline
└── README.md
🧾 Data Sources
📊 Blockchain Data (Etherscan)
Ethereum transaction history via Etherscan API

Extracted fields: value, gasUsed, timestamp

🐦 Twitter Data (Twitter API v2)
Real-time tweets fetched with OAuth2 Bearer Token

Query keywords (e.g., “bitcoin”, “crypto”)

Extracted fields: text, created_at

🔧 Pipeline Components
1. Data Collection
Fetches tweets and Ethereum transactions over a time window

Rate-limit handling + caching implemented

Outputs:

data/raw/tweets.csv

data/raw/blockchain.csv

2. Data Preprocessing
Sentiment data:

Convert created_at to datetime

Drop nulls

Blockchain data:

Convert timestamps, ensure gasUsed and value fields are numeric

Outputs:

data/processed/clean_tweets.csv

data/processed/clean_blockchain.csv

3. Feature Engineering
Sentiment:

Sentiment polarity (TextBlob)

Time-based features: hour, day, weekday, month

Blockchain:

Average gasUsed, total value, transaction count per hour

4. Model Training
Isolation Forest:

Unsupervised ML model for outlier detection

Uses combined sentiment and blockchain features

Autoencoder:

Keras-based deep neural network

Learns to reconstruct normal patterns; flags deviations

5. Evaluation & Visualization
Merges hourly features and predicts anomalies

Outputs final anomalies.csv

Plots Autoencoder reconstruction errors vs time

🔍 Use Case & Value
✅ Fraud Detection in DeFi
Identifies suspicious spikes in value, gas fees, or negative sentiment aligned with unusual blockchain activity.

✅ Early Warning System for Hacks or Scams
Spikes in tweet negativity and gas fees could indicate flash loan attacks, rug pulls, or scam promotions.

✅ Investor Sentiment Analytics
Detects when public sentiment shifts significantly compared to blockchain trends.

✅ RegTech & Compliance
Can aid regulatory teams in flagging suspicious behavior across wallets.

📈 Example Outputs
📄 data/processed/anomalies.csv
Contains model predictions with timestamps, anomaly scores, and classifications.

📊 Autoencoder Visualization
Shows high reconstruction error as an anomaly signal. 

🛠️ Technologies Used
Tool	Purpose
Python	Core programming language
Pandas, NumPy	Data wrangling
Scikit-learn	Isolation Forest, preprocessing
TensorFlow/Keras	Autoencoder training
Matplotlib	Visualizations
TextBlob	Sentiment analysis
Requests, dotenv	API interaction
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
Set up the environment:

bash
Copy
Edit
pip install -r requirements.txt
Configure .env file:

ini
Copy
Edit
TWITTER_BEARER_TOKEN=your_token_here
ETHERSCAN_API_KEY=your_api_key_here
Run the pipeline:

bash
Copy
Edit
PYTHONPATH=. python3 src/main.py
PYTHONPATH=. python3 src/models/evaluate.py
📚 References
Isolation Forest: Liu et al., “Isolation Forest,” ICDM 2008

Autoencoders: Hinton & Salakhutdinov, “Reducing the Dimensionality of Data with Neural Networks,” Science 2006

TextBlob: https://textblob.readthedocs.io

Etherscan API: https://docs.etherscan.io

Twitter API v2: https://developer.twitter.com/en/docs

🙋‍♂️ Author
Shivam Singh
Master's in Computer Science
Illinois Institute of Technology
LinkedIn • GitHub

