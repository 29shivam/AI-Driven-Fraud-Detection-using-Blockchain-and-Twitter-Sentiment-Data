# src/main.py

import os
import sys
from src.data.collect_blockchain import collect_blockchain_data
from src.data.collect_sentiment import collect_tweets_over_time_range
from src.data.preprocess import preprocess_sentiment_data, preprocess_blockchain_data

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# File paths
RAW_TWEETS = "data/raw/tweets.csv"
RAW_BLOCKCHAIN = "data/raw/blockchain.csv"
PROC_TWEETS = "data/processed/clean_tweets.csv"
PROC_BLOCKCHAIN = "data/processed/clean_blockchain.csv"

def main():
    print("🚀 Starting Fraud Detection Data Pipeline")

    # Step 1: Collect Twitter Sentiment Data (only if not already cached)
    if not os.path.exists(RAW_TWEETS):
        print("\n📡 Collecting sentiment data from Twitter...")
        collect_tweets_over_time_range(output_path=RAW_TWEETS, query="bitcoin")

    else:
        print("[✓] Skipping Twitter data collection – already cached.")

    # Step 2: Collect Blockchain Transaction Data
    print("⛓️  Collecting blockchain data from Etherscan...")
    collect_blockchain_data(output_path=RAW_BLOCKCHAIN)

    # Step 3: Preprocess Data
    print("\n🧹 Preprocessing sentiment data...")
    preprocess_sentiment_data(input_path=RAW_TWEETS, output_path=PROC_TWEETS)

    print("🧼 Preprocessing blockchain data...")
    preprocess_blockchain_data(input_path=RAW_BLOCKCHAIN, output_path=PROC_BLOCKCHAIN)

    print("\n✅ All data collected and preprocessed. Ready for feature engineering.")

if __name__ == "__main__":
    main()
