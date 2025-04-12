import pandas as pd
from textblob import TextBlob
from pathlib import Path

def add_sentiment_features(tweet_csv, output_csv):
    print("📊 Building features from sentiment data...")

    df = pd.read_csv(tweet_csv)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    # Drop rows with invalid datetime
    df = df.dropna(subset=['created_at'])

    # Add basic datetime features
    df['hour'] = df['created_at'].dt.hour
    df['weekday'] = df['created_at'].dt.weekday
    df['day'] = df['created_at'].dt.day
    df['month'] = df['created_at'].dt.month

    # Add sentiment polarity using TextBlob
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)

    # Save to file
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[+] Sentiment features saved to {output_csv}")


def add_blockchain_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # 🔧 Normalize column names to handle case sensitivity or whitespace issues
    df.columns = df.columns.str.strip().str.lower()

    print(f"[+] Blockchain columns: {list(df.columns)}")

    # ✅ Check required columns after normalization
    if "gasused" not in df.columns or "value" not in df.columns:
        print("[!] Required columns 'gasUsed' or 'value' are missing.")
        return

    df["gasused"] = pd.to_numeric(df["gasused"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Aggregate features hourly
    agg_df = df.groupby(df["timestamp"].dt.floor("h")).agg(
        avg_gas_used=("gasused", "mean"),
        total_value=("value", "sum"),
        txn_count=("hash", "count")
    ).reset_index().rename(columns={"timestamp": "hour"})

    agg_df.to_csv(output_path, index=False)
    print(f"[+] Blockchain features saved to {output_path}")



if __name__ == "__main__":
    # Define your input/output file paths
    TWEET_INPUT = "data/processed/clean_tweets.csv"
    BLOCKCHAIN_INPUT = "data/processed/clean_blockchain.csv"
    TWEET_OUTPUT = "data/processed/features_sentiment.csv"
    BLOCKCHAIN_OUTPUT = "data/processed/features_blockchain.csv"

    add_sentiment_features(TWEET_INPUT, TWEET_OUTPUT)
    add_blockchain_features(BLOCKCHAIN_INPUT, BLOCKCHAIN_OUTPUT)
