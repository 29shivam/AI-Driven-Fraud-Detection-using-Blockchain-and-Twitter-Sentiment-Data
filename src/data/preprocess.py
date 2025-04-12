import pandas as pd
import re
import string

def clean_tweet_text(text):
    # Remove mentions, URLs, hashtags, and punctuation
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip().lower()

def preprocess_sentiment_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df['clean_text'] = df['text'].apply(clean_tweet_text)
    df.to_csv(output_path, index=False)
    print(f"Cleaned sentiment data saved to {output_path}")

import pandas as pd

def preprocess_blockchain_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check for necessary columns
    required_cols = ["hash", "from", "to", "value", "gasused", "timestamp"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[!] Missing columns in blockchain data: {missing}")
        return

    # Convert numeric fields
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["gasused"] = pd.to_numeric(df["gasused"], errors="coerce")

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows with any nulls in required columns
    df = df.dropna(subset=required_cols)

    # Select only relevant columns
    df = df[required_cols]

    # Save cleaned file
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned blockchain data saved to {output_path}")

