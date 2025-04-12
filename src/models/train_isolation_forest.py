import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# File paths
SENTIMENT_FEATURES = "data/processed/features_sentiment.csv"
BLOCKCHAIN_FEATURES = "data/processed/features_blockchain.csv"
MODEL_PATH = "models/isolation_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/isolation_forest_features.txt"

# Define the feature set used for training
FEATURES = ["sentiment", "day", "month", "weekday", "avg_gas_used", "total_value", "txn_count"]

def train_isolation_forest(df):
    print("[*] Training Isolation Forest...")

    X = df[FEATURES].copy().dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    return model, scaler

def main():
    print("🌲 Training Isolation Forest Model...")

    # Load datasets
    df_sentiment = pd.read_csv(SENTIMENT_FEATURES, parse_dates=["created_at"])
    df_blockchain = pd.read_csv(BLOCKCHAIN_FEATURES, parse_dates=["hour"])

    # Round 'created_at' to hourly and remove timezone if present
    df_sentiment["hour"] = df_sentiment["created_at"].dt.floor("h").dt.tz_localize(None)
    df_sentiment["day"] = df_sentiment["created_at"].dt.day
    df_sentiment["month"] = df_sentiment["created_at"].dt.month
    df_sentiment["weekday"] = df_sentiment["created_at"].dt.weekday

    df_blockchain["hour"] = df_blockchain["hour"].dt.tz_localize(None)

    print("📊 Sentiment Hours:", df_sentiment["hour"].dropna().unique())
    print("🔗 Blockchain Hours:", df_blockchain["hour"].dropna().unique())

    # Merge on hour
    df = pd.merge(df_sentiment, df_blockchain, on="hour", how="inner")
    print(f"[+] Loaded merged dataset with shape: {df.shape}")

    if df.empty:
        print("[!] Merged dataset is empty. Check timestamps and try again.")
        return

    # Train model
    model, scaler = train_isolation_forest(df)

    # Save model, scaler, and feature list
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Save feature list for evaluation
    with open(FEATURES_PATH, "w") as f:
        for feat in FEATURES:
            f.write(f"{feat}\n")

    print(f"[+] Model saved to {MODEL_PATH}")
    print(f"[+] Scaler saved to {SCALER_PATH}")
    print(f"[+] Feature list saved to {FEATURES_PATH}")

if __name__ == "__main__":
    main()
