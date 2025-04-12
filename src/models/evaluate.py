import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# File paths
SENTIMENT_FEATURES = "data/processed/features_sentiment.csv"
BLOCKCHAIN_FEATURES = "data/processed/features_blockchain.csv"
IFOREST_MODEL_PATH = "models/isolation_forest_model.pkl"
IFOREST_SCALER_PATH = "models/scaler.pkl"
IFOREST_FEATURES_PATH = "models/isolation_forest_features.txt"
AUTOENCODER_MODEL_PATH = "models/autoencoder_model.h5"
AUTOENCODER_SCALER_PATH = "models/autoencoder_scaler.pkl"
OUTPUT_ANOMALIES = "data/processed/anomalies.csv"

def load_feature_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def load_and_merge_data():
    df_sentiment = pd.read_csv(SENTIMENT_FEATURES, parse_dates=["created_at"])
    df_blockchain = pd.read_csv(BLOCKCHAIN_FEATURES, parse_dates=["hour"])

    df_sentiment["hour"] = pd.to_datetime(df_sentiment["created_at"].dt.floor("h")).dt.tz_localize(None)
    df_sentiment["day"] = df_sentiment["created_at"].dt.day
    df_sentiment["month"] = df_sentiment["created_at"].dt.month
    df_sentiment["weekday"] = df_sentiment["created_at"].dt.weekday

    df = pd.merge(df_sentiment, df_blockchain, on="hour", how="inner")
    return df.dropna()

def evaluate_isolation_forest(df):
    print("🌲 Evaluating with Isolation Forest...")

    features = load_feature_list(IFOREST_FEATURES_PATH)  # 👈 load saved list
    X = df[features].copy()

    scaler = joblib.load(IFOREST_SCALER_PATH)
    model = joblib.load(IFOREST_MODEL_PATH)

    X_scaled = scaler.transform(X)
    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)
    return scores, preds

def evaluate_autoencoder(df):
    print("🤖 Evaluating with Autoencoder...")

    # ✅ Load exact feature list from training
    with open("models/autoencoder_features.txt", "r") as f:
        features = f.read().splitlines()

    X = df[features]

    scaler = joblib.load(AUTOENCODER_SCALER_PATH)
    model = load_model(AUTOENCODER_MODEL_PATH, compile=False)

    X_scaled = scaler.transform(X)
    X_pred = model.predict(X_scaled)

    reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)
    return reconstruction_error


def main():
    df = load_and_merge_data()

    # Evaluate both models
    iforest_scores, iforest_preds = evaluate_isolation_forest(df)
    ae_errors = evaluate_autoencoder(df)

    # Annotate
    df["iforest_score"] = iforest_scores
    df["iforest_anomaly"] = (iforest_preds == -1).astype(int)
    df["ae_error"] = ae_errors
    df["ae_anomaly"] = (ae_errors > np.percentile(ae_errors, 95)).astype(int)

    df.to_csv(OUTPUT_ANOMALIES, index=False)
    print(f"[+] Evaluation results saved to {OUTPUT_ANOMALIES}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["hour"], df["ae_error"], label="Autoencoder Error", color="orange")
    plt.axhline(np.percentile(ae_errors, 95), color="red", linestyle="--", label="95th percentile")
    plt.xticks(rotation=45)
    plt.title("Autoencoder Reconstruction Errors")
    plt.xlabel("Hour")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
