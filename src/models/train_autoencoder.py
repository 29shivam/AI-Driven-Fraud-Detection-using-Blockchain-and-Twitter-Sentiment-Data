import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import joblib

# Define file paths
SENTIMENT_FEATURES = "data/processed/features_sentiment.csv"
BLOCKCHAIN_FEATURES = "data/processed/features_blockchain.csv"
MODEL_OUTPUT = "models/autoencoder_model.h5"
SCALER_OUTPUT = "models/autoencoder_scaler.pkl"
FEATURES_OUTPUT = "models/autoencoder_features.txt"

Path("models").mkdir(exist_ok=True)

def load_and_prepare_data():
    df_sentiment = pd.read_csv(SENTIMENT_FEATURES, parse_dates=["created_at"])
    df_blockchain = pd.read_csv(BLOCKCHAIN_FEATURES, parse_dates=["hour"])

    # Round and remove timezone
    df_sentiment["hour"] = pd.to_datetime(df_sentiment["created_at"]).dt.floor("h").dt.tz_localize(None)
    df_blockchain["hour"] = pd.to_datetime(df_blockchain["hour"]).dt.tz_localize(None)

    df = pd.merge(df_sentiment, df_blockchain, on="hour", how="inner")
    print(f"[+] Merged data shape: {df.shape}")

    # ✅ Use the same feature set as during evaluation
    features = ["sentiment", "avg_gas_used", "total_value", "txn_count", "day", "month", "weekday"]

    # Ensure no missing data in selected features
    df = df.dropna(subset=features)
    df_selected = df[features]

    # Save feature list for evaluation consistency
    with open(FEATURES_OUTPUT, "w") as f:
        f.write("\n".join(features))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected)

    return X_scaled, scaler

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
    encoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(32, activation="relu")(encoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

def main():
    print("🤖 Training Autoencoder Model...")
    X, scaler = load_and_prepare_data()

    autoencoder = build_autoencoder(X.shape[1])
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_val, X_val))

    autoencoder.save(MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)

    print(f"[+] Autoencoder model saved to {MODEL_OUTPUT}")
    print(f"[+] Scaler saved to {SCALER_OUTPUT}")
    print(f"[+] Feature list saved to {FEATURES_OUTPUT}")

if __name__ == "__main__":
    main()
