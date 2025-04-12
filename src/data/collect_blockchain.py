# src/data/collect_blockchain.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_eth_transactions(address, api_key):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
    response = requests.get(url)
    return response.json()  # return full JSON response

def collect_blockchain_data(output_path="data/raw/blockchain.csv"):
    ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
    sample_address = "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"

    data = fetch_eth_transactions(sample_address, ETHERSCAN_API_KEY)

    if isinstance(data, dict) and data.get("status") == "1":
        txs = data["result"]
        df = pd.DataFrame(txs)

        # Convert timestamp and numeric columns
        df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["gasUsed"] = pd.to_numeric(df["gasUsed"], errors="coerce")

        # Keep relevant columns
        df = df[["hash", "from", "to", "value", "gasUsed", "timeStamp"]]

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"[+] Blockchain data saved to {output_path} with {len(df)} transactions")
    else:
        print(f"[!] Failed to fetch blockchain data: {data}")

if __name__ == "__main__":
    collect_blockchain_data()
