import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "CryptoSentimentBot"
    return r

def fetch_tweets_for_window(query, start_time, end_time, max_results=100):
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": f"{query} lang:en -is:retweet",
        "start_time": start_time.isoformat("T") + "Z",
        "end_time": end_time.isoformat("T") + "Z",
        "max_results": max_results,
        "tweet.fields": "created_at,text"
    }

    response = requests.get(url, auth=bearer_oauth, params=params)
    if response.status_code != 200:
        print(f"[!] Failed for window {start_time} → {end_time}: {response.text}")
        return []

    tweets = response.json().get("data", [])
    return [{
        "text": tweet["text"],
        "created_at": tweet["created_at"]
    } for tweet in tweets]

def collect_tweets_over_time_range(output_path, query="crypto", hours_back=24, interval_hours=1):
    """
    Collect tweets in hourly intervals over the past `hours_back` hours.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours_back)

    all_tweets = []

    print(f"🕓 Collecting tweets from {start_time} to {end_time} every {interval_hours}h")

    for dt in pd.date_range(start=start_time, end=end_time, freq=f"{interval_hours}H"):
        window_start = dt
        window_end = dt + timedelta(hours=interval_hours)

        print(f"🔍 Fetching from {window_start} → {window_end}")
        batch = fetch_tweets_for_window(query, window_start, window_end)
        all_tweets.extend(batch)

    df = pd.DataFrame(all_tweets)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Collected {len(df)} tweets over time range to {output_path}")

# Run as script
if __name__ == "__main__":
    collect_tweets_over_time_range(
        output_path="data/raw/tweets.csv",
        query="bitcoin",
        hours_back=72,           # Last 3 days
        interval_hours=1         # Hourly batches
    )
