#!/usr/bin/env python3
import os
import time
import json
import argparse
import random
import requests
import pandas as pd
import yaml
from datetime import datetime, timedelta
from kafka import KafkaProducer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

# --- Helper Functions ---

def load_config():
    """
    Loads config.yaml, then overrides any placeholders with environment variables.
    This ensures you can keep placeholders in config.yaml while reading
    real secrets from environment (e.g., Docker, .env, cloud platforms).
    """
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    cfg_path = os.path.join(root, "config", "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Override placeholders with environment variables if available
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", config["alpha_vantage"]["key"])
    news_key = os.getenv("NEWS_API_KEY", config["news_api"]["key"])
    kafka_bootstrap = os.getenv("BOOTSTRAP_SERVERS", config["kafka"]["bootstrap_servers"])
    kafka_user = os.getenv("KAFKA_USERNAME", config["kafka"]["sasl_plain_username"])
    kafka_pass = os.getenv("KAFKA_PASSWORD", config["kafka"]["sasl_plain_password"])

    # Inject them back into the config structure
    config["alpha_vantage"]["key"] = av_key
    config["news_api"]["key"] = news_key
    config["kafka"]["bootstrap_servers"] = kafka_bootstrap
    config["kafka"]["sasl_plain_username"] = kafka_user
    config["kafka"]["sasl_plain_password"] = kafka_pass

    return config

def fetch_alpha_vantage_data(symbol, alpha_vantage_config):
    """
    Fetches intraday data from Alpha Vantage using the params in alpha_vantage_config.
    """
    params = {
        "function": alpha_vantage_config["function"],
        "symbol": symbol,
        "interval": alpha_vantage_config["interval"],
        "outputsize": alpha_vantage_config["outputsize"],
        "apikey": alpha_vantage_config["key"]
    }
    response = requests.get("https://www.alphavantage.co/query", params=params)
    data = response.json()
    ts_key = f"Time Series ({params['interval']})"
    if ts_key not in data:
        raise Exception("Error fetching data from Alpha Vantage. Check API key and parameters.")
    df = pd.DataFrame.from_dict(data[ts_key], orient='index')
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])
    return df

def add_RSI(df, window=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def fetch_news_sentiment(symbol, news_api_config):
    """
    Fetches recent news articles from NewsAPI and computes an average sentiment score.
    """
    params = {
        "q": symbol,
        "apiKey": news_api_config["key"],
        "language": "en",
        "pageSize": 5
    }
    response = requests.get(news_api_config["url"], params=params)
    data = response.json()
    articles = data.get("articles", [])
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for article in articles:
        title = article.get("title", "")
        if title:
            score = analyzer.polarity_scores(title)["compound"]
            scores.append(score)
    return sum(scores)/len(scores) if scores else 0.0

def enrich_data(df, symbol, news_api_config):
    """
    Adds RSI and a single sentiment score to the DataFrame.
    """
    df = add_RSI(df)
    sentiment = fetch_news_sentiment(symbol, news_api_config)
    df["News_Sentiment"] = sentiment
    return df

def append_to_csv(df, csv_path):
    """
    Appends the DataFrame rows to CSV. Creates CSV with headers if it doesn't exist.
    """
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='w', header=True, index=True, index_label="Time")
    else:
        df.to_csv(csv_path, mode='a', header=False, index=True)

# --- Main Producer Code ---
def main():
    parser = argparse.ArgumentParser(description="Ingest & enrich historical data, then publish to Kafka and save to CSV.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol to fetch")
    args = parser.parse_args()

    # 1. Load config with environment variable overrides
    config = load_config()

    # 2. Determine which ticker to fetch
    ticker = args.ticker if args.ticker else config.get("ticker", "AAPL")

    # 3. Extract relevant sections from our config
    alpha_config = config["alpha_vantage"]
    news_api_config = config["news_api"]
    data_file_rel = config.get("historical_data", "data/historical_data.csv")
    kafka_conf = config["kafka"]

    # 4. Construct CSV path (project root)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    csv_path = os.path.abspath(os.path.join(parent_dir, data_file_rel))

    # 5. Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=kafka_conf["bootstrap_servers"],
        security_protocol=kafka_conf["security_protocol"],
        sasl_mechanism=kafka_conf["sasl_mechanism"],
        sasl_plain_username=kafka_conf["sasl_plain_username"],
        sasl_plain_password=kafka_conf["sasl_plain_password"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    # 6. Fetch alpha vantage data & enrich
    df = fetch_alpha_vantage_data(ticker, alpha_config)
    df = enrich_data(df, ticker, news_api_config)

    # 7. Publish each row to Kafka & append to CSV
    for idx, row in df.iterrows():
        message = {
            "Time": row.name.strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": ticker,
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"],
            "Volume": row["Volume"],
            "RSI": row["RSI"],
            "News_Sentiment": row["News_Sentiment"]
        }
        producer.send(kafka_conf["topic"], value=message)

        single_row_df = pd.DataFrame([row])
        append_to_csv(single_row_df, csv_path)
        time.sleep(0.1)

    producer.flush()
    print(f"Data ingestion complete. Data saved to {csv_path} and published to Kafka topic '{kafka_conf['topic']}'.")

if __name__ == "__main__":
    main()
