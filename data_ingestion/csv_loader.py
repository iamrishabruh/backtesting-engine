import pandas as pd

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Load a historical CSV with a Time column (or datetime index). Optional columns such as
    RSI or News_Sentiment are preserved when present for optional ML features.
    """
    df = pd.read_csv(filepath, parse_dates=["Time"])
    if "Time" not in df.columns:
        raise ValueError("CSV must include a 'Time' column")
    df = df.set_index("Time").sort_index()
    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    for col in REQUIRED_OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "News_Sentiment" not in df.columns:
        df["News_Sentiment"] = 0.0
    else:
        df["News_Sentiment"] = pd.to_numeric(df["News_Sentiment"], errors="coerce").fillna(0.0)
    return df
