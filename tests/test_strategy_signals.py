import pandas as pd

from strategies.sma_crossover import SMACrossoverStrategy


def test_sma_crossover_emits_buy_on_cross_up():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    closes = [10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 12.0, 13.0]
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [100] * len(closes),
        },
        index=idx,
    )
    strat = SMACrossoverStrategy(fast_window=2, slow_window=3, trade_size=5)
    signals = [strat.on_bar(t, df.loc[t], df) for t in idx]
    assert any(s is not None and s.get("action") == "buy" for s in signals)
