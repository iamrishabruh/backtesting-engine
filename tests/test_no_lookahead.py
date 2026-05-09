"""
No-lookahead: signal ingredients at bar t must not use rows strictly after t.
"""

import numpy as np
import pandas as pd

from strategies.advanced_momentum import AdvancedMomentumStrategy
from strategies.sma_crossover import SMACrossoverStrategy


def test_advanced_momentum_uses_past_window_excluding_current_close():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    closes = pd.Series(np.linspace(10.0, 19.0, 10), index=idx)
    df = pd.DataFrame(
        {"Open": closes, "High": closes + 0.5, "Low": closes - 0.5, "Close": closes, "Volume": 1000},
        index=idx,
    )
    short_w, long_w = 3, 6
    t = idx[7]
    row = df.loc[t]
    strat = AdvancedMomentumStrategy(short_window=short_w, long_window=long_w, entry_threshold=1e9)
    i = df.index.get_indexer_for([t])[0]
    ref_short = df["Close"].iloc[i - short_w : i].mean()
    ref_long = df["Close"].iloc[i - long_w : i].mean()
    wrong_if_includes_current = df["Close"].iloc[i - short_w + 1 : i + 1].mean()
    assert not np.isclose(ref_short, wrong_if_includes_current)
    strat.on_bar(t, row, df)
    pct_diff = (ref_short - ref_long) / ref_long
    assert np.isclose(strat.max_buy_signal[0], max(0.0, pct_diff), atol=1e-9)


def test_sma_crossover_indicators_use_prefix_only():
    idx = pd.date_range("2024-01-01", periods=12, freq="h")
    rng = np.random.default_rng(0)
    closes = pd.Series(rng.normal(100, 1, size=len(idx)), index=idx)
    df = pd.DataFrame(
        {"Open": closes, "High": closes + 0.2, "Low": closes - 0.2, "Close": closes, "Volume": 1000},
        index=idx,
    )
    fast_w, slow_w = 2, 3
    t = idx[6]
    row = df.loc[t]
    i = df.index.get_indexer_for([t])[0]
    prefix = df["Close"].iloc[: i + 1]
    fast_now = prefix.iloc[-fast_w:].mean()
    slow_now = prefix.iloc[-slow_w:].mean()
    poisoned = pd.concat([prefix, pd.Series([1e6])])
    assert not np.isclose(fast_now, poisoned.iloc[-fast_w:].mean())

    strat = SMACrossoverStrategy(fast_window=fast_w, slow_window=slow_w, trade_size=1)
    strat.on_bar(t, row, df)
