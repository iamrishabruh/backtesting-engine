import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from strategies.buy_and_hold import BuyAndHoldStrategy


def test_round_trip_trade_updates_cash_and_closes_trade():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 12.0],
            "High": [10.0, 12.0],
            "Low": [10.0, 12.0],
            "Close": [10.0, 12.0],
            "Volume": [100, 100],
        },
        index=idx,
    )
    size = 50
    initial = 10_000.0
    eng = BacktestEngine(
        df,
        BuyAndHoldStrategy(trade_size=size),
        initial_cash=initial,
        commission=0.0,
        slippage_pct=0.0,
        random_seed=0,
    )
    perf = eng.run()
    tr = perf["Trades"][0]
    assert tr.is_closed()
    assert np.isclose(tr.profit(), (12.0 - 10.0) * size)
    assert eng.position == 0
    assert np.isclose(perf["Final Cash"], initial + tr.profit())
