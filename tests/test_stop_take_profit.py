import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.strategy import Strategy


class OneShotBuy(Strategy):
    def __init__(self, size: int, stop: float | None, take: float | None):
        self.size = size
        self.stop = stop
        self.take = take
        self._sent = False

    def on_bar(self, time, row, data):
        if self._sent:
            return None
        self._sent = True
        return {
            "action": "buy",
            "size": self.size,
            "stop_loss": self.stop,
            "take_profit": self.take,
        }


def test_stop_loss_triggers_before_strategy_on_same_bar():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 100.0, 100.0],
            "Low": [100.0, 90.0, 100.0],
            "Close": [100.0, 95.0, 100.0],
            "Volume": [1, 1, 1],
        },
        index=idx,
    )
    eng = BacktestEngine(
        df,
        OneShotBuy(size=10, stop=95.0, take=None),
        initial_cash=100_000,
        commission=0.0,
        slippage_pct=0.0,
        random_seed=0,
    )
    perf = eng.run()
    assert len(perf["Trades"]) == 1
    tr = perf["Trades"][0]
    assert tr.is_closed()
    assert tr.exit_reason == "stop_loss"
    assert np.isclose(tr.exit_price, 95.0)


def test_take_profit_triggers_on_high():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 112.0, 100.0],
            "Low": [100.0, 100.0, 100.0],
            "Close": [100.0, 101.0, 100.0],
            "Volume": [1, 1, 1],
        },
        index=idx,
    )
    eng = BacktestEngine(
        df,
        OneShotBuy(size=10, stop=None, take=110.0),
        initial_cash=100_000,
        commission=0.0,
        slippage_pct=0.0,
        random_seed=0,
    )
    perf = eng.run()
    tr = perf["Trades"][0]
    assert tr.exit_reason == "take_profit"
    assert np.isclose(tr.exit_price, 110.0)


def test_stop_loss_takes_precedence_when_both_touch_same_bar():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [100.0, 120.0],
            "Low": [100.0, 90.0],
            "Close": [100.0, 100.0],
            "Volume": [1, 1],
        },
        index=idx,
    )
    eng = BacktestEngine(
        df,
        OneShotBuy(size=10, stop=95.0, take=110.0),
        initial_cash=100_000,
        commission=0.0,
        slippage_pct=0.0,
        random_seed=0,
    )
    perf = eng.run()
    assert perf["Trades"][0].exit_reason == "stop_loss"
