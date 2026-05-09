import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from strategies.buy_and_hold import BuyAndHoldStrategy


def _two_bar_frame(price_buy: float, price_sell: float):
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    return pd.DataFrame(
        {
            "Open": [price_buy, price_sell],
            "High": [price_buy, price_sell],
            "Low": [price_buy, price_sell],
            "Close": [price_buy, price_sell],
            "Volume": [1000, 1000],
        },
        index=idx,
    )


def test_commission_on_buy_and_sell():
    size = 100
    commission = 0.01
    initial = 1_000_000.0
    df = _two_bar_frame(100.0, 100.0)
    eng = BacktestEngine(
        df,
        BuyAndHoldStrategy(trade_size=size),
        initial_cash=initial,
        commission=commission,
        slippage_pct=0.0,
        random_seed=0,
    )
    eng.run()
    buy_cost = 100.0 * size * (1 + commission)
    sell_proceeds = 100.0 * size * (1 - commission)
    assert np.isclose(eng.cash, initial - buy_cost + sell_proceeds)


def test_slippage_is_deterministic_with_seed():
    size = 10
    df = _two_bar_frame(50.0, 60.0)
    r1 = BacktestEngine(
        df,
        BuyAndHoldStrategy(trade_size=size),
        initial_cash=100_000,
        commission=0.0,
        slippage_pct=0.01,
        random_seed=123,
    ).run()["Final Cash"]
    r2 = BacktestEngine(
        df,
        BuyAndHoldStrategy(trade_size=size),
        initial_cash=100_000,
        commission=0.0,
        slippage_pct=0.01,
        random_seed=123,
    ).run()["Final Cash"]
    assert r1 == r2
