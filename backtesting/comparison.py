"""Run the same dataset through baseline strategies for side-by-side evaluation."""

from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestEngine
from strategies.advanced_momentum import AdvancedMomentumStrategy
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.sma_crossover import SMACrossoverStrategy


def run_baseline_comparison(
    data: pd.DataFrame,
    *,
    initial_cash: float = 100_000,
    commission: float = 0.001,
    trade_size: int = 100,
    slippage_pct: float = 0.001,
    random_seed: int | None = 42,
    sma_fast: int = 5,
    sma_slow: int = 10,
    advanced_short: int = 50,
    advanced_long: int = 200,
) -> dict[str, dict]:
    """
    Returns performance dicts keyed by strategy name. Each engine uses a fresh strategy instance
    and the same execution parameters so differences reflect signal logic, not simulator settings.
    """
    n = len(data)
    short_w = max(2, min(advanced_short, max(n // 6, 2)))
    long_w = max(short_w + 1, min(advanced_long, max(n // 3, short_w + 2)))
    if long_w >= n:
        long_w = min(short_w + 2, n - 2) if n > short_w + 2 else short_w + 1

    specs: list[tuple[str, object]] = [
        ("buy_and_hold", BuyAndHoldStrategy(trade_size=trade_size)),
        (
            "sma_crossover",
            SMACrossoverStrategy(fast_window=sma_fast, slow_window=sma_slow, trade_size=trade_size),
        ),
        (
            "advanced_momentum",
            AdvancedMomentumStrategy(
                short_window=short_w,
                long_window=long_w,
                base_trade_size=trade_size,
            ),
        ),
    ]
    out: dict[str, dict] = {}
    for name, strat in specs:
        engine = BacktestEngine(
            data.copy(),
            strat,
            initial_cash=initial_cash,
            commission=commission,
            trade_size=trade_size,
            slippage_pct=slippage_pct,
            random_seed=random_seed,
            allow_force_trade=False,
        )
        out[name] = engine.run()
    return out
