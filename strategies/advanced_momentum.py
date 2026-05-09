import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.strategy import Strategy


class AdvancedMomentumStrategy(Strategy):
    """
    Momentum-style rule using spread between short and long average closes (past window only,
    excluding the current bar for the averages — same convention as the original project).
    """

    def __init__(
        self,
        short_window: int = 50,
        long_window: int = 200,
        entry_threshold: float = 0.01,
        exit_threshold: float = 0.005,
        base_trade_size: int = 100,
        force_trade_threshold: float = 0.005,
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.base_trade_size = base_trade_size
        self.force_trade_threshold = force_trade_threshold
        self.last_signal = None
        self.max_buy_signal = (0.0, None, None)

    def on_bar(self, time, row, data: pd.DataFrame):
        idx = data.index.get_indexer_for([time])[0]
        if idx < self.long_window:
            return None

        short_ma = data["Close"].iloc[idx - self.short_window : idx].mean()
        long_ma = data["Close"].iloc[idx - self.long_window : idx].mean()
        pct_diff = (short_ma - long_ma) / long_ma

        if pct_diff > 0 and pct_diff > self.max_buy_signal[0]:
            self.max_buy_signal = (float(pct_diff), time, row)

        if pct_diff > self.entry_threshold and self.last_signal != "buy":
            self.last_signal = "buy"
            return {
                "action": "buy",
                "size": self.base_trade_size,
                "stop_loss": float(row["Close"]) * (1 - 0.02),
                "take_profit": float(row["Close"]) * (1 + 0.04),
            }
        if pct_diff < -self.exit_threshold and self.last_signal != "sell":
            self.last_signal = "sell"
            return {"action": "sell", "reason": "Exit threshold reached"}
        return None

    def optimize_parameters(self, data, initial_cash, commission, trade_size, random_seed=42):
        """Grid search on the same timeline (in-sample). See backtesting.evaluation module."""
        best_return = -np.inf
        best_params = {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "base_trade_size": self.base_trade_size,
        }
        candidate_entry = [0.005, 0.01, 0.015]
        candidate_exit = [0.0025, 0.005, 0.0075]
        candidate_trade_size = [50, 100, 150]

        for et in candidate_entry:
            for ex in candidate_exit:
                for ts in candidate_trade_size:
                    test_strategy = AdvancedMomentumStrategy(
                        short_window=self.short_window,
                        long_window=self.long_window,
                        entry_threshold=et,
                        exit_threshold=ex,
                        base_trade_size=ts,
                        force_trade_threshold=self.force_trade_threshold,
                    )
                    engine = BacktestEngine(
                        data,
                        test_strategy,
                        initial_cash=initial_cash,
                        commission=commission,
                        trade_size=ts,
                        slippage_pct=0.001,
                        random_seed=random_seed,
                        allow_force_trade=False,
                    )
                    performance = engine.run()
                    ret = performance["Total Return"]
                    if ret > best_return:
                        best_return = ret
                        best_params = {"entry_threshold": et, "exit_threshold": ex, "base_trade_size": ts}
        self.entry_threshold = best_params["entry_threshold"]
        self.exit_threshold = best_params["exit_threshold"]
        self.base_trade_size = best_params["base_trade_size"]
        return best_params
