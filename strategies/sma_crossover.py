import pandas as pd

from backtesting.strategy import Strategy


class SMACrossoverStrategy(Strategy):
    """
    Classic dual moving-average crossover using only closes up to and including the current bar.
    Buy when fast SMA crosses above slow SMA; sell when it crosses below.
    """

    def __init__(
        self,
        fast_window: int = 5,
        slow_window: int = 10,
        trade_size: int = 100,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.trade_size = trade_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._position_side: str | None = None  # "long" or None

    def on_bar(self, time, row, data: pd.DataFrame):
        idx = data.index.get_indexer_for([time])[0]
        if idx < self.slow_window:
            return None

        closes = data["Close"].iloc[: idx + 1]
        fast_now = closes.iloc[-self.fast_window :].mean()
        slow_now = closes.iloc[-self.slow_window :].mean()
        fast_prev = closes.iloc[-self.fast_window - 1 : -1].mean()
        slow_prev = closes.iloc[-self.slow_window - 1 : -1].mean()

        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        close_px = float(row["Close"])
        if cross_up and self._position_side is None:
            self._position_side = "long"
            sig = {
                "action": "buy",
                "size": self.trade_size,
            }
            if self.stop_loss_pct is not None:
                sig["stop_loss"] = close_px * (1 - self.stop_loss_pct)
            if self.take_profit_pct is not None:
                sig["take_profit"] = close_px * (1 + self.take_profit_pct)
            return sig
        if cross_down and self._position_side == "long":
            self._position_side = None
            return {"action": "sell", "reason": "sma_crossunder"}
        return None
