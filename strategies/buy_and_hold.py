from backtesting.strategy import Strategy


class BuyAndHoldStrategy(Strategy):
    """Enter once at the first bar, hold until the engine liquidates at the end of the series."""

    def __init__(self, trade_size: int = 100):
        self.trade_size = trade_size
        self._entered = False

    def on_bar(self, time, row, data):
        if self._entered:
            return None
        self._entered = True
        return {
            "action": "buy",
            "size": self.trade_size,
            "stop_loss": None,
            "take_profit": None,
        }
