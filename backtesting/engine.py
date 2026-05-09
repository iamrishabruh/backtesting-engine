import numpy as np
import pandas as pd

from backtesting.trade import Trade


class BacktestEngine:
    """
    Bar-by-bar execution simulator: commissions, optional slippage (seeded for reproducibility),
    and stop-loss / take-profit checks using the current bar's High/Low range.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        initial_cash: float = 100_000,
        commission: float = 0.001,
        trade_size: int = 100,
        slippage_pct: float = 0.001,
        random_seed: int | None = 42,
        allow_force_trade: bool = False,
    ):
        self.data = data.sort_index()
        self.strategy = strategy
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0
        self.trade_size = trade_size
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.allow_force_trade = allow_force_trade
        self.trades: list[Trade] = []
        self.logs: list[tuple] = []
        self._rng = np.random.default_rng(random_seed)

    def simulate_slippage(self, price: float) -> float:
        if self.slippage_pct <= 0:
            return float(price)
        delta = self._rng.uniform(-self.slippage_pct, self.slippage_pct)
        return float(price) * (1.0 + delta)

    def _intrabar_exit_price(self, row: pd.Series, trade: Trade) -> tuple[float | None, str | None]:
        """
        Long-only risk exits. If both stop and take-profit could trigger in the same bar,
        stop-loss takes precedence (conservative for longs).
        """
        low = float(row["Low"])
        high = float(row["High"])
        if trade.stop_loss is not None and low <= trade.stop_loss:
            return float(trade.stop_loss), "stop_loss"
        if trade.take_profit is not None and high >= trade.take_profit:
            return float(trade.take_profit), "take_profit"
        return None, None

    def run(self):
        for current_time, row in self.data.iterrows():
            if self.position > 0 and self.trades:
                active = self.trades[-1]
                if not active.is_closed():
                    exit_px, reason = self._intrabar_exit_price(row, active)
                    if exit_px is not None:
                        self.execute_sell(current_time, exit_px, {"action": "sell", "reason": reason})

            signal = self.strategy.on_bar(current_time, row, self.data)
            self.logs.append((current_time, f"Signal: {signal}"))
            if signal and signal.get("action") == "buy" and self.position == 0:
                self.execute_buy(current_time, float(row["Close"]), signal)
            elif signal and signal.get("action") == "sell" and self.position > 0:
                self.execute_sell(current_time, float(row["Close"]), signal)

        if self.position > 0:
            last_time = self.data.index[-1]
            last_price = float(self.data.iloc[-1]["Close"])
            self.execute_sell(last_time, last_price, {"action": "sell", "reason": "End of Data"})

        if self.allow_force_trade and len(self.trades) == 0:
            self._maybe_force_trade()

        return self.get_performance()

    def _maybe_force_trade(self):
        if not hasattr(self.strategy, "max_buy_signal") or not hasattr(self.strategy, "force_trade_threshold"):
            self.logs.append((self.data.index[-1], "Force trade requested but strategy has no threshold fields."))
            return
        max_signal, force_time, force_row = self.strategy.max_buy_signal
        thr = self.strategy.force_trade_threshold
        if max_signal >= thr and force_time is not None:
            self.logs.append(
                (force_time, f"No trade executed normally. Forcing BUY with signal {max_signal:.4f}")
            )
            self.execute_buy(
                force_time,
                float(force_row["Close"]),
                {
                    "action": "buy",
                    "size": self.trade_size,
                    "stop_loss": float(force_row["Close"]) * (1 - 0.02),
                    "take_profit": float(force_row["Close"]) * (1 + 0.04),
                    "forced": True,
                },
            )
            if len(self.trades) > 0:
                last_time = self.data.index[-1]
                last_price = float(self.data.iloc[-1]["Close"])
                self.execute_sell(last_time, last_price, {"action": "sell", "reason": "Forced trade at end"})
                self.logs.append((last_time, "Forced trade executed due to insufficient signals."))
            else:
                self.logs.append((self.data.index[-1], "Forced BUY failed (insufficient cash)."))
        else:
            self.logs.append((self.data.index[-1], "No forced trade executed: best signal below threshold."))

    def execute_buy(self, time, price, signal):
        exec_price = self.simulate_slippage(price)
        size = int(signal.get("size", self.trade_size))
        cost = exec_price * size * (1 + self.commission)
        if self.cash >= cost:
            self.position = size
            self.cash -= cost
            trade = Trade(
                entry_time=time,
                entry_price=exec_price,
                size=size,
                stop_loss=signal.get("stop_loss"),
                take_profit=signal.get("take_profit"),
            )
            trade.update_log(f"Bought at {exec_price}")
            self.trades.append(trade)
            self.logs.append((time, f"Executed BUY at {exec_price}"))
        else:
            self.logs.append((time, "Insufficient cash for BUY"))

    def execute_sell(self, time, price, signal):
        exec_price = self.simulate_slippage(price)
        proceeds = exec_price * self.position * (1 - self.commission)
        self.cash += proceeds
        if not self.trades:
            self.logs.append((time, "SELL ignored: no open trade"))
            return
        trade = self.trades[-1]
        reason = signal.get("reason") or signal.get("action")
        trade.close(time, exec_price, reason=reason)
        self.position = 0
        self.logs.append((time, f"Executed SELL at {exec_price} ({reason})"))

    def equity(self, mark_price: float) -> float:
        """Cash plus marked-to-market position value (no exit costs applied)."""
        return self.cash + self.position * float(mark_price)

    def get_performance(self):
        total_return = (self.cash - self.initial_cash) / self.initial_cash
        profits = [t.profit() for t in self.trades if t.is_closed()]
        num_trades = len(profits)
        avg_profit = float(np.mean(profits)) if profits else 0.0
        max_profit = float(np.max(profits)) if profits else 0.0
        max_loss = float(np.min(profits)) if profits else 0.0
        return {
            "Final Cash": self.cash,
            "Total Return": total_return,
            "Number of Trades": num_trades,
            "Average Profit per Trade": avg_profit,
            "Max Profit": max_profit,
            "Max Loss": max_loss,
            "Trades": self.trades,
            "Logs": self.logs,
        }
