class Trade:
    def __init__(self, entry_time, entry_price, size, stop_loss=None, take_profit=None):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.log = []

    def update_log(self, message, log_time=None):
        self.log.append((log_time if log_time is not None else self.entry_time, message))

    def close(self, time, price, reason=None):
        self.exit_time = time
        self.exit_price = price
        self.exit_reason = reason
        self.update_log(f"Closed at {price}", log_time=time)

    def profit(self):
        if self.is_closed():
            return (self.exit_price - self.entry_price) * self.size
        return 0.0

    def is_closed(self):
        return self.exit_time is not None

    def __repr__(self):
        return (
            f"Trade(entry_time={self.entry_time}, entry_price={self.entry_price}, "
            f"exit_time={self.exit_time}, exit_price={self.exit_price}, profit={self.profit()}, log={self.log})"
        )
