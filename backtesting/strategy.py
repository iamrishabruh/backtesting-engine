class Strategy:
    def on_bar(self, time, row, data):
        raise NotImplementedError("Implement on_bar in your strategy subclass.")
