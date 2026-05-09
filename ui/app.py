# ui/app.py
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

config_path = os.path.join(parent_dir, "config", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

st.set_page_config(page_title="Backtest & execution simulator", layout="wide")

data_file_rel = config.get("historical_data", "examples/sample_ohlcv.csv")
data_file = os.path.abspath(os.path.join(parent_dir, data_file_rel))
if not os.path.exists(data_file):
    st.warning(f"Data file not found: {data_file}")
else:
    st.caption(f"Data file: `{data_file}`")

from backtesting.comparison import run_baseline_comparison
from backtesting.engine import BacktestEngine
from backtesting.evaluation import WALK_FORWARD_NOT_IMPLEMENTED
from data_ingestion.csv_loader import load_historical_data
from ml_models.ml_trading_model import adjust_strategy_with_predictions, train_lstm_model
from strategies.advanced_momentum import AdvancedMomentumStrategy

st.title("Backtesting and execution simulation")
st.markdown(
    """
This UI drives a **bar-level execution simulator** (commissions, slippage, stop / take-profit) and
optional **baseline strategy comparison**. Machine learning is **optional** and not required for
core backtests.

**Not financial advice.** Outputs are for research and engineering evaluation only.
"""
)

if WALK_FORWARD_NOT_IMPLEMENTED:
    st.info(
        "Walk-forward / rolling validation is **not implemented**. Parameter grids and LSTM fitting "
        "on the same timeline as the backtest are **in-sample** and can look better than true "
        "out-of-sample performance. See `backtesting/evaluation.py` and the README."
    )

st.sidebar.header("Data")
ticker = st.sidebar.text_input("Ticker label (for ingestion only)", value=config.get("ticker", "AAPL"))
data_file_input = st.sidebar.text_input("Historical CSV path", value=data_file_rel)
if not os.path.isabs(data_file_input):
    data_file_input = os.path.abspath(os.path.join(parent_dir, data_file_input))

st.sidebar.header("Execution parameters")
short_window = st.sidebar.number_input("Advanced strategy: short MA window", min_value=2, value=50)
long_window = st.sidebar.number_input("Advanced strategy: long MA window", min_value=3, value=200)
base_trade_size = st.sidebar.number_input("Trade size (shares)", min_value=1, value=int(config.get("trade_size", 100)))
initial_cash = st.sidebar.number_input("Initial cash", value=float(config.get("initial_cash", 100_000)))
commission = st.sidebar.number_input("Commission rate (per side, e.g. 0.001 = 0.1%)", value=float(config.get("commission", 0.001)))
slippage_pct = st.sidebar.number_input("Slippage band (+/- fraction)", min_value=0.0, value=0.001, format="%.4f")
random_seed = st.sidebar.number_input("Random seed (slippage)", value=42, step=1)
entry_threshold = st.sidebar.slider("Entry threshold (fraction)", min_value=0.001, max_value=0.05, value=float(config.get("entry_threshold", 0.01)))
exit_threshold = st.sidebar.slider("Exit threshold (fraction)", min_value=0.001, max_value=0.05, value=float(config.get("exit_threshold", 0.005)))

st.sidebar.header("Optional ML (LSTM)")
st.sidebar.markdown(
    """
LSTM training here is **experimental**. It does not establish predictive edge; see module docstring
in `ml_models/ml_trading_model.py` for limitations (non-stationarity, leakage risk in scaling, no
walk-forward design).
"""
)
run_lstm = st.sidebar.checkbox("Adjust strategy parameters with LSTM (slow, requires TensorFlow)", value=False)

use_optimization = st.sidebar.checkbox("In-sample parameter grid (advanced strategy only)", value=False)

if st.sidebar.button("Reset cached CSV (config path)"):
    if os.path.exists(data_file):
        os.remove(data_file)
        st.sidebar.success("Removed file at config path.")
    else:
        st.sidebar.info("Nothing to delete at config path.")

st.sidebar.header("Actions")
if st.sidebar.button("Fetch & enrich (Alpha Vantage / Kafka)"):
    st.write("Ingestion job for ticker:", ticker)
    import subprocess

    subprocess.run(
        [sys.executable, os.path.join(parent_dir, "data_ingestion", "producer.py"), "--ticker", ticker],
        cwd=parent_dir,
    )
    st.success("Producer finished (check logs if APIs or Kafka are unavailable).")

if st.sidebar.button("Run baseline comparison (buy & hold, SMA cross, advanced)"):
    data = load_historical_data(data_file_input)
    st.write("Bars loaded:", data.shape[0])
    comp = run_baseline_comparison(
        data,
        initial_cash=initial_cash,
        commission=commission,
        trade_size=base_trade_size,
        slippage_pct=slippage_pct,
        random_seed=int(random_seed),
        advanced_short=int(short_window),
        advanced_long=int(long_window),
    )
    rows = []
    for name, perf in comp.items():
        rows.append(
            {
                "Strategy": name,
                "Final cash": round(perf["Final Cash"], 2),
                "Total return": round(perf["Total Return"] * 100, 4),
                "Trades": perf["Number of Trades"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

if st.sidebar.button("Run advanced-strategy backtest"):
    data = load_historical_data(data_file_input)
    st.write("Bars loaded:", data.shape[0])

    if long_window <= short_window:
        st.error("Long window must be greater than short window.")
        st.stop()

    strategy = AdvancedMomentumStrategy(
        short_window=int(short_window),
        long_window=int(long_window),
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        base_trade_size=int(base_trade_size),
    )

    if use_optimization:
        st.write("Running in-sample grid search (same timeline as backtest).")
        opt_params = strategy.optimize_parameters(data, initial_cash, commission, base_trade_size, random_seed=int(random_seed))
        st.json(opt_params)

    if run_lstm:
        st.write("Training LSTM (illustrative; see limitations in `ml_models/ml_trading_model.py`).")
        model, scaler = train_lstm_model(data_file_input, look_back=min(60, max(10, len(data) // 4)), epochs=3, batch_size=32)
        if model is not None and scaler is not None:
            strategy = adjust_strategy_with_predictions(strategy, model, scaler, data, look_back=min(60, max(10, len(data) // 4)))
            st.json(
                {
                    "entry_threshold": strategy.entry_threshold,
                    "exit_threshold": strategy.exit_threshold,
                    "base_trade_size": strategy.base_trade_size,
                }
            )

    engine = BacktestEngine(
        data,
        strategy,
        initial_cash=initial_cash,
        commission=commission,
        trade_size=strategy.base_trade_size,
        slippage_pct=slippage_pct,
        random_seed=int(random_seed),
        allow_force_trade=False,
    )
    performance = engine.run()

    st.subheader("Results (one simulated path)")
    st.metric("Final cash", f"${performance['Final Cash']:,.2f}")
    st.metric("Total return", f"{performance['Total Return'] * 100:.4f}%")
    st.metric("Closed trades", performance["Number of Trades"])

    trades = performance["Trades"]
    if trades:
        trade_df = pd.DataFrame(
            [
                {
                    "Entry Time": t.entry_time,
                    "Entry Price": t.entry_price,
                    "Exit Time": t.exit_time,
                    "Exit Price": t.exit_price,
                    "P/L": t.profit(),
                    "Exit reason": t.exit_reason,
                    "Log": str(t.log),
                }
                for t in trades
                if t.is_closed()
            ]
        )
        st.dataframe(trade_df, use_container_width=True)

        pnl = [t.profit() for t in trades if t.is_closed()]
        fig, ax = plt.subplots()
        ax.plot(range(1, len(pnl) + 1), pnl, marker="o", color="C0", label="Trade P/L")
        ax.set_title("P/L per closed trade (simulated)")
        ax.set_xlabel("Trade index")
        ax.set_ylabel("P/L (currency)")
        ax.legend()
        st.pyplot(fig)
