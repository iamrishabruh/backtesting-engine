# Backtesting and execution simulation

## 1. What this is

This repository is a **backtesting and execution-simulation** toolkit. It loads historical OHLCV (and optional sentiment) data, generates **strategy signals** on a bar-by-bar timeline, and simulates **orders** with commissions, optional **slippage** (seeded for reproducibility), and **stop-loss / take-profit** handling using each bar’s high–low range.

The focus is **evaluation discipline**: explicit assumptions, baselines, and tests—not a story about predictive alpha.

## 2. What it is not

- **Not financial advice** and not a recommendation to buy or sell any instrument.
- **Not** a turnkey live trading system; there is no production broker integration or order-management guarantees.
- **Not** proof of future performance. Any metrics are path-dependent outputs of a simplified simulator.

## 3. Architecture

| Area | Role |
|------|------|
| `data_ingestion/` | CSV loading (`csv_loader.py`) and optional live-style ingestion (`producer.py`: APIs, enrichment, Kafka). |
| `strategies/` | Signal logic: `buy_and_hold`, `sma_crossover`, `advanced_momentum` (trend / threshold rule from the original project). |
| `backtesting/` | `BacktestEngine` (execution, risk exits), `Trade` accounting, `comparison` helpers, `evaluation` notes on validation scope. |
| `ml_models/` | **Optional** LSTM experiment (`ml_trading_model.py`) with documented limitations. |
| `ui/` | Streamlit front-end to run baselines and single-path backtests. |
| `tests/` | Pytest suite for bias checks, costs, risk exits, accounting, signals, reproducibility. |
| `examples/` | Small sample CSV for local runs and CI-style checks. |

## 4. Data ingestion

- **Primary path for evaluation**: place a CSV with columns `Time`, `Open`, `High`, `Low`, `Close`, `Volume`. Optional: `News_Sentiment`, `RSI`, etc. Use `data_ingestion.csv_loader.load_historical_data`.
- **Producer** (`data_ingestion/producer.py`): fetches intraday data (Alpha Vantage), enriches (e.g. sentiment), appends to the configured path, and can publish to Kafka. Requires credentials and network; failures there do not affect offline backtests on a local file.

Default config path: `config/config.yaml` → `historical_data` points at `examples/sample_ohlcv.csv` for an offline-friendly default.

## 5. Strategy engine

Strategies subclass `backtesting.strategy.Strategy` and implement `on_bar(time, row, data)` returning a dict such as `{"action": "buy", "size": 100, "stop_loss": ..., "take_profit": ...}` or `{"action": "sell", ...}`.

The engine walks bars in time order; with an open position it first checks **stop-loss / take-profit** against the current bar’s range, then applies the strategy signal.

## 6. Backtesting assumptions

- **Long-only** simulation in the current engine; one open position at a time.
- **Commissions** applied as a fraction on notional for buys and sells (see tests for formulas).
- **Slippage** is a symmetric random fraction of price around each fill, driven by a **NumPy `Generator`** with a user-set seed (default `42`) so repeated runs match when the seed matches.
- **Stop / take-profit**: if both could trigger on the same bar, **stop-loss is applied first** for longs (conservative ordering).
- **End of series**: any open position is closed at the **last bar’s close** (with slippage/commission).
- **“Force trade”** behavior from early prototypes is **disabled by default** (`allow_force_trade=False`) so empty signal streams do not fabricate trades.

## 7. ML forecasting module (optional)

`ml_models/ml_trading_model.py` can train a small LSTM on scaled features. **Read the module docstring before using it.** In short:

- A single chronological train/test split is **not** a full validation design.
- Fitting scalers on the pre-split sample can **leak information** relative to a strict rolling fit.
- Markets are **non-stationary**; out-of-sample generalization is not established by one backtest path.
- The UI can **adjust strategy knobs** from predictions; that path is explicitly experimental.

TensorFlow is listed in `requirements.txt` for environments that run training; CI installs it so imports resolve. For minimal installs you may split ML dependencies in your own fork.

## 8. Baseline comparisons

`backtesting.comparison.run_baseline_comparison` runs the same execution settings over:

1. **Buy and hold** — one entry at the first bar, held until the forced liquidation at the end.
2. **Simple moving average crossover** — fast vs slow SMA on closes through the current bar.
3. **Advanced momentum** — short vs long average spread with entry/exit thresholds (windows scale down on short series so the function can run on small fixtures).

Use this to compare **signal definitions** under shared costs, not to rank “best” strategies on one path alone.

## 9. Walk-forward validation

**Rolling / walk-forward evaluation is not implemented in code.** The module `backtesting/evaluation.py` describes why single-timeline grids and model fitting are **in-sample** and how a proper walk-forward design would differ. Treat any optimization or LSTM step in the UI as a research convenience, not an out-of-sample protocol.

## 10. How to run tests

```bash
pip install -r requirements.txt
python3 -m pytest -q
```

Tests cover: no-lookahead window usage for key strategies, commission and slippage math (with deterministic slippage), stop-loss and take-profit behavior, trade and cash accounting, signal generation, and reproducibility of baseline comparison on the sample CSV.

## 11. How to run the UI

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

Docker (same entrypoint as before):

```bash
docker build -t backtest-sim:latest .
docker run -p 8080:8080 -e PORT=8080 backtest-sim:latest
```

`docker-compose.yml` builds the image and mounts the repo; map host port **8080** to container **8080** and set `PORT=8080` if you use Compose.

## 12. Known limitations

- Single-path backtests do not characterize sampling uncertainty or regime changes.
- No short selling, partial fills, margin, borrow, or latency models.
- Kafka / News / Alpha Vantage ingestion depends on external services and keys.
- LSTM and in-sample optimization can **overstate** apparent edge; walk-forward and held-out protocols are not bundled.

---

For deployment automation, see `.github/workflows/deploy.yml` (optional Cloud Run flow). **CI** runs on every push and PR via `.github/workflows/ci.yml`.
