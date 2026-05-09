import numpy as np

from backtesting.comparison import run_baseline_comparison
from data_ingestion.csv_loader import load_historical_data


def test_baseline_comparison_reproducible_on_fixture(sample_csv_path):
    data = load_historical_data(sample_csv_path)
    a = run_baseline_comparison(data, random_seed=7, sma_fast=2, sma_slow=3)
    b = run_baseline_comparison(data, random_seed=7, sma_fast=2, sma_slow=3)
    for name in a:
        assert np.isclose(a[name]["Final Cash"], b[name]["Final Cash"])
        assert a[name]["Number of Trades"] == b[name]["Number of Trades"]
